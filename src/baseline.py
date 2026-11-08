# 2627-LTXLDL-Airbnb-Nhom12
"""Phương pháp đối chứng KHÔNG dùng LLM: từ điển cảm xúc + từ khoá khía cạnh + từ dừng nhận diện ngôn ngữ.

Chạy riêng:  python src/baseline.py --config configs/London.yaml

Dùng đúng cùng mẫu review với llm.py (chon_mau) và cùng định dạng đầu ra, để evaluate.py so sánh công bằng.
Đầu ra: data/processed/baseline_results.csv

Cách làm (không huấn luyện nên không bị rò rỉ nhãn tay):
  1. Chia câu; trong mỗi câu đếm từ khen/chê, có xử lý phủ định ("not clean" -> chê).
  2. Cảm xúc cả review: tỷ lệ chê/(khen+chê) <= 0.15 -> positive, >= 0.6 -> negative, còn lại -> mixed.
  3. Khía cạnh: câu chứa từ khoá khía cạnh thì lấy chiều khen/chê của câu đó (riêng 'độ ồn' và 'an toàn'
     dùng từ vựng chuyên biệt vì "noise" tự thân đã mang nghĩa chê, còn "no noise" là khen).
  4. Ngôn ngữ: đếm từ dừng của en/fr/es/de/it/pt; chữ không phải Latin -> "other".
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:  # chạy "python src/baseline.py" hoặc import dạng "src.baseline" đều được
    from src.llm import chon_mau, doc_cau_hinh, ASPECT_NOISE, ASPECT_SAFETY
    from src.qa import PROJECT_PATH
except ImportError:
    from llm import chon_mau, doc_cau_hinh, ASPECT_NOISE, ASPECT_SAFETY
    from qa import PROJECT_PATH

POS = set("""great good excellent amazing wonderful perfect lovely fantastic comfortable clean spotless cozy cosy
beautiful friendly helpful nice superb brilliant recommend recommended recommending enjoyed enjoy loved love
stunning spacious tidy convenient welcoming responsive kind pleasant fabulous awesome delightful immaculate
charming gem happy easy smooth seamless quick fast flawless outstanding gorgeous bright quiet peaceful safe
super genial parfait propre agréable magnifique excelente perfecto limpio genial encantador sehr schön sauber""".split())
NEG = set("""bad terrible awful horrible dirty filthy disappointing disappointed poor worst uncomfortable noisy
smelly smell mouldy mold mould broken cold damp rude unhelpful overpriced misleading cramped small old
unsafe sketchy dangerous scary problem problems issue issues unfortunately avoid rubbish disgusting
stain stains dusty leak leaking unresponsive cancelled canceled complaint annoying uncomfortable loud
bugs cockroach cockroaches mice infested thinwalls sale sucio horrible terrible schmutzig laut""".split())
NEGATORS = {"not", "no", "never", "without", "hardly", "barely", "nothing", "none", "neither", "nor", "pas", "sin", "nicht", "kein"}

# Từ vựng chuyên biệt theo khía cạnh nhạy cảm (nghĩa phụ thuộc phủ định)
NOISE_POS = {"quiet", "peaceful", "silent", "tranquil", "calm", "soundproof", "soundproofed"}
NOISE_NEG = {"noisy", "noise", "noises", "loud", "thinwalls", "disturbance", "disturbed", "sirens", "siren", "construction"}
SAFE_POS = {"safe", "secure", "securely", "safely"}
SAFE_NEG = {"unsafe", "sketchy", "dangerous", "crime", "robbed", "robbery", "burglary", "theft", "scary", "shady", "stolen"}

# Từ khoá nhận diện câu có nhắc tới khía cạnh khác (nhãn tiếng Việt khớp llm.aspects trong config)
ASPECT_KW = {
    "vị trí": r"locat\w*|area|neighbou?rhood|close to|near|walking distance|tube|station|transport|central|metro",
    "sạch sẽ": r"clean\w*|dirty|spotless|filth\w*|dust\w*|smell\w*|hygien\w*|tidy|stain\w*|mou?ld\w*",
    "chủ nhà": r"host|owner|landlord|communicat\w*|check-?in|responsive|welcom\w*|hospitalit\w*",
    "giá trị": r"value|price\w*|expensive|cheap|overpriced|worth|money|cost\w*",
    "tiện nghi": r"kitchen|wifi|wi-fi|bed\w*|shower|bathroom|heating|amenit\w*|equipped|facilit\w*|\btv\b|washing machine|towels?",
}
ASPECT_KW[ASPECT_NOISE] = "|".join(sorted(NOISE_POS | NOISE_NEG | {"thin walls", "street noise"}))
ASPECT_KW[ASPECT_SAFETY] = "|".join(sorted(SAFE_POS | SAFE_NEG | {"safety", "neighbourhood felt", "neighborhood felt"}))

STOP = {
    "en": set("the and was is were are with for very had have this that our you but not they from great".split()),
    "fr": set("le la les et est était très pour avec nous un une des dans que pas sur appartement".split()),
    "es": set("el la los las y es muy para con una un del que en nos estaba apartamento".split()),
    "de": set("der die das und ist war sehr mit für wir ein eine nicht den dem auf wohnung".split()),
    "it": set("il lo la gli e è era molto per con noi un una che non del appartamento".split()),
    "pt": set("o os as e é foi muito para com nós um uma que não do da apartamento".split()),
}
TOKEN = re.compile(r"[^\W\d_]+(?:'[a-z]+)?", re.UNICODE)


def chuan_hoa(text):
    t = str(text).lower().replace("’", "'")
    for cu, moi in (("thin walls", "thinwalls"), ("street noise", "noisy"), ("n't", " not"), ("felt unsafe", "unsafe")):
        t = t.replace(cu, moi)
    return t


def dem_khen_che(tokens, khen, che):
    """Đếm (khen, chê) trong 1 câu; từ phủ định xuất hiện trong 3 từ liền trước thì đảo chiều."""
    k = c = 0
    for i, w in enumerate(tokens):
        la_khen, la_che = w in khen, w in che
        if not (la_khen or la_che):
            continue
        if any(t in NEGATORS for t in tokens[max(0, i - 3):i]):
            la_khen, la_che = la_che, la_khen
        k += la_khen
        c += la_che
    return k, c


def doan_ngon_ngu(text):
    chu = [ch for ch in str(text) if ch.isalpha()]
    if chu and sum(ord(ch) > 0x024F for ch in chu) / len(chu) > 0.3:
        return "other"
    tokens = TOKEN.findall(str(text).lower())
    diem = {ng: sum(w in sw for w in tokens) for ng, sw in STOP.items()}
    ng, d = max(diem.items(), key=lambda kv: kv[1])
    return ng if d >= 2 else "other"


def du_doan(text, aspects):
    """Một review -> {sentiment, language, aspects_pos, aspects_neg}."""
    cau_list = [c for c in re.split(r"[.!?;\n]+|\bbut\b|\bhowever\b", chuan_hoa(text)) if c.strip()]
    khen_tong = che_tong = 0
    pos, neg = set(), set()
    for cau in cau_list:
        tokens = TOKEN.findall(cau)
        k, c = dem_khen_che(tokens, POS, NEG)
        khen_tong += k
        che_tong += c
        for asp in aspects:
            kw = ASPECT_KW.get(asp)
            if not kw or not re.search(rf"\b(?:{kw})\b", cau):
                continue
            if asp == ASPECT_NOISE:
                kk, cc = dem_khen_che(tokens, NOISE_POS, NOISE_NEG)
            elif asp == ASPECT_SAFETY:
                kk, cc = dem_khen_che(tokens, SAFE_POS, SAFE_NEG)
            else:
                kk, cc = k, c
            if cc > kk:
                neg.add(asp)
            elif kk > cc:
                pos.add(asp)
    tong = khen_tong + che_tong
    ty_le_che = che_tong / tong if tong else 0.0
    sentiment = "positive" if ty_le_che <= 0.15 else "negative" if ty_le_che >= 0.6 else "mixed"
    return {"sentiment": sentiment, "language": doan_ngon_ngu(text),
            "aspects_pos": sorted(pos), "aspects_neg": sorted(neg)}


# ----------------------------------------------------------------------------- CHẠY CHÍNH
def run_baseline(config_path, root=PROJECT_PATH):
    root = Path(root)
    config, llm_cfg = doc_cau_hinh(config_path)
    aspects = list(llm_cfg["aspects"])
    mau = chon_mau(config, llm_cfg, root)
    dong = []
    for r in mau.itertuples():
        d = du_doan(r.comments_clean, aspects)
        dong.append({"review_id": r.review_id, "listing_id": r.listing_id, "sentiment": d["sentiment"],
                     "language": d["language"], "aspects_pos": "|".join(d["aspects_pos"]),
                     "aspects_neg": "|".join(d["aspects_neg"]),
                     "noise_neg": int(ASPECT_NOISE in d["aspects_neg"]),
                     "safety_neg": int(ASPECT_SAFETY in d["aspects_neg"]), "model": "baseline-lexicon"})
    kq = pd.DataFrame(dong)
    p = root / "data" / "processed" / "baseline_results.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    kq.to_csv(p, index=False, encoding="utf-8")
    print(f"[BASELINE] {len(kq)} review -> {p.relative_to(root)}")
    return kq


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Đối chứng không LLM")
    ap.add_argument("--config", default="configs/London.yaml")
    run_baseline(ap.parse_args().config)
