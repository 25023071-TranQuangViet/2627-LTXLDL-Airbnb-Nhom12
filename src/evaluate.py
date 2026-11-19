
# 2627-LTXLDL-Airbnb-Nhom12
"""Đánh giá chất lượng và chi phí của LLM so với phương pháp đối chứng, theo Bài 11 mục 5.

Chạy riêng:
    python src/evaluate.py --config configs/London.yaml                   # chấm điểm
    python src/evaluate.py --config configs/London.yaml --make-template   # tạo tệp mẫu + hướng dẫn gán nhãn tay

Bộ nhãn tay đóng vai trò bộ dữ liệu kiểm thử: không có nó, nhận xét về chất lượng chỉ là cảm nhận.
Tệp data/samples/hand_labels.csv (>= 100 dòng), mỗi dòng một review, các danh sách nối bằng "|":
    review_id | sentiment (positive/mixed/negative) | aspects_positive | aspects_negative | language (ISO 639-1, "und")
(cột thừa như "comments" được bỏ qua; khía cạnh dùng mã tiếng Anh ASPECTS của llm.py)

Điểm (Bài 11): cảm xúc, ngôn ngữ = tỷ lệ đúng (accuracy); khía cạnh = F1 (cộng số nhãn đúng, thừa, thiếu của cả
bộ rồi mới tính), tập khía cạnh gộp khen + chê, có thêm F1 riêng cho khía cạnh bị chê. Thêm cho Nhóm C: F1 phát hiện
"chê ồn" và "chê an toàn". Cả baseline lẫn LLM đều ghi khen/chê riêng (aspects_pos / aspects_neg) nên đều có điểm "bị chê".
Ngôn ngữ được chuẩn hoá trước khi so: "en-GB" -> "en"; "other" (đối chứng) và "und" -> "und".

Đầu ra: reports/llm_eval.csv            điểm của baseline và LLM (dạng dài: metric, system, score, n)
        reports/llm_eval_history.csv    mỗi lần chấm một dòng: mô hình, phiên bản prompt, điểm (để so khi sửa prompt)
        reports/llm_error_analysis.csv  từng review sai: trường, dự đoán, nhãn tay, thừa, thiếu, nội dung
        reports/llm_hau_kiem.csv        số dòng LLM bị cờ bởi các quy tắc hậu kiểm
        reports/llm_cost_estimate.csv   chi phí theo token và số ngày ở gói miễn phí
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:  # chạy "python src/evaluate.py" hoặc import dạng "src.evaluate" đều được
    from src.llm import doc_cau_hinh, tai_reviews_cleaned, ASPECTS, SENTIMENTS, PROMPT_VERSION
    from src.qa import PROJECT_PATH
except ImportError:
    from llm import doc_cau_hinh, tai_reviews_cleaned, ASPECTS, SENTIMENTS, PROMPT_VERSION
    from qa import PROJECT_PATH

GAN_NHAN = """# Hướng dẫn gán nhãn tay (Bài 11 mục 5)

Hai người gán ĐỘC LẬP; chỗ bất đồng thì thảo luận và sửa hướng dẫn này, rồi gán lại.

- sentiment: positive = chủ yếu khen; negative = chủ yếu chê; mixed = có cả khen lẫn chê rõ ràng, hoặc không
  nghiêng hẳn về bên nào. Review không có nội dung: mixed.
- aspects_positive / aspects_negative: chỉ ghi khía cạnh THỰC SỰ được nhắc tới. Danh mục cố định:
  location (cả khu phố, giao thông), cleanliness, host, value, noise, amenities (wifi, giường, bếp, phòng tắm...), safety.
- Mỗi khía cạnh chỉ thuộc MỘT danh sách, theo ý chính của review (không ghi cùng một khía cạnh ở cả hai).
- noise: yên tĩnh = khen noise; ồn ào = chê noise. safety: thấy an toàn = khen; thấy không an toàn = chê.
- Nhận xét chung về chỗ ở (không nêu cụ thể) thì KHÔNG gán amenities.
- language: mã ISO 639-1 của ngôn ngữ chính (en, es, fr...); "und" nếu chỉ có emoji hoặc không xác định được.
- Các danh sách nối bằng dấu | ; để trống nếu không có.
"""

CAC_DIEM = {"cam_xuc": "accuracy cảm xúc", "khia_canh_f1": "F1 khía cạnh (khen + chê)",
            "bi_che_f1": "F1 khía cạnh bị chê", "on_bi_che_f1": "F1 phát hiện chê ồn",
            "an_toan_bi_che_f1": "F1 phát hiện chê an toàn", "ngon_ngu": "accuracy ngôn ngữ"}

COT_NHAN_TAY = {"review_id", "sentiment", "aspects_positive", "aspects_negative", "language"}


# ----------------------------------------------------------------------------- tiện ích chuẩn hoá
def _ma(s):
    """Mã review về chuỗi: khớp nhau dù tệp nào đọc ra số nguyên, số thực hay chuỗi."""
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _la_dung(s):
    """Cột cờ đọc ra bool hoặc chuỗi 'True'/'False' đều dùng được."""
    return s.astype(str).str.strip().str.lower().eq("true")


def chuan_ngon_ngu(v):
    """'en-GB' -> 'en'; rỗng, 'other' (đối chứng không nhận ra) và 'und' -> 'und'."""
    v = str(v).strip().lower().replace("_", "-").split("-")[0]
    return "und" if v in ("", "nan", "other", "und") else v


def tach(s):
    return [a for a in str(s).split("|") if a and a != "nan"]


def _tim_cot(d, ung_vien, ten_tep):
    for c in ung_vien:
        if c in d.columns:
            return c
    raise ValueError(f"{ten_tep}: thiếu cột {' hoặc '.join(ung_vien)}")


# ----------------------------------------------------------------------------- hàm chấm điểm (Bài 11 mục 5)
def ti_le_dung(du_doan, nhan_tay, truong):
    dung = [du_doan[truong].get(ma) == g[truong] for ma, g in nhan_tay.items()]
    return sum(dung) / len(dung)


def f1(du_doan, nhan_tay, truong):
    """F1 gộp: cộng nhãn đúng (tp), thừa (fp), thiếu (fn) của cả bộ rồi mới tính."""
    tp = fp = fn = 0
    for ma, g in nhan_tay.items():
        p, t = du_doan[truong].get(ma, set()), set(g[truong])
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan


def doc_nhan_tay(p):
    """Đọc và KIỂM TRA bộ nhãn tay: sai danh mục thì dừng và nói rõ."""
    d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    thieu = COT_NHAN_TAY - set(d.columns)
    if thieu:
        raise ValueError(f"{p.name}: thiếu cột {sorted(thieu)}")
    d["review_id"] = _ma(d["review_id"])
    d = d.drop_duplicates("review_id")
    d["sentiment"] = d["sentiment"].str.strip().str.lower()
    loi = sorted(set(d["sentiment"]) - set(SENTIMENTS))
    if loi:
        raise ValueError(f"{p.name}: sentiment chỉ nhận {SENTIMENTS}; thấy giá trị lạ/rỗng: {loi}")
    gold, ca_hai = {}, []
    for r in d.itertuples():
        pos, neg = tach(r.aspects_positive), tach(r.aspects_negative)
        la = sorted((set(pos) | set(neg)) - set(ASPECTS))
        if la:
            raise ValueError(f"{p.name}: review {r.review_id} có khía cạnh ngoài danh mục {ASPECTS}: {la}")
        ngon_ngu = chuan_ngon_ngu(r.language)
        if ngon_ngu != "und" and not re.fullmatch(r"[a-z]{2}", ngon_ngu):
            raise ValueError(f"{p.name}: review {r.review_id} có language '{r.language}' (cần mã ISO 639-1 hoặc 'und')")
        if set(pos) & set(neg):
            ca_hai.append(r.review_id)
        gold[r.review_id] = {"sentiment": r.sentiment, "aspects_positive": pos, "aspects_negative": neg,
                             "language": ngon_ngu,
                             "aspects": pos + neg,
                             "noise_neg": [a for a in neg if a == "noise"],
                             "safety_neg": [a for a in neg if a == "safety"]}
    if ca_hai:
        print(f"[EVAL] CẢNH BÁO: {len(ca_hai)} review ghi cùng khía cạnh ở cả khen và chê "
              f"(trái hướng dẫn gán nhãn): {ca_hai[:5]}")
    return gold


def doc_du_doan(p, ids):
    """CSV kết quả -> {trường: {mã review: giá trị}} cho các review trong ids (cùng dạng với nhãn tay).

    llm.py và baseline.py ghi khía cạnh ở cột aspects_pos / aspects_neg (chấp nhận cả aspects_positive / aspects_negative).
    """
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d["review_id"] = _ma(d["review_id"])
    d = d.drop_duplicates("review_id")
    d = d[d["review_id"].isin(ids)].set_index("review_id")
    c_pos = _tim_cot(d, ("aspects_pos", "aspects_positive"), p.name)
    c_neg = _tim_cot(d, ("aspects_neg", "aspects_negative"), p.name)
    pos = {m: set(tach(v)) for m, v in d[c_pos].items()}
    neg = {m: set(tach(v)) for m, v in d[c_neg].items()}
    return {"sentiment": d["sentiment"].str.strip().str.lower().to_dict(),
            "language": d["language"].map(chuan_ngon_ngu).to_dict(),
            "aspects_positive": pos, "aspects_negative": neg,
            "aspects": {m: pos[m] | neg[m] for m in d.index},
            "noise_neg": {m: {a for a in neg[m] if a == "noise"} for m in d.index},
            "safety_neg": {m: {a for a in neg[m] if a == "safety"} for m in d.index}}


def cot_diem(du, gold):
    return {"cam_xuc": ti_le_dung(du, gold, "sentiment"), "khia_canh_f1": f1(du, gold, "aspects"),
            "bi_che_f1": f1(du, gold, "aspects_negative"),
            "on_bi_che_f1": f1(du, gold, "noise_neg"), "an_toan_bi_che_f1": f1(du, gold, "safety_neg"),
            "ngon_ngu": ti_le_dung(du, gold, "language")}


def liet_ke_loi(he, gold, text):
    """Từng review sai (Bài 11 'LLM sai ở đâu?'): trường, dự đoán, nhãn tay, thừa, thiếu."""
    cot = ["system", "review_id", "truong", "du_doan", "nhan_tay", "thua", "thieu", "comments"]
    dong = []
    for ten, du in he.items():
        for ma, g in gold.items():
            for truong in ("sentiment", "language"):
                doan = du[truong].get(ma)
                if doan != g[truong]:
                    dong.append({"system": ten, "review_id": ma, "truong": truong, "du_doan": doan,
                                 "nhan_tay": g[truong], "thua": "", "thieu": "", "comments": text.get(ma, "")})
            for truong in ("aspects_positive", "aspects_negative", "aspects"):
                doan, nhan = set(du[truong].get(ma, set())), set(g[truong])
                if doan != nhan:
                    dong.append({"system": ten, "review_id": ma, "truong": truong,
                                 "du_doan": "|".join(sorted(doan)), "nhan_tay": "|".join(sorted(nhan)),
                                 "thua": "|".join(sorted(doan - nhan)), "thieu": "|".join(sorted(nhan - doan)),
                                 "comments": text.get(ma, "")})
    return pd.DataFrame(dong, columns=cot)


def dem_review_thanh_pho(root, snap):
    p = root / "data" / "raw" / snap / "reviews.csv.gz"
    if not p.exists():
        return None
    return int(sum(len(c) for c in pd.read_csv(p, compression="gzip", usecols=[0], chunksize=500_000)))


def uoc_tinh_chi_phi(cfg, ket_qua_llm, n_mau, n_tp):
    """Bài 11 mục 5: đo token trung bình mỗi review rồi nhân với số review định xử lý."""
    ok = ket_qua_llm[["in_tok", "out_tok"]].apply(pd.to_numeric, errors="coerce").dropna()
    if ok.empty:
        return None
    vao, ra = float(ok["in_tok"].mean()), float(ok["out_tok"].mean())
    g_vao, g_ra = float(cfg["price_input_per_m_usd"]), float(cfg["price_output_per_m_usd"])
    dong = []
    for ten, n in (("mẫu của nhóm", n_mau), ("100.000 review", 100_000), ("toàn bộ review thành phố", n_tp)):
        if not n:
            continue
        dong.append({"pham_vi": ten, "so_review": int(n), "token_vao_tb": round(vao, 1), "token_ra_tb": round(ra, 1),
                     "chi_phi_usd": round(n * (vao * g_vao + ra * g_ra) / 1e6, 2),
                     "ngay_o_goi_mien_phi": round(n / float(cfg["free_calls_per_day"]), 1),
                     "gia_vao_usd_moi_trieu": g_vao, "gia_ra_usd_moi_trieu": g_ra, "model": cfg["model"]})
    return pd.DataFrame(dong)


def do_dong_thuan(gold_a, gold_b):
    """Hai người gán độc lập (Bài 11): tỷ lệ hai người giống nhau ở từng trường."""
    chung = sorted(set(gold_a) & set(gold_b))
    if not chung:
        return None
    kh = lambda g, m, t: sorted(g[m][t]) if isinstance(g[m][t], list) else g[m][t]
    return pd.DataFrame([{"truong": t, "n": len(chung),
                          "ty_le_giong_nhau": round(float(np.mean([kh(gold_a, m, t) == kh(gold_b, m, t) for m in chung])), 3)}
                         for t in ("sentiment", "aspects_positive", "aspects_negative", "language")])


# ----------------------------------------------------------------------------- DỮ LIỆU REVIEW (qua llm.py)
def tai_review(cfg, snapshots, root):
    """review_cleaned của qa.py qua llm.tai_reviews_cleaned, thêm cột text và chuẩn hoá kiểu dữ liệu."""
    rv = tai_reviews_cleaned({**cfg, "snapshots": snapshots}, root).copy()   # llm.py cần cfg["snapshots"] cho nhánh dự phòng
    rv["review_id"] = _ma(rv["review_id"])
    for c in ("usable_for_llm", "flag_LLM", "flag_id_duplicates"):
        rv[c] = _la_dung(rv[c])
    rv["text"] = rv["comments_fixed"].fillna("").astype(str)
    return rv


def ma_nhan_tay(rv, proc):
    """Mã review được giao gán nhãn tay: cột flag_human nếu có, không thì cột 'Nhãn tay' của reviews-loc.csv."""
    if "flag_human" in rv.columns:
        return set(rv.loc[_la_dung(rv["flag_human"]), "review_id"])
    p = proc / "reviews-loc.csv"
    if p.exists():
        d = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
        if "Nhãn tay" in d.columns:
            return set(_ma(d.loc[d["Nhãn tay"].notna(), "review_id"]))
    raise FileNotFoundError("Không xác định được review nhãn tay: thiếu cột flag_human và reviews-loc.csv")


# ----------------------------------------------------------------------------- CHẠY CHÍNH
def run_evaluate(config_path, root=PROJECT_PATH, make_template=False):
    root = Path(root)
    p_cfg = Path(config_path)
    if not p_cfg.exists() and not p_cfg.is_absolute():
        p_cfg = root / p_cfg
    config = yaml.safe_load(p_cfg.read_text(encoding="utf-8"))
    cfg = doc_cau_hinh(config_path)                      # llm.py trả về MỘT dict (khối llm: đã điền mặc định)
    ecfg = config.get("evaluate", {}) or {}
    snapshots = config["snapshots"]
    samples, proc, rep = root / "data" / "samples", root / "data" / "processed", root / "reports"

    if make_template:
        rv = tai_review(cfg, snapshots, root)
        mau_llm = rv[rv["usable_for_llm"] & rv["flag_LLM"] & ~rv["flag_id_duplicates"]]   # đúng mẫu llm.py sẽ chạy
        ids_tay = ma_nhan_tay(rv, proc)
        mau = (mau_llm[mau_llm["review_id"].isin(ids_tay)].drop_duplicates("review_id")
               [["review_id", "text"]].rename(columns={"text": "comments"}))
        if len(mau) < len(ids_tay):
            print(f"[EVAL] CẢNH BÁO: {len(ids_tay) - len(mau)}/{len(ids_tay)} review nhãn tay không nằm trong mẫu LLM dùng được "
                  "(rỗng, trùng hoặc không có ở mốc nào): không thể chấm các review này")
        for cot in ("sentiment", "aspects_positive", "aspects_negative", "language"):
            mau[cot] = ""
        samples.mkdir(parents=True, exist_ok=True)
        mau.to_csv(samples / "hand_labels_template.csv", index=False, encoding="utf-8-sig")
        (samples / "huong_dan_gan_nhan.md").write_text(GAN_NHAN, encoding="utf-8")
        print(f"[EVAL] tạo data/samples/hand_labels_template.csv ({len(mau)} review) và huong_dan_gan_nhan.md. "
              "Hai người gán độc lập, thống nhất rồi lưu thành data/samples/hand_labels.csv")
        return None

    p_gold = root / ecfg.get("hand_labels", "data/samples/hand_labels.csv")
    if not p_gold.exists():
        print(f"[EVAL] CHƯA CHẤM: không thấy nhãn tay tại {p_gold} (chạy --make-template rồi gán nhãn)")
        return None
    gold = doc_nhan_tay(p_gold)
    if len(gold) < 100:
        print(f"[EVAL] CẢNH BÁO: chỉ có {len(gold)} nhãn tay, đề bài yêu cầu >= 100")
    p_gold2 = samples / "hand_labels_2.csv"
    if p_gold2.exists():
        dt_ = do_dong_thuan(gold, doc_nhan_tay(p_gold2))
        if dt_ is not None:
            rep.mkdir(parents=True, exist_ok=True)
            dt_.to_csv(rep / "gan_nhan_dong_thuan.csv", index=False, encoding="utf-8-sig")
            print("[EVAL] mức đồng thuận hai người gán:\n" + dt_.to_string(index=False))

    nguon = {"baseline": proc / "baseline_results.csv", "llm": proc / "llm_results.csv"}
    ids = set(gold)
    for ten, p in list(nguon.items()):
        if not p.exists():
            print(f"[EVAL] thiếu {p.name}: bỏ qua hệ thống {ten}")
            del nguon[ten]
            continue
        ids &= set(_ma(pd.read_csv(p, dtype=str, usecols=["review_id"])["review_id"]))
    if not nguon or not ids:
        print("[EVAL] CHƯA CHẤM: chưa có review nào vừa có nhãn tay vừa có dự đoán của mọi hệ thống")
        return None
    gold = {m: g for m, g in gold.items() if m in ids}
    print(f"[EVAL] chấm trên {len(gold)} review có nhãn tay và có dự đoán của {' & '.join(nguon)}")

    he = {ten: doc_du_doan(p, ids) for ten, p in nguon.items()}
    bang = pd.DataFrame({ten: cot_diem(du, gold) for ten, du in he.items()})
    print(bang.round(2).to_string())

    rep.mkdir(parents=True, exist_ok=True)
    dai = bang.reset_index(names="metric").melt(id_vars="metric", var_name="system", value_name="score")
    dai["n"] = len(gold)
    dai.to_csv(rep / "llm_eval.csv", index=False, encoding="utf-8-sig")

    rv = tai_review(cfg, snapshots, root)
    text = rv.drop_duplicates("review_id").set_index("review_id")["text"].str.slice(0, 400).to_dict()
    loi = liet_ke_loi(he, gold, text)
    loi.to_csv(rep / "llm_error_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"[EVAL] {len(loi)} dòng lỗi -> reports/llm_error_analysis.csv")

    if "llm" in he:
        kq = pd.read_csv(nguon["llm"], dtype={"review_id": str})
        cu_hon = kq["prompt_version"].astype(str) != str(PROMPT_VERSION)
        if cu_hon.any():
            print(f"[EVAL] CẢNH BÁO: {int(cu_hon.sum())} dòng llm_results.csv sinh bởi prompt khác bản hiện tại "
                  f"({PROMPT_VERSION}); chạy lại llm.py trước khi chấm")
        # Lịch sử: mỗi lần chấm một dòng (mô hình, phiên bản prompt, điểm) để so khi sửa prompt
        lich_su = {"thoi_gian": dt.datetime.now().isoformat(timespec="seconds"), "model": kq["model"].iloc[0],
                   "prompt_version": kq["prompt_version"].iloc[0], "n_nhan_tay": len(gold),
                   **{f"llm_{k}": (round(v, 3) if v == v else np.nan) for k, v in bang["llm"].items()}}
        p_ls = rep / "llm_eval_history.csv"
        cu = pd.read_csv(p_ls) if p_ls.exists() else pd.DataFrame()
        pd.concat([cu, pd.DataFrame([lich_su])], ignore_index=True).to_csv(p_ls, index=False, encoding="utf-8-sig")

        cac_co = [c for c in ("flag_hai_chieu", "flag_ngan_nhieu_khia_canh", "flag_khong_can_cu") if c in kq.columns]
        hk = pd.DataFrame([{"quy_tac": c, "so_dong": int(_la_dung(kq[c]).sum()),
                            "ty_le_pct": round(float(_la_dung(kq[c]).mean() * 100), 2)} for c in cac_co])
        hk.to_csv(rep / "llm_hau_kiem.csv", index=False, encoding="utf-8-sig")

        chi_phi = uoc_tinh_chi_phi(cfg, kq, len(kq), dem_review_thanh_pho(root, str(max(map(str, snapshots)))))
        if chi_phi is not None:
            chi_phi.to_csv(rep / "llm_cost_estimate.csv", index=False, encoding="utf-8-sig")
            print(chi_phi[["pham_vi", "so_review", "chi_phi_usd", "ngay_o_goi_mien_phi"]].to_string(index=False))
    return bang


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chấm LLM so với phương pháp đối chứng")
    ap.add_argument("--config", default="configs/London.yaml")
    ap.add_argument("--make-template", action="store_true", help="tạo tệp mẫu và hướng dẫn gán nhãn tay")
    a = ap.parse_args()
    run_evaluate(a.config, make_template=a.make_template)