from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:  # chạy "python src/evaluate.py" hoặc import dạng "src.evaluate" đều được
    from src.llm import doc_cau_hinh, tai_reviews_cleaned, ASPECTS, SENTIMENTS, PROMPT_VERSION
    from src.qa import PROJECT_PATH, kiem_cau_truc
except ImportError:
    from llm import doc_cau_hinh, tai_reviews_cleaned, ASPECTS, SENTIMENTS, PROMPT_VERSION
    from qa import PROJECT_PATH, kiem_cau_truc


CAC_DIEM = {"cam_xuc": "accuracy cảm xúc", "khia_canh_f1": "F1 khía cạnh (khen + chê)",
            "bi_che_f1": "F1 khía cạnh bị chê", "on_bi_che_f1": "F1 phát hiện chê ồn",
            "an_toan_bi_che_f1": "F1 phát hiện chê an toàn", "ngon_ngu": "accuracy ngôn ngữ"}

COT_NHAN_TAY = {"review_id", "sentiment", "aspects_pos", "aspects_neg", "language"}


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


def tach(s):
    return [a for a in str(s).split("|") if a and a != "nan"]


def doc_nhan_tay(p):
    """Đọc và KIỂM TRA bộ nhãn tay: sai danh mục thì dừng và nói rõ."""
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    kiem_cau_truc(d, COT_NHAN_TAY, set(), p.name)
    d = d.drop_duplicates("review_id")
    d["sentiment"] = d["sentiment"].str.strip().str.lower()
    loi = sorted(set(d["sentiment"]) - set(SENTIMENTS))
    if loi:
        raise ValueError(f"{p.name}: sentiment chỉ nhận {SENTIMENTS}; thấy giá trị lạ/rỗng: {loi}")
    gold = {}
    for r in d.itertuples():
        pos, neg = tach(r.aspects_pos), tach(r.aspects_neg)
        la = sorted((set(pos) | set(neg)) - set(ASPECTS))
        if la:
            raise ValueError(f"{p.name}: review {r.review_id} có khía cạnh ngoài danh mục {ASPECTS}: {la}")
        gold[r.review_id] = {"sentiment": r.sentiment, "aspects_pos": pos, "aspects_neg": neg,
                             "language": r.language.strip().lower() or "und",
                             "aspects": pos + neg,
                             "noise_neg": [a for a in neg if a == "noise"],
                             "safety_neg": [a for a in neg if a == "safety"]}
    return gold


def doc_du_doan(p, ids):
    """CSV kết quả -> {trường: {mã review: giá trị}} cho các review trong ids (cùng dạng với nhãn tay)."""
    d = pd.read_csv(p, dtype={"review_id": str}, keep_default_na=False).drop_duplicates("review_id")
    d = d[d["review_id"].isin(ids)].set_index("review_id")
    pos = {m: set(tach(v)) for m, v in d["aspects_pos"].items()}
    neg = {m: set(tach(v)) for m, v in d["aspects_neg"].items()}
    du = {"sentiment": d["sentiment"].str.strip().str.lower().to_dict(),
          "language": d["language"].str.strip().str.lower().replace("", "und").to_dict(),
          "aspects": {m: pos[m] | neg[m] for m in d.index},
          "aspects_pos": pos, "aspects_neg": neg,
          "noise_neg": {m: {a for a in neg[m] if a == "noise"} for m in d.index},
          "safety_neg": {m: {a for a in neg[m] if a == "safety"} for m in d.index}}
    return du


def cot_diem(du, gold):
    return {"cam_xuc": ti_le_dung(du, gold, "sentiment"), "khia_canh_f1": f1(du, gold, "aspects"), 
            "bi_che_f1": f1(du, gold, "aspects_neg"),
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
            for truong in ("aspects_pos", "aspects_neg", "aspects"):
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
    ok = ket_qua_llm[["in_tok", "out_tok"]].dropna()
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

# ----------------------------------------------------------------------------- CHẠY CHÍNH
def run_evaluate(config_path, root=PROJECT_PATH, make_template=False):
    root = Path(root)
    p_cfg = Path(config_path)
    if not p_cfg.exists() and not p_cfg.is_absolute():
        p_cfg = root / p_cfg
    config = yaml.safe_load(p_cfg.read_text(encoding="utf-8"))
    snapshots = config["snapshots"]
    cfg = doc_cau_hinh(config_path)
    samples, proc = root / "data" / "samples", root / "data" / "processed"

    if make_template:
        rv = tai_reviews_cleaned(cfg, root)
        mau = rv.loc[(rv["flag_human"]) & (~rv["flag_id_duplicates"]), ["review_id", "comments_fixed"]]
        for cot in ("sentiment", "aspects_pos", "aspects_neg", "language"):
            mau[cot] = ""
        samples.mkdir(parents=True, exist_ok=True)
        mau.to_csv(samples / "hand_labels_template.csv", index=False, encoding="utf-8-sig")
        print(f"[EVAL] tạo data/samples/hand_labels_template.csv ({len(mau)} review) và huong_dan_gan_nhan.md. "
              "Hai người gán độc lập, thống nhất rồi lưu thành data/samples/hand_labels.csv")
        return None

    p_gold = root / "data/samples/hand_labels.csv"
    if not p_gold.exists():
        print(f"[EVAL] CHƯA CHẤM: không thấy nhãn tay tại {p_gold} (chạy --make-template rồi gán nhãn)")
        return None
    gold = doc_nhan_tay(p_gold)

    nguon = {"baseline": proc / "baseline_results.csv", "llm": proc / "llm_results.csv"}
    ids = set(gold)
    for ten, p in list(nguon.items()):
        if not p.exists():
            print(f"[EVAL] thiếu {p.name}: bỏ qua hệ thống {ten}")
            del nguon[ten]
            continue
        ids &= set(pd.read_csv(p, dtype={"review_id": str}, usecols=["review_id"])["review_id"])
    if not nguon or not ids:
        print("[EVAL] CHƯA CHẤM: chưa có review nào vừa có nhãn tay vừa có dự đoán của mọi hệ thống")
        return None
    gold = {m: g for m, g in gold.items() if m in ids}
    print(f"[EVAL] chấm trên {len(gold)} review có nhãn tay và có dự đoán của {' & '.join(nguon)}")

    he = {ten: doc_du_doan(p, ids) for ten, p in nguon.items()}
    bang = pd.DataFrame({ten: cot_diem(du, gold) for ten, du in he.items()})
    print(bang.round(2).to_string())

    proc.mkdir(parents=True, exist_ok=True)
    dai = bang.reset_index(names="metric").melt(id_vars="metric", var_name="system", value_name="score")
    dai["n"] = len(gold)
    dai.to_csv(proc / "llm_eval.csv", index=False, encoding="utf-8-sig")

    rv = tai_reviews_cleaned(cfg, root)
    text = rv.set_index("review_id")["comments_fixed"].str.slice(0, 400).to_dict()
    loi = liet_ke_loi(he, gold, text)
    loi.to_csv(proc / "llm_error_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"[EVAL] {len(loi)} dòng lỗi -> data/processed/llm_error_analysis.csv")

    if "llm" in he:
        kq = pd.read_csv(nguon["llm"], dtype={"review_id": str})
        # Lịch sử: mỗi lần chấm một dòng (mô hình, phiên bản prompt, điểm) để so khi sửa prompt
        lich_su = {"thoi_gian": dt.datetime.now().isoformat(timespec="seconds"), "model": kq["model"].iloc[0],
                   "prompt_version": kq["prompt_version"].iloc[0], "n_nhan_tay": len(gold),
                   **{f"llm_{k}": (round(v, 3) if v == v else np.nan) for k, v in bang["llm"].items()}}
        p_ls = proc / "llm_eval_history.csv"
        cu = pd.read_csv(p_ls) if p_ls.exists() else pd.DataFrame()
        pd.concat([cu, pd.DataFrame([lich_su])], ignore_index=True).to_csv(p_ls, index=False, encoding="utf-8-sig")

        cac_co = [c for c in ("flag_hai_chieu", "flag_ngan_nhieu_khia_canh", "flag_khong_can_cu") if c in kq.columns]
        hk = pd.DataFrame([{"quy_tac": c, "so_dong": int(kq[c].astype(str).str.lower().eq("true").sum()),
                            "ty_le_pct": round(float(kq[c].astype(str).str.lower().eq("true").mean() * 100), 2)}
                           for c in cac_co])
        hk.to_csv(proc / "llm_hau_kiem.csv", index=False, encoding="utf-8-sig")

        chi_phi = uoc_tinh_chi_phi(cfg, kq, len(kq), dem_review_thanh_pho(root, str(max(map(str, snapshots)))))
        if chi_phi is not None:
            chi_phi.to_csv(proc / "llm_cost_estimate.csv", index=False, encoding="utf-8-sig")
            print(chi_phi[["pham_vi", "so_review", "chi_phi_usd", "ngay_o_goi_mien_phi"]].to_string(index=False))
    return bang


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chấm LLM so với phương pháp đối chứng")
    ap.add_argument("--config", default="configs/London.yaml")
    ap.add_argument("--make-template", action="store_true", help="tạo tệp mẫu và hướng dẫn gán nhãn tay")
    a = ap.parse_args()
    run_evaluate(a.config, make_template=a.make_template)