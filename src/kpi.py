from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:  # chạy "python src/kpi_listings.py" hoặc import dạng "src.kpi_listings" đều được
    from src.qa import kiem_cau_truc
except ImportError:
    try:
        from qa import kiem_cau_truc
    except ImportError:
        def kiem_cau_truc(df, req_cols, num_cols, name):
            missing = req_cols - set(df.columns)
            if missing: raise ValueError(f"Bảng {name} thiếu cột: {missing}")


ROOT = Path(__file__).resolve().parents[1]

def bootstrap_ci(values, stat, n_boot=1000, seed=42):
    """Khoảng tin cậy 95% bằng bootstrap: lấy mẫu có hoàn lại n_boot lần, tính stat mỗi lần."""
    vals = np.asarray(values.dropna(), dtype=float)
    if len(vals) < 3:
        return np.nan, np.nan
    mau = np.random.default_rng(seed).choice(vals, size=(n_boot, len(vals)))
    lo, hi = np.percentile(stat(mau, axis=1), [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


# ----------------------------------------------------------------------------- K1-K3: THEO BOROUGH
def kpi_borough(ls, last_snap, min_n, min_reviews, seed):
    """K1-K3: Mỗi borough tính giá/người (trung vị), điểm value (trung bình) và phân loại giá tốt."""
    d = ls[(ls["snapshot"] == last_snap) & ls["id"].notna() & ls["neighbourhood_cleansed"].notna()
           & ~ls["flag_duplicate"]].copy()
    d.loc[d["accommodates"] > 0, "ppp"] = d["price_clean"] / d["accommodates"]

    dung_gia = d["ppp"].notna() & ~d["flag_outlier_price"]
    dung_value = ~d["flag_missing_scores"] & (d["number_of_reviews"] >= min_reviews)
    
    print(f"[KPI] Mốc {last_snap}: {len(d)} listing | Dùng cho giá/người: {int(dung_gia.sum())} "
          f"| Dùng cho value: {int(dung_value.sum())} (>= {min_reviews} review)")

    city_ppp = round(float(d.loc[dung_gia, "ppp"].median()), 2)
    city_value = round(float(d.loc[dung_value, "review_scores_value"].mean()), 3)

    dong = []
    for ten, g in d.groupby("neighbourhood_cleansed"):
        gia = g.loc[dung_gia[g.index], "ppp"]
        val = g.loc[dung_value[g.index], "review_scores_value"]
        p_lo, p_hi = bootstrap_ci(gia, np.median, seed=seed)
        v_lo, v_hi = bootstrap_ci(val, np.mean, seed=seed)
        dong.append({
            "neighbourhood_cleansed": ten, "n_listing": len(g), "n_price": len(gia), "n_value": len(val),
            "ppp_median": round(float(gia.median()), 2) if len(gia) else np.nan, "ppp_ci_low": p_lo, "ppp_ci_high": p_hi,
            "value_mean": round(float(val.mean()), 3) if len(val) else np.nan,
            "value_median": round(float(val.median()), 2) if len(val) else np.nan,
            "value_ci_low": v_lo, "value_ci_high": v_hi,
        })
    b = pd.DataFrame(dong)
    b["reliable"] = (b["n_price"] >= min_n) & (b["n_value"] >= min_n)
    
    dat = b["reliable"] & (b["ppp_median"] <= city_ppp) & (b["value_mean"] >= city_value)
    chac = dat & (b["ppp_ci_high"] <= city_ppp) & (b["value_ci_low"] >= city_value)
    b["good_value"] = np.select([chac, dat], ["Đạt (Chắc chắn)", "Đạt (Sát ngưỡng)"], default="Không đạt")
    b["city_ppp_median"], b["city_value_mean"] = city_ppp, city_value
    
    return b.sort_values(["n_listing", "neighbourhood_cleansed"], ascending=[False, True]).reset_index(drop=True)


# ----------------------------------------------------------------------------- K4 & K5: THỜI GIAN VÀ ĐỘ SẴN SÀNG
def kpi_thoi_gian(ls, cac_moc, last_snap, proc):
    """K4 & K5: Tổng hợp tỷ lệ chưa có review và độ sẵn sàng từ Calendar (đã xóa K6 cũ price_median)."""
    theo_moc, theo_thang, theo_mua = [], [], []
    for snap in cac_moc:
        s = ls[(ls["snapshot"] == snap) & ls["id"].notna() & ~ls["flag_duplicate"]]
        
        # Chỉ tính ppp ở mốc cuối
        if snap == last_snap:
            gia = s["price_clean"].where(s["price_clean"] > 0)
            ppp = (gia / s["accommodates"].where(s["accommodates"] > 0)).dropna()
        else:
            ppp = pd.Series(dtype=float)

        p_cal = proc / snap / "calendar_cleaned.csv"
        # Bọc file Calendar bằng exists() như thống nhất ở lần trước
        if not p_cal.exists():
            raise FileNotFoundError(f"Chưa có {p_cal}; chạy qa.py trước")
            
        cal = pd.read_csv(p_cal, usecols=["date", "available", "flag_date_missing", "flag_available_missing", "flag_duplicated"])
        cal = cal[~cal["flag_date_missing"] & ~cal["flag_available_missing"] & ~cal["flag_duplicated"]].copy()
        cal["a"] = cal["available"].astype("string").str.lower().map({"true": 1.0, "false": 0.0})
        cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
        cal = cal.dropna(subset=["a", "date"])

        # Bỏ price_median (K6 cũ)
        theo_moc.append({
            "snapshot": snap, "n_listing": len(s),
            "ppp_median": round(float(ppp.median()), 2) if len(ppp) else np.nan,
            "pct_no_reviews": round(float((s["number_of_reviews"] == 0).mean() * 100), 2),
            "rating_mean": round(float(s["review_scores_rating"].mean()), 3),
            "availability_pct": round(float(cal["a"].mean() * 100), 2), "n_calendar_days": len(cal),
        })
        
        cal["date_thang"] = cal["date"].dt.strftime("%Y-%m")
        a_thang = cal.groupby("date_thang")["a"].agg(n_days = "size", a_pct = "mean").reset_index()
        a_thang["availability_pct"] = a_thang["a_pct"] * 100
        a_thang["snapshot"] = snap
        theo_thang.append(a_thang[["snapshot", "date_thang", "availability_pct", "n_days"]])
        
    
    thang = pd.concat(theo_thang, ignore_index=True)        
    return (pd.DataFrame(theo_moc), thang)


def kpi_room_type(ls, moc_cuoi):    #K9 - K10
    ls_last = ls[(ls["snapshot"] == moc_cuoi) & (ls["room_type"].notna()) 
                & (ls["neighbourhood_cleansed"].notna()) 
                & (~ls["flag_missing_price_last"])]
    kieu_theo_vung = []
    median_entire = ls_last.loc[ls_last["room_type"] == "Entire home/apt", "price_clean"].median()
    median_private = ls_last.loc[ls_last["room_type"] == "Private room", "price_clean"].median()
    for borough, g in ls_last.groupby("neighbourhood_cleansed"):
        ty_le_entire = round((g["room_type"] == "Entire home/apt").mean(), 2)
        ty_le_private = round((g["room_type"] == "Private room").mean(), 2)
        trung_vi_en = g.loc[g["room_type"] == "Entire home/apt", "price_clean"].median()
        trung_vi_pri = g.loc[g["room_type"] == "Private room", "price_clean"].median()
        lech_trong_vung = round(trung_vi_en / trung_vi_pri, 2)
        gia_tuong_doi_en = round(trung_vi_en / median_entire, 2)
        gia_tuong_doi_pri = round(trung_vi_pri / median_private, 2)
        kieu_theo_vung.append({"neighbour_hood": borough,
                               "ty_le_entire": ty_le_entire,
                               "ty_le_private": ty_le_private,
                               "trung_vi_en": trung_vi_en,
                               "trung_vi_pri": trung_vi_pri,
                               "chenh_lech_gia_loai_phong_trong_vung": lech_trong_vung,
                               "chi_so_gia_tuong_doi_cua_entire_so_voi_city": gia_tuong_doi_en,
                               "chi_so_gia_tuong_doi_cua_private_so_voi_city": gia_tuong_doi_pri})
    
    p_cal = ROOT / "data" / "processed" / "calendar_cleaned.csv"
    calendar = pd.read_csv(p_cal, usecols=["listing_id", "date", "available", "flag_date_missing",
                                            "flag_available_missing", "flag_duplicated", "flag_not_in_snapshot"])
    calendar["trung_ngay_in_moc"] = calendar.duplicated(subset = ["listing_id", "date"], keep = "last")
    cal = calendar[~calendar["trung_ngay_in_moc"]]
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
    cal["date_thang"] = cal["date"].dt.strftime("%Y-%m")
    ls_room = ls.loc[(ls["room_type"].notna()), ["listing_id", "room_type"]]
    cal_room_type = cal.merge(ls_room, on = "listing_id", how = "inner")
    cal_entire = cal_room_type[cal_room_type["room_type"] == "Entire home/apt"]
    cal_private = cal_room_type[cal_room_type["room_type"] == "Private room"]
    a_thang_entire = cal_entire.groupby("date_thang")["available"].agg(n_days = "size", a_pct = "mean").reset_index()
    a_thang_entire["availability_pct"] = a_thang_entire["a_pct"] * 100
    a_thang_private = cal_private.groupby("date_thang")["available"].agg(n_days = "size", a_pct = "mean").reset_index()
    a_thang_private["availability_pct"] = a_thang_private["a_pct"] * 100
    return pd.DataFrame(kieu_theo_vung), a_thang_entire, a_thang_entire
        

# ----------------------------------------------------------------------------- K7-K8: ỒN / AN TOÀN (LLM)
def kpi_complaints(ls_cuoi, p_llm, min_reviews):
    """Tỷ lệ review chê tiếng ồn / an toàn theo borough, từ kết quả LLM."""
    r = pd.read_csv(p_llm, dtype={"review_id": str, "listing_id": str})
    kiem_cau_truc(r, {"review_id", "listing_id", "noise_neg", "safety_neg"}, set(), "llm_results.csv")
    r = r.drop_duplicates("review_id")
    nha = ls_cuoi[["id", "neighbourhood_cleansed"]].dropna().drop_duplicates("id")
    r = r.merge(nha, left_on="listing_id", right_on="id", how="inner")
    g = r.groupby("neighbourhood_cleansed").agg(n_reviews=("review_id", "size"),
                                                n_noise=("noise_neg", "sum"), n_safety=("safety_neg", "sum")).reset_index()
    g["noise_pct"] = (g["n_noise"] / g["n_reviews"] * 100).round(2)
    g["safety_pct"] = (g["n_safety"] / g["n_reviews"] * 100).round(2)
    g["reliable"] = g["n_reviews"] >= min_reviews
    return g.sort_values(["reliable", "noise_pct"], ascending=[False, False]).reset_index(drop=True)


def dinh_nghia(tien_te, min_reviews):
    return pd.DataFrame([
        ("K1", "Giá trung vị trên mỗi khách theo borough", "Biết borough nào rẻ", "median(price_clean / accommodates), mốc cuối, sau QA", f"{tien_te}/khách/đêm", "kpi_borough_value.csv"),
        ("K2", "Điểm value trung bình theo borough", "Biết borough nào khách thấy đáng tiền", f"mean(review_scores_value) với number_of_reviews >= {min_reviews}", "điểm 0-5", "kpi_borough_value.csv"),
        ("K3", "Borough giá tốt + value cao", "Gợi ý borough nên chọn", "Đạt: giá/người <= trung vị TP và value >= trung bình TP (Chắc chắn: cả khoảng tin cậy 95% cũng vượt ngưỡng)", "phân loại", "kpi_borough_value.csv"),
        ("K4", "Tỷ lệ listing chưa có review, điểm rating theo mốc", "Theo dõi chất lượng dữ liệu đánh giá", "mean(number_of_reviews == 0) và mean(review_scores_rating)", "% và điểm", "kpi_snapshot.csv"),
        ("K5", "Độ sẵn sàng theo tháng", "Biết lúc nào còn nhiều chỗ trống", "số ngày available = t / tổng số ngày hợp lệ, theo (mốc, tháng)", "%", "kpi_availability_month.csv"),
        ("K6", "Tỷ lệ review chê tiếng ồn theo borough", "Cảnh báo borough hay bị phàn nàn về ồn", "số review có khía cạnh 'độ ồn' bị chê / số review đã gán nhãn LLM", "%", "kpi_complaints_borough.csv"),
        ("K7", "Tỷ lệ review chê an toàn theo borough", "Cảnh báo borough hay bị phàn nàn về an toàn", "số review có khía cạnh 'an toàn' bị chê / số review đã gán nhãn LLM", "%", "kpi_complaints_borough.csv"),
        ("K8", "Tỷ lệ Entire home/apt và Private room trên mỗi khu vực và giá trung vị của mỗi loại phòng trong khu vực đó", "Xem loại phòng nào chiếm tỷ lệ nhiều hơn và có giá tốt hơn trong khu vực", "groupby neighbourhood_cleansed, mean(roomtype), median(roomtype)", "%, GBP", "kpi_room_type.csv"),
        ("K9", "Độ chênh lệch giá trung vị của từng loại phòng theo từng khu vực với giá trung vị theo cả thành phố", "Xem giá các loại phòng của khu vực có cao hơn so với mặt bằng chung của toàn thành phố không", "median(roomtype_neighbourhood) / median(roomtype_city)", "%", "kpi_room_type.csv")
    ], columns=["kpi_id", "kpi_name", "target", "formula", "unit", "output_file"])


# ----------------------------------------------------------------------------- CHẠY CHÍNH
def tinh_kpi(config_path, root=ROOT):
    root = Path(root)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cac_moc = [str(d) for d in cfg["snapshots"]]
    last_snap = max(cac_moc)
    k = cfg.get("kpi", {}) or {}
    min_n = int(k.get("min_listing_per_borough", 10))
    min_reviews = int(k.get("min_reviews_for_score", 5))
    seed = int(k.get("seed", 42))
    tien_te = cfg.get("currency", "GBP")
    proc = root / "data" / "processed"

    p_ls = proc / "listings_cleaned.csv"
    if not p_ls.exists():
        raise FileNotFoundError(f"Chưa có {p_ls}; chạy qa.py trước")
    ls = pd.read_csv(p_ls, dtype={"id": str}, low_memory=False)
    
    # Loại bỏ price_raw_num khỏi kiem_cau_truc
    kiem_cau_truc(ls, {"snapshot", "id", "neighbourhood_cleansed", "accommodates", "price_clean",
                       "number_of_reviews", "review_scores_rating", "review_scores_value",
                       "flag_duplicate", "flag_outlier_price", "flag_missing_scores"},
                  {"accommodates", "number_of_reviews", "review_scores_value"}, "listings_cleaned")
    ls["snapshot"] = ls["snapshot"].astype(str)

    kq = {"borough_value": kpi_borough(ls, last_snap, min_n, min_reviews, seed)}
    
    
    # Tính toán K4-K5 (Thiếu Calendar sẽ văng lỗi tại đây theo đúng rule gốc)
    moc, thang = kpi_thoi_gian(ls, cac_moc, last_snap, proc)
    kq.update({"snapshot": moc, "availability_month": thang})

    room_type_median, entire, private = kpi_room_type(ls, last_snap)
    kq.update({"room_type_median": room_type_median, "entire_availability": entire, "private_availability": private})
    
    p_llm = proc / "llm_results.csv"
    if p_llm.exists():
        kq["complaints_borough"] = kpi_complaints(ls[ls["snapshot"] == last_snap], p_llm,
                                                  int(k.get("min_reviews_llm", 10)))
    else:
        print(f"[KPI] K7-K8 CHƯA TÍNH: không thấy {p_llm} (chạy llm.py trước)")

    for ten, df in kq.items():
        df.to_csv(proc / f"kpi_{ten}.csv", index=False, encoding="utf-8")
    dinh_nghia(tien_te, min_reviews).to_csv(proc / "kpi_definitions.csv", index=False, encoding="utf-8")

    b = kq["borough_value"]
    n_chac = int((b["good_value"] == "Đạt (Chắc chắn)").sum())
    n_sat = int((b["good_value"] == "Đạt (Sát ngưỡng)").sum())
    print(f"\n[K1-K3] {len(b)} borough, {int(b['reliable'].sum())} đủ tin cậy; nhóm giá tốt + value cao: "
          f"{n_chac + n_sat} ({n_chac} chắc chắn, {n_sat} sát ngưỡng) | trung vị giá/người TP "
          f"{b['city_ppp_median'].iloc[0]} {tien_te}, value TB TP {b['city_value_mean'].iloc[0]}")
    
    
    return kq


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tính KPI Nhóm C")
    ap.add_argument("--config", default="configs/London.yaml")
    tinh_kpi(ap.parse_args().config)
