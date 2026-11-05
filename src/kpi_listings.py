"""Mô-đun tính toán các chỉ số KPI cho bảng listings (Nhóm C - London).

Chạy độc lập (hỗ trợ cả python src/kpi_listings.py và python -m src.kpi_listings):
    python src/kpi_listings.py --config configs/London.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Đảm bảo chạy được từ bất kỳ đâu
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# CÁC HÀM BOOTSTRAP (KHOẢNG TIN CẬY)
# -----------------------------------------------------------------------------
def bootstrap_median_ci(series: pd.Series, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Tính khoảng tin cậy 95% cho Trung vị. Trả về (NaN, NaN) nếu n < 3."""
    vals = series.dropna().to_numpy()
    if len(vals) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_medians[i] = np.median(sample)
    return round(float(np.percentile(boot_medians, 2.5)), 2), round(float(np.percentile(boot_medians, 97.5)), 2)


def bootstrap_mean_ci(series: pd.Series, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Tính khoảng tin cậy 95% cho Trung bình. Trả về (NaN, NaN) nếu n < 3."""
    vals = series.dropna().to_numpy()
    if len(vals) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_means[i] = np.mean(sample)
    return round(float(np.percentile(boot_means, 2.5)), 3), round(float(np.percentile(boot_means, 97.5)), 3)


# -----------------------------------------------------------------------------
# CÁC MODULE XỬ LÝ 
# -----------------------------------------------------------------------------
def build_listing_borough(df: pd.DataFrame, last_snap: str, proc_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Bước A: Đồng bộ borough mốc cuối và tạo tệp cho mô-đun LLM."""
    df_last = df[(df["snapshot"] == last_snap) & (~df["flag_missing_listing"])]
    borough_last = (
        df_last.dropna(subset=["neighbourhood_cleansed"])
        .drop_duplicates(subset=["id"], keep="first")
        .set_index("id")["neighbourhood_cleansed"]
    )
    borough_fallback = (
        df[~df["flag_missing_listing"]]
        .dropna(subset=["neighbourhood_cleansed"])
        .drop_duplicates(subset=["id"], keep="last")
        .set_index("id")["neighbourhood_cleansed"]
    )
    full_borough_map = borough_last.combine_first(borough_fallback)

    listing_area_map = (
        df[~df["flag_missing_listing"]]
        .drop_duplicates(subset=["id"], keep="first")
        .set_index("id")["analysis_area"]
    )

    df_listing_borough = pd.DataFrame({
        "listing_id": list(full_borough_map.index),
        "neighbourhood_cleansed": list(full_borough_map.values),
    })
    df_listing_borough["analysis_area"] = df_listing_borough["listing_id"].map(listing_area_map)
    df_listing_borough = df_listing_borough.sort_values("listing_id").reset_index(drop=True)

    df_listing_borough.to_csv(proc_dir / "listing_borough.csv", index=False, encoding="utf-8")
    return df_listing_borough, full_borough_map


def compute_borough_kpis(df: pd.DataFrame, last_snap: str, borough_map: pd.Series,
                         min_borough_n: int, min_reviews: int, seed: int, proc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bước B: Tính toán K1, K2, K3 và tách tệp ngưỡng thành phố."""
    df_q1 = df[(df["snapshot"] == last_snap) & (~df["flag_missing_listing"])].copy()
    df_q1["borough"] = df_q1["id"].map(borough_map)

    missing_borough_count = int(df_q1["borough"].isna().sum())
    if missing_borough_count > 0:
        print(f"[WARN] Bỏ qua {missing_borough_count} listing ở mốc cuối do thiếu Borough.")
    df_q1 = df_q1.dropna(subset=["borough"])

    # Lọc K1: Giá hợp lệ
    k1_mask = (~df_q1["flag_duplicate"]) & (df_q1["price_per_person"].notna()) & (df_q1["price_per_person"] > 0)
    
    # Lọc K2: Đủ số review và có điểm value
    k2_mask = (
        (~df_q1["flag_missing_score"]) 
        & (~df_q1["flag_duplicate"]) 
        & (df_q1["review_scores_value"].notna())
        & (df_q1["number_of_reviews"] >= min_reviews)
    )

    k1_df = df_q1[k1_mask]
    k2_df = df_q1[k2_mask]

    city_median_ppp = round(float(k1_df["price_per_person"].median()), 2)
    city_mean_val = round(float(k2_df["review_scores_value"].mean()), 3)

    borough_records = []
    total_counts = df_q1.groupby("borough").size()

    for borough, n_total_b in total_counts.items():
        grp_k1 = k1_df[k1_df["borough"] == borough]
        grp_k2 = k2_df[k2_df["borough"] == borough]
        
        n_k1_b = len(grp_k1)
        n_k2_b = len(grp_k2)

        # Tính K1 (Giá)
        if n_k1_b > 0:
            med_ppp = round(float(grp_k1["price_per_person"].median()), 2)
            p25_ppp = round(float(grp_k1["price_per_person"].quantile(0.25)), 2)
            p75_ppp = round(float(grp_k1["price_per_person"].quantile(0.75)), 2)
            ci_low_ppp, ci_high_ppp = bootstrap_median_ci(grp_k1["price_per_person"], seed=seed)
        else:
            med_ppp = p25_ppp = p75_ppp = ci_low_ppp = ci_high_ppp = np.nan

        # Tính K2 (Value)
        if n_k2_b > 0:
            med_val = round(float(grp_k2["review_scores_value"].median()), 2)
            mean_val = round(float(grp_k2["review_scores_value"].mean()), 3)
            ci_low_val, ci_high_val = bootstrap_mean_ci(grp_k2["review_scores_value"], seed=seed)
        else:
            med_val = mean_val = ci_low_val = ci_high_val = np.nan

        reliable = bool(min(n_k1_b, n_k2_b) >= min_borough_n)

        # Đánh giá K3 (Đạt chuẩn khi CẢ HAI cột giá trị ước lượng đều vượt ngưỡng)
        price_confident = False
        value_confident = False
        good_val_status = "Không đạt"

        if reliable and pd.notna(med_ppp) and pd.notna(mean_val):
            if (med_ppp <= city_median_ppp) and (mean_val >= city_mean_val):
                price_confident = bool(pd.notna(ci_high_ppp) and ci_high_ppp <= city_median_ppp)
                value_confident = bool(pd.notna(ci_low_val) and ci_low_val >= city_mean_val)
                
                if price_confident and value_confident:
                    good_val_status = "Đạt (Chắc chắn)"
                else:
                    good_val_status = "Đạt (Sát ngưỡng)"

        borough_records.append({
            "snapshot": last_snap,
            "neighbourhood_cleansed": borough,
            "n_listing": n_total_b,
            "n_valid_ppp": n_k1_b,
            "n_valid_value": n_k2_b,
            "ppp_median": med_ppp,
            "ppp_p25": p25_ppp,
            "ppp_p75": p75_ppp,
            "ppp_ci_95_low": ci_low_ppp,
            "ppp_ci_95_high": ci_high_ppp,
            "value_mean": mean_val,
            "value_ci_95_low": ci_low_val,
            "value_ci_95_high": ci_high_val,
            "value_median": med_val,
            "price_confident": price_confident,
            "value_confident": value_confident,
            "reliable": reliable,
            "good_value_status": good_val_status,
        })

    df_kpi_b = pd.DataFrame(borough_records)
    # Sắp xếp ổn định (Stable Sort) cho đúng 33 dòng borough
    df_kpi_b = df_kpi_b.sort_values(by=["n_listing", "neighbourhood_cleansed"], ascending=[False, True]).reset_index(drop=True)
    df_kpi_b.to_csv(proc_dir / "kpi_borough_value.csv", index=False, encoding="utf-8")

    # Tạo tệp Summary riêng cho ngưỡng thành phố
    df_city_sum = pd.DataFrame([{
        "snapshot": last_snap,
        "metric_scope": "City Threshold",
        "city_ppp_median": city_median_ppp,
        "city_value_mean": city_mean_val,
        "notes": f"Ngưỡng tính trên các listing hợp lệ (đã loại bỏ outlier và missing). n_ppp={len(k1_df)}, n_value={len(k2_df)}."
    }])
    df_city_sum.to_csv(proc_dir / "kpi_city_summary.csv", index=False, encoding="utf-8")

    return df_kpi_b, df_city_sum


def compute_time_kpis(df: pd.DataFrame, snapshots: list[str], proc_dir: Path) -> pd.DataFrame:
    """Bước C: Tính KPI Thời gian (K4)."""
    df_valid = df[(~df["flag_missing_listing"]) & (~df["flag_duplicate"])].copy()

    time_records = []
    for snap in snapshots:
        snap_df = df_valid[df_valid["snapshot"] == snap]
        
        # 1. Toàn bộ (all)
        n_all = len(snap_df)
        n_no_rev_all = int((snap_df["number_of_reviews"] == 0).sum())
        pct_no_rev_all = round((n_no_rev_all / n_all * 100), 2) if n_all > 0 else np.nan
        
        # Đếm n_scored trực tiếp từ review_scores_rating
        n_scored_all = int(snap_df["review_scores_rating"].notna().sum())
        mean_rating_all = round(float(snap_df["review_scores_rating"].mean()), 3) if n_scored_all > 0 else np.nan

        time_records.append({
            "snapshot": snap,
            "room_type": "all",
            "total_listings": n_all,
            "n_no_reviews": n_no_rev_all,
            "pct_no_reviews": pct_no_rev_all,
            "n_scored": n_scored_all,
            "mean_rating_scores": mean_rating_all,
        })

        # 2. Theo room_type
        for r_type, r_grp in snap_df.groupby("room_type"):
            n_rt = len(r_grp)
            n_no_rev_rt = int((r_grp["number_of_reviews"] == 0).sum())
            pct_no_rev_rt = round((n_no_rev_rt / n_rt * 100), 2) if n_rt > 0 else np.nan
            
            n_scored_rt = int(r_grp["review_scores_rating"].notna().sum())
            mean_rating_rt = round(float(r_grp["review_scores_rating"].mean()), 3) if n_scored_rt > 0 else np.nan

            time_records.append({
                "snapshot": snap,
                "room_type": r_type,
                "total_listings": n_rt,
                "n_no_reviews": n_no_rev_rt,
                "pct_no_reviews": pct_no_rev_rt,
                "n_scored": n_scored_rt,
                "mean_rating_scores": mean_rating_rt,
            })

    df_kpi_time = pd.DataFrame(time_records)
    df_kpi_time.to_csv(proc_dir / "kpi_time_reviews.csv", index=False, encoding="utf-8")
    return df_kpi_time


def build_definitions(currency: str, min_reviews: int, rep_dir: Path) -> pd.DataFrame:
    """Bước D: Tạo Từ điển KPI."""
    kpi_defs = [
        {
            "kpi_id": "K1",
            "kpi_name": "Giá trung vị trên mỗi khách theo borough",
            "target": "Xác định chi phí lưu trú thực tế tại mốc cuối",
            "formula": "median(price_clean / accommodates)",
            "unit": f"{currency}/khách/đêm",
            "currency": currency,
            "stakeholder": "Khách du lịch (Nhóm C)",
        },
        {
            "kpi_id": "K2",
            "kpi_name": "Điểm độ đáng tiền (Value) trung bình",
            "target": "Đo lường mức độ hài lòng về chất lượng nhận lại (Lọc listing có >= 5 đánh giá)",
            "formula": f"mean(review_scores_value) với number_of_reviews >= {min_reviews}",
            "unit": "thang điểm 0–5",
            "currency": "n/a",
            "stakeholder": "Khách du lịch (Nhóm C)",
        },
        {
            "kpi_id": "K3",
            "kpi_name": "Quận có giá tốt và độ hài lòng cao",
            "target": "Phân loại quận Đạt (Chắc chắn) khi cả 2 khoảng tin cậy vượt ngưỡng, ngược lại Đạt (Sát ngưỡng)",
            "formula": "Đạt: (ppp_median <= City_Median_PPP) AND (value_mean >= City_Mean_Value); Chắc chắn: thêm (ppp_ci_high <= City_Median_PPP) AND (value_ci_low >= City_Mean_Value)",
            "unit": "phân loại chữ",
            "currency": "n/a",
            "stakeholder": "Khách du lịch (Nhóm C)",
        },
        {
            "kpi_id": "K4",
            "kpi_name": "Biến động điểm và tỷ lệ chưa có review",
            "target": "Theo dõi chất lượng dịch vụ và nguồn cung mới",
            "formula": "mean(review_scores_rating) và (reviews==0 / listings * 100)",
            "unit": "điểm và %",
            "currency": "n/a",
            "stakeholder": "Khách du lịch",
        },
        {
            "kpi_id": "K5",
            "kpi_name": "Độ sẵn sàng phòng theo mùa",
            "target": "Phân tích chu kỳ mùa vụ",
            "formula": "Chờ module Calendar tích hợp",
            "unit": "tỷ lệ %",
            "currency": "n/a",
            "stakeholder": "Khách du lịch (Nhóm C)",
        },
        {
            "kpi_id": "K6",
            "kpi_name": "Tỷ lệ phàn nàn độ ồn/an ninh",
            "target": "Cảnh báo quận rủi ro",
            "formula": "Chờ module LLM tích hợp",
            "unit": "tỷ lệ %",
            "currency": "n/a",
            "stakeholder": "Khách du lịch (Nhóm C)",
        },
    ]
    df_kpi_defs = pd.DataFrame(kpi_defs)
    df_kpi_defs.to_csv(rep_dir / "kpi_definitions.csv", index=False, encoding="utf-8")
    return df_kpi_defs


# -----------------------------------------------------------------------------
# ĐIỀU PHỐI (CHẠY TOÀN BỘ)
# -----------------------------------------------------------------------------
def calculate_listings_kpis(config_path: str | Path, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    snapshots = [str(d) for d in cfg["snapshots"]]
    last_snap = max(snapshots)
    currency = cfg.get("currency", "GBP")

    kpi_cfg = cfg.get("kpi", {}) or {}
    min_borough_n = int(kpi_cfg.get("min_listing_per_borough", 10))
    min_reviews = int(kpi_cfg.get("min_reviews_for_score", 5))
    seed = int(kpi_cfg.get("seed", 42))

    proc_dir = root / cfg.get("processed_dir", "data/processed")
    rep_dir = root / "reports"
    
    cleaned_file = proc_dir / "listings_cleaned.csv"
    if not cleaned_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {cleaned_file}. Hãy chạy qa_listings.py trước.")

    df = pd.read_csv(cleaned_file, dtype={"id": str}, low_memory=False)

    # ĐỘ BỀN: Kiểm tra mốc
    missing_snaps = set(snapshots) - set(df["snapshot"].unique())
    if missing_snaps:
        raise ValueError(f"Dữ liệu làm sạch bị thiếu mốc trong config: {missing_snaps}. Hãy chạy lại QA.")

    # Chạy các Bước
    df_listing_borough, borough_map = build_listing_borough(df, last_snap, proc_dir)
    df_kpi_borough, df_city_sum = compute_borough_kpis(df, last_snap, borough_map, min_borough_n, min_reviews, seed, proc_dir)
    df_kpi_time = compute_time_kpis(df, snapshots, proc_dir)
    df_kpi_defs = build_definitions(currency, min_reviews, rep_dir)

    # In kết quả
    print("=" * 70)
    print("=== TỰ ĐỐI CHIẾU SỐ LIỆU KPI LISTINGS VỚI KẾ HOẠCH NHÓM 12 ===")
    print(f"1. Số borough (kpi_borough_value.csv)    : {len(df_kpi_borough)} dòng")
    print(f"2. Borough tin cậy (>= {min_borough_n} căn)          : {int(df_kpi_borough['reliable'].sum())}")
    
    n_chac_chan = int((df_kpi_borough["good_value_status"] == "Đạt (Chắc chắn)").sum())
    n_sat = int((df_kpi_borough["good_value_status"] == "Đạt (Sát ngưỡng)").sum())
    print(f"3. Nhóm Giá tốt - Value cao (Tổng {n_chac_chan + n_sat}) : {n_chac_chan} Chắc chắn, {n_sat} Sát ngưỡng")
    
    print(f"4. Trung vị giá/người TP                 : {df_city_sum['city_ppp_median'].iloc[0]} {currency}")
    print(f"5. Trung bình điểm value TP              : {df_city_sum['city_value_mean'].iloc[0]}")
    
    pct_no_rev = df_kpi_time[df_kpi_time["room_type"] == "all"]["pct_no_reviews"].tolist()
    print(f"6. Tỷ lệ chưa có review (all)            : {pct_no_rev}")
    print("=" * 70)

    return {
        "listing_borough": df_listing_borough,
        "kpi_borough": df_kpi_borough,
        "kpi_city_summary": df_city_sum,
        "kpi_time": df_kpi_time,
        "kpi_definitions": df_kpi_defs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/London.yaml", help="Đường dẫn file config")
    args = parser.parse_args()
    calculate_listings_kpis(args.config)