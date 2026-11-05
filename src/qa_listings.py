"""Xử lý làm sạch, điền khuyết và kiểm tra chất lượng (QA) riêng cho bảng listings.

Chạy độc lập để kiểm thử (từ thư mục gốc repo):
    python src/qa_listings_main.py --config configs/London.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAN_TOKENS = {"", "nan", "n/a", "na", "null", "none"}

KEEP_COLS = {
    "id", "last_scraped", "room_type", "neighbourhood_cleansed",
    "latitude", "longitude",
    "price", "accommodates", "minimum_nights", "maximum_nights",
    "amenities", "bathrooms", "bathrooms_text", "bedrooms", "beds",
    "number_of_reviews", "review_scores_rating", "review_scores_value",
    "host_id", "calculated_host_listings_count",
    "has_availability", "availability_365",
}

def clean_price_series(s: pd.Series) -> pd.Series:
    s_str = s.astype("string").str.strip()
    is_empty = s_str.isna() | s_str.str.lower().isin(NAN_TOKENS)
    cleaned = s_str.where(~is_empty, np.nan).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def extract_bathrooms_from_text(text_series: pd.Series) -> pd.Series:
    def parse_one(val) -> float:
        if pd.isna(val): return np.nan
        s = str(val).strip().lower()
        if not s or s in NAN_TOKENS: return np.nan
        if "half-bath" in s or "half bath" in s: return 0.5
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else np.nan
    return text_series.map(parse_one)

def parse_amenities_features(amenities_series: pd.Series) -> pd.DataFrame:
    counts, wifi, kitchen = [], [], []
    for val in amenities_series:
        if pd.isna(val):
            counts.append(0)
            wifi.append(False)
            kitchen.append(False)
            continue
        try:
            items = json.loads(str(val))
            if not isinstance(items, list): items = []
        except Exception:
            items = []
        counts.append(len(items))
        lower_items = [str(x).lower() for x in items]
        wifi.append(any("wifi" in x or "wi-fi" in x for x in lower_items))
        kitchen.append(any("kitchen" in x for x in lower_items))
    return pd.DataFrame({
        "amenities_count": counts,
        "has_wifi": wifi,
        "has_kitchen": kitchen,
    }, index=amenities_series.index)

def process_listings(config_path: str | Path, root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    snapshots = [str(d) for d in cfg["snapshots"]]
    last_snap = snapshots[-1]

    qa_cfg = cfg.get("qa", {}) or {}
    price_max = float(qa_cfg.get("price_max_gbp", qa_cfg.get("price_max", 5000.0)))
    min_nights_cap = int(qa_cfg.get("min_nights_cap", 365))
    allowed_room_types = qa_cfg.get("room_types", ["Entire home/apt", "Private room"])

    proc_dir = root / cfg.get("processed_dir", "data/processed")
    listing_loc = pd.read_csv(
        proc_dir / cfg.get("listings_ids", "listings-loc.csv"),
        dtype={"listing_id": str},
        usecols=["listing_id", "analysis_area"],
        encoding="utf-8-sig",
    )
    vung_loc = pd.read_csv(proc_dir / "vung-loc.csv", encoding="utf-8-sig")
    for col in ["lon_min", "lat_min", "lon_max", "lat_max"]:
        vung_loc[col] = pd.to_numeric(
            vung_loc[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

    frames = []
    for snap in snapshots:
        raw_path = root / "data" / "raw" / snap / "listings.csv.gz"
        df_raw = pd.read_csv(
            raw_path,
            compression="gzip",
            dtype={"id": str, "host_id": str},
            usecols=lambda c: c in KEEP_COLS,
            low_memory=False,
        )
        df_merged = (
            df_raw.merge(listing_loc, left_on="id", right_on="listing_id", how="right")
            .merge(vung_loc[["analysis_area", "lon_min", "lat_min", "lon_max", "lat_max"]],
                   on="analysis_area", how="left")
        )
        df_merged["snapshot"] = snap
        
        # SỬA A1: Đánh dấu dòng vắng & khôi phục ID
        df_merged["flag_missing_listing"] = df_merged["id"].isna()
        df_merged["id"] = df_merged["id"].fillna(df_merged["listing_id"])
        frames.append(df_merged)

    df = pd.concat(frames, ignore_index=True)
    
    # SỬA A1 & A2: Định nghĩa present mask để không tính dòng ma
    present = ~df["flag_missing_listing"]
    n_present = present.sum()

    raw_price_non_empty = ~(
        df["price"].astype("string").isna()
        | df["price"].astype("string").str.strip().str.lower().isin(NAN_TOKENS)
    )
    df["price_raw_num"] = clean_price_series(df["price"])
    df["price_clean"] = np.where(df["snapshot"] == last_snap, df["price_raw_num"], np.nan)

    bath_num = pd.to_numeric(df["bathrooms"], errors="coerce")
    bath_from_text = extract_bathrooms_from_text(df["bathrooms_text"])
    text_non_empty = ~(
        df["bathrooms_text"].astype("string").isna()
        | df["bathrooms_text"].astype("string").str.strip().str.lower().isin(NAN_TOKENS)
    )
    
    df["flag_wrong_type"] = present & (
        (raw_price_non_empty & df["price_raw_num"].isna())
        | (text_non_empty & bath_from_text.isna() & bath_num.isna())
    )
    df["bathrooms_clean"] = bath_num.fillna(bath_from_text)

    for num_col in [
        "accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights",
        "number_of_reviews", "review_scores_rating", "review_scores_value",
        "calculated_host_listings_count", "availability_365", "latitude", "longitude",
    ]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    if "has_availability" in df.columns:
        df["has_availability_clean"] = df["has_availability"].astype(str).str.lower().map(
            {"t": True, "true": True, "f": False, "false": False}
        )
    else:
        df["has_availability_clean"] = np.nan

    amen_df = parse_amenities_features(df["amenities"])
    df["amenities_count"] = amen_df["amenities_count"]
    df["has_wifi"] = amen_df["has_wifi"]
    df["has_kitchen"] = amen_df["has_kitchen"]

    df["flag_missing_price_last"] = present & (df["snapshot"] == last_snap) & df["price_clean"].isna()

    # GIỮ NGUYÊN: flag_no_reviews của Lead
    df["flag_no_reviews"] = present & ((df["number_of_reviews"].fillna(0) == 0) | df["review_scores_rating"].isna())

    # SỬA A3: Thêm flag_missing_score rõ ràng
    df["flag_missing_score"] = present & (
        df["number_of_reviews"].isna() 
        | df["review_scores_rating"].isna() 
        | df["review_scores_value"].isna()
    )

    df["flag_missing_bed_bath"] = present & (
        df["bedrooms"].isna() | df["beds"].isna() | df["bathrooms_clean"].isna()
    )

    df["bedrooms_raw"] = df["bedrooms"]
    df["beds_raw"] = df["beds"]

    df["bedrooms_clean"] = df.groupby(["room_type", "accommodates"])["bedrooms"].transform(lambda s: s.fillna(s.median()))
    df["bedrooms_clean"] = df["bedrooms_clean"].fillna(df["bedrooms"].median()).fillna(1.0)
    df["beds_clean"] = df.groupby(["room_type", "accommodates"])["beds"].transform(lambda s: s.fillna(s.median()))
    df["beds_clean"] = df["beds_clean"].fillna(df["beds"].median()).fillna(1.0)
    df["bathrooms_clean"] = df.groupby("room_type")["bathrooms_clean"].transform(lambda s: s.fillna(s.median())).fillna(1.0)

    df["flag_outlier_price"] = present & (df["snapshot"] == last_snap) & df["price_clean"].notna() & ((df["price_clean"] <= 0) | (df["price_clean"] > price_max))
    
    # SỬA A2: Cờ không bắt dòng ma
    df["flag_outlier_min_nights"] = present & (
        df["minimum_nights"].isna()
        | (df["minimum_nights"] <= 0)
        | (df["minimum_nights"] > min_nights_cap)
    )
    df["flag_out_of_bounds"] = present & ~(
        df["longitude"].between(df["lon_min"], df["lon_max"])
        & df["latitude"].between(df["lat_min"], df["lat_max"])
    )

    dup_in_snap = present & df.duplicated(subset=["snapshot", "id"], keep="first")
    dup_across_snap = present & df.duplicated(subset=["id", "last_scraped"], keep="first")
    df["flag_duplicate"] = dup_in_snap | dup_across_snap

    room_type_changed = df.groupby("id")["room_type"].transform("nunique") > 1
    df["flag_category_mismatch"] = present & ((~df["room_type"].isin(allowed_room_types)) | room_type_changed)

    # =========================================================================
    # TỔNG HỢP BÁO CÁO QA (SỬA A4 & A5)
    # =========================================================================
    last_present = present & (df["snapshot"] == last_snap)
    n_last = int(last_present.sum())
    last_df = df[last_present]
    valid_price_df = last_df[~last_df["flag_missing_price_last"] & ~last_df["flag_outlier_price"]]

    miss_bath_pct = round((df.loc[present, "bathrooms"].isna().sum() / max(n_present, 1)) * 100, 1)
    miss_price_n = int(df["flag_missing_price_last"].sum())
    no_rev_by_snap = df[present].groupby("snapshot")["flag_no_reviews"].sum().astype(int).tolist()

    qa_rules = [
        {
            "table": "listings",
            "rule_name": "R1_Wrong_Type_Bathrooms_Price",
            "category": "Kiểu sai",
            "condition": "price không ép kiểu được hoặc bathrooms rỗng phải tách từ bathrooms_text",
            "reason": f"Dữ liệu gốc lưu price dạng chuỗi và khuyết cột số bathrooms ({miss_bath_pct}%)",
            "action": "Ép kiểu float cho price; tách Regex từ bathrooms_text sang bathrooms_clean",
            "affected_rows": int(df.loc[present, "bathrooms"].isna().sum() + df["flag_wrong_type"].sum()),
            "affected_pct": round((df.loc[present, "bathrooms"].isna().sum() + df["flag_wrong_type"].sum()) / max(n_present, 1) * 100, 2),
            "kpi_metric": "Tỷ lệ đủ dữ liệu phòng tắm (%)",
            "kpi_before": round((1 - df.loc[present, "bathrooms"].isna().mean()) * 100, 2),
            "kpi_after": round((1 - df.loc[present, "bathrooms_clean"].isna().mean()) * 100, 2),
        },
        {
            "table": "listings",
            "rule_name": "R2_Missing_Price_Last_Snapshot",
            "category": "Thiếu dữ liệu",
            "condition": f"snapshot == '{last_snap}' và price IS NULL",
            "reason": f"Mốc cuối có {miss_price_n} listing chưa cập nhật giá",
            "action": "Gắn cờ flag_missing_price_last để loại trừ khi tính KPI giá",
            "affected_rows": miss_price_n,
            "affected_pct": round(miss_price_n / max(n_last, 1) * 100, 2),
            "kpi_metric": "Số listing mốc cuối hợp lệ để tính Giá trung vị",
            "kpi_before": n_last,
            "kpi_after": int((~last_df["flag_missing_price_last"]).sum()),
        },
        {
            "table": "listings",
            "rule_name": "R3_Missing_Bedrooms_Beds_Imputation",
            "category": "Thiếu dữ liệu",
            "condition": "bedrooms IS NULL hoặc beds IS NULL hoặc bathrooms_clean IS NULL",
            "reason": f"Chủ nhà bỏ trống thông tin số phòng ngủ/giường (thường ở căn hộ Studio)",
            "action": "Gắn cờ flag_missing_bed_bath; điền khuyết bằng trung vị theo nhóm",
            "affected_rows": int(df["flag_missing_bed_bath"].sum()),
            "affected_pct": round(df["flag_missing_bed_bath"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Số phòng ngủ trung bình",
            "kpi_before": round(float(df.loc[present, "bedrooms_raw"].mean()), 3),
            "kpi_after": round(float(df.loc[present, "bedrooms_clean"].mean()), 3),
        },
        {
            "table": "listings",
            "rule_name": "R4_Missing_Review_Scores",
            "category": "Thiếu dữ liệu",
            "condition": "number_of_reviews == 0 hoặc review_scores_rating IS NULL",
            "reason": f"Listing mới đăng chưa từng có khách đánh giá qua 4 mốc: {no_rev_by_snap}",
            "action": "Gắn cờ flag_no_reviews; loại trừ khi tính KPI điểm đánh giá trung bình",
            "affected_rows": int(df["flag_no_reviews"].sum()),
            "affected_pct": round(df["flag_no_reviews"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Điểm đánh giá trung bình (nếu gán 0 vs bỏ căn chưa có review)",
            "kpi_before": round(float(df.loc[present, "review_scores_rating"].fillna(0).mean()), 3),
            "kpi_after": round(float(df.loc[present & ~df["flag_no_reviews"], "review_scores_rating"].mean()), 3),
        },
        {
            "table": "listings",
            "rule_name": "R5_Outlier_Price_Last_Snapshot",
            "category": "Ngoại lai",
            "condition": f"snapshot == '{last_snap}' và (price_clean <= 0 hoặc price_clean > {price_max})",
            "reason": f"Giá phòng niêm yết quá cao bất thường hoặc <= 0",
            "action": "Gắn cờ flag_outlier_price; loại khỏi phép tính Giá trung bình mốc cuối",
            "affected_rows": int(df["flag_outlier_price"].sum()),
            "affected_pct": round(df["flag_outlier_price"].sum() / max(n_last, 1) * 100, 2),
            "kpi_metric": "Giá phòng trung bình mốc cuối (GBP)",
            "kpi_before": round(float(last_df["price_clean"].mean()), 2),
            "kpi_after": round(float(valid_price_df["price_clean"].mean()), 2),
        },
        {
            "table": "listings",
            "rule_name": "R6_Outlier_Minimum_Nights",
            "category": "Ngoại lai",
            "condition": f"minimum_nights <= 0 hoặc minimum_nights > {min_nights_cap}",
            "reason": "Yêu cầu số đêm tối thiểu vượt quá 1 năm (>365 ngày) không phải lưu trú ngắn hạn",
            "action": "Gắn cờ flag_outlier_min_nights",
            "affected_rows": int(df["flag_outlier_min_nights"].sum()),
            "affected_pct": round(df["flag_outlier_min_nights"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Số đêm tối thiểu trung bình (ngày)",
            "kpi_before": round(float(df.loc[present, "minimum_nights"].mean()), 2),
            "kpi_after": round(float(df.loc[present & ~df["flag_outlier_min_nights"], "minimum_nights"].mean()), 2),
        },
        {
            "table": "listings",
            "rule_name": "R7_Outlier_Spatial_Bounds",
            "category": "Ngoại lai không gian",
            "condition": "longitude ngoài [lon_min, lon_max] hoặc latitude ngoài [lat_min, lat_max]",
            "reason": "Kiểm tra tọa độ thực tế có bị lệch khỏi 8 vùng được phân công hay không",
            "action": "Gắn cờ flag_out_of_bounds",
            "affected_rows": int(df["flag_out_of_bounds"].sum()),
            "affected_pct": round(df["flag_out_of_bounds"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Số listing nằm đúng trong 8 vùng phân tích",
            "kpi_before": n_present,
            "kpi_after": int((present & ~df["flag_out_of_bounds"]).sum()),
        },
        {
            "table": "listings",
            "rule_name": "R8_Duplicate_Records",
            "category": "Trùng lặp",
            "condition": "Trùng (snapshot, id) trong cùng mốc hoặc trùng (id, last_scraped) giữa các mốc",
            "reason": "Đảm bảo mỗi listing chỉ xuất hiện 1 lần ở mỗi mốc",
            "action": "Gắn cờ flag_duplicate",
            "affected_rows": int(df["flag_duplicate"].sum()),
            "affected_pct": round(df["flag_duplicate"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Tổng số bản ghi duy nhất qua 4 mốc",
            "kpi_before": n_present,
            "kpi_after": int((present & ~df["flag_duplicate"]).sum()),
        },
        {
            "table": "listings",
            "rule_name": "R9_Category_Mismatch_RoomType",
            "category": "Danh mục lệch",
            "condition": f"room_type không thuộc {allowed_room_types} hoặc thay đổi loại phòng giữa các mốc",
            "reason": "Kiểm tra tính nhất quán danh mục loại phòng qua 4 quý",
            "action": "Gắn cờ flag_category_mismatch",
            "affected_rows": int(df["flag_category_mismatch"].sum()),
            "affected_pct": round(df["flag_category_mismatch"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Tỷ lệ Entire home/apt hợp lệ (%)",
            "kpi_before": round(float((df.loc[present, "room_type"] == "Entire home/apt").mean() * 100), 2),
            "kpi_after": round(float((df.loc[present & ~df["flag_category_mismatch"], "room_type"] == "Entire home/apt").mean() * 100), 2),
        },
    ]
    qa_report_df = pd.DataFrame(qa_rules)

    # SỬA A6: Thêm "price" và "price_raw_num" vào drop_cols
    drop_cols = ["price", "price_raw_num", "amenities", "listing_id", "bedrooms_raw", "beds_raw"]
    df_out = df.drop(columns=[c for c in drop_cols if c in df.columns])

    proc_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(proc_dir / "listings_cleaned.csv", index=False, encoding="utf-8")

    for snap, g in df_out.groupby("snapshot"):
        snap_dir = proc_dir / str(snap)
        snap_dir.mkdir(parents=True, exist_ok=True)
        g.to_csv(snap_dir / "listing_cleaned.csv", index=False, encoding="utf-8")

    # SỬA A4: Xuất qa_report.csv chung
    rep_dir = root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    qa_report_path = rep_dir / "qa_report.csv"
    
    if qa_report_path.exists():
        try:
            existing_rep = pd.read_csv(qa_report_path, encoding="utf-8")
            if "table" in existing_rep.columns:
                other_tables = existing_rep[existing_rep["table"] != "listings"]
                qa_report_df = pd.concat([qa_report_df, other_tables], ignore_index=True)
        except Exception:
            pass
    qa_report_df.to_csv(qa_report_path, index=False, encoding="utf-8")

    return df_out, qa_report_df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/London.yaml")
    args = ap.parse_args()
    df_out, rep = process_listings(args.config)
    print("=" * 70)
    print(f"[OK] listings_cleaned.csv: {len(df_out)} dòng, {df_out['snapshot'].nunique()} mốc")
    print("[OK] reports/qa_report.csv (table = listings)")
    print(rep[["rule_name", "affected_rows", "affected_pct", "kpi_before", "kpi_after"]].to_string(index=False))
    print("=" * 70)