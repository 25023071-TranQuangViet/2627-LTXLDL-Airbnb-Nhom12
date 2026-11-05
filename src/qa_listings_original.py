"""Làm sạch và kiểm tra chất lượng (QA) cho bảng listings (Nhóm C - London).

Chạy độc lập:
    python src/qa_listings.py --config configs/London.yaml

Đầu ra:
    1. data/processed/listings_cleaned.csv  (Tệp gộp duy nhất cho 4 mốc)
    2. reports/qa_report.csv                (Có cột table='listings' để gộp chung với reviews, calendar)
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

# Các cột cần giữ lại từ listings.csv.gz gốc
KEEP_COLS = {
    "id", "last_scraped", "room_type", "property_type", "neighbourhood_cleansed",
    "latitude", "longitude",
    "price", "accommodates", "minimum_nights", "maximum_nights",
    "amenities", "bathrooms", "bathrooms_text", "bedrooms", "beds",
    "number_of_reviews", "review_scores_rating", "review_scores_value",
}

# Các cột số cần kiểm tra lỗi ép kiểu trước khi dùng pd.to_numeric
NUM_COLS = [
    "accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights",
    "number_of_reviews", "review_scores_rating", "review_scores_value",
    "latitude", "longitude",
]

DEFAULT_EXPECTED = {
    "listings": 1000,
    "entire_home": 875,
    "private_room": 125,
    "no_review": [50, 30, 25, 25],
    "missing_price_last": 20,
    "area_min": 120,
    "area_max": 130,
}


# ----------------------------------------------------------------------------- Tiện ích làm sạch & định dạng
def is_empty_str(s: pd.Series, extra: set[str] | None = None) -> pd.Series:
    s_str = s.astype("string").str.strip().str.lower()
    tokens = NAN_TOKENS | (extra or set())
    return (s_str.isna() | s_str.isin(tokens)).fillna(True).astype(bool)


def fmt_snap_pct(series_by_snap: pd.Series) -> str:
    """Định dạng chuỗi 'mốc: %' gọn đẹp thay vì in dict thô."""
    return ", ".join(f"{snap}: {val:.1f}%" for snap, val in series_by_snap.items())


def fmt_snap_int(series_by_snap: pd.Series) -> str:
    """Định dạng chuỗi 'mốc: số lượng' gọn đẹp thay vì in dict thô."""
    return ", ".join(f"{snap}: {int(val)}" for snap, val in series_by_snap.items())


def clean_price_series(s: pd.Series) -> pd.Series:
    """Chuyển chuỗi tiền tệ '$1,234.00' sang float; chuỗi rỗng trả về NaN."""
    empty = is_empty_str(s)
    cleaned = s.astype("string").where(~empty, np.nan).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def extract_bathrooms_from_text(text_series: pd.Series) -> pd.Series:
    """Trích xuất số phòng tắm từ bathrooms_text ('1.5 baths' -> 1.5, 'Half-bath' -> 0.5)."""
    def parse_one(val) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if not s or s in NAN_TOKENS:
            return np.nan
        if "half-bath" in s or "half bath" in s:
            return 0.5
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else np.nan

    return text_series.map(parse_one)


def parse_amenities_count(amenities_series: pd.Series) -> pd.Series:
    """Đếm số tiện ích từ chuỗi JSON; nếu thiếu, rỗng ('[]') hoặc lỗi JSON thì trả về NaN."""
    def count_one(val) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if not s or s.lower() in NAN_TOKENS or s == "[]":
            return np.nan
        try:
            items = json.loads(s)
            if not isinstance(items, list) or len(items) == 0:
                return np.nan
            return float(len(items))
        except Exception:
            return np.nan

    return amenities_series.map(count_one)


# ----------------------------------------------------------------------------- Kiểm tra kỳ vọng (Không dùng assert)
def check_expected_listings(df: pd.DataFrame, snapshots: list[str], last_snap: str, expected: dict) -> list[str]:
    """Đối chiếu các con số kiểm tra sau khi lọc với expected trong config (chỉ cảnh báo)."""
    warnings: list[str] = []
    present = ~df["flag_missing_listing"]

    for idx, snap in enumerate(snapshots):
        sub = df[(df["snapshot"] == snap) & present]
        n_rows = len(sub)
        if n_rows != expected["listings"]:
            warnings.append(f"[WARN] Mốc {snap}: có {n_rows} listing hiện diện (kỳ vọng {expected['listings']}).")

        n_entire = int((sub["room_type"] == "Entire home/apt").sum())
        n_priv = int((sub["room_type"] == "Private room").sum())
        if n_entire != expected["entire_home"] or n_priv != expected["private_room"]:
            warnings.append(
                f"[WARN] Mốc {snap}: Entire home/apt={n_entire}, Private room={n_priv} "
                f"(kỳ vọng {expected['entire_home']} / {expected['private_room']})."
            )

        n_no_rev = int((sub["number_of_reviews"] == 0).sum())
        exp_nr = expected.get("no_review", [])
        if idx < len(exp_nr) and n_no_rev != exp_nr[idx]:
            warnings.append(f"[WARN] Mốc {snap}: number_of_reviews==0 là {n_no_rev} (kỳ vọng {exp_nr[idx]}).")

        area_counts = sub["analysis_area"].value_counts()
        bad_areas = area_counts[(area_counts < expected["area_min"]) | (area_counts > expected["area_max"])]
        if not bad_areas.empty:
            warnings.append(
                f"[WARN] Mốc {snap}: có vùng nằm ngoài khoảng [{expected['area_min']}, {expected['area_max']}]: "
                f"{bad_areas.to_dict()}"
            )

    last_sub = df[(df["snapshot"] == last_snap) & present]
    n_miss_price = int(last_sub["price_clean"].isna().sum())
    if n_miss_price != expected["missing_price_last"]:
        warnings.append(
            f"[WARN] Mốc cuối {last_snap}: thiếu price={n_miss_price} (kỳ vọng {expected['missing_price_last']})."
        )

    for w in warnings:
        print(w)
    if not warnings:
        print("[OK] Kiểm tra số lượng sau lọc: Khớp 100% với thông số expected trong config.")
    return warnings


# ----------------------------------------------------------------------------- Xử lý chính
def process_listings(config_path: str | Path, root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    snapshots = [str(d) for d in cfg["snapshots"]]
    last_snap = max(snapshots)

    qa_cfg = cfg.get("qa", {}) or {}
    price_max = float(qa_cfg["price_max_gbp"])
    min_nights_cap = int(qa_cfg["min_nights_cap"])
    allowed_room_types = list(qa_cfg["room_types"])
    dedupe_keys = list(qa_cfg.get("dedupe_keys", ["id", "last_scraped"]))
    expected = {**DEFAULT_EXPECTED, **(cfg.get("expected", {}) or {})}

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
            dtype=str,
            keep_default_na=False,
            usecols=lambda c: c in KEEP_COLS,
            low_memory=False,
        )
        df_merged = (
            df_raw.merge(listing_loc, left_on="id", right_on="listing_id", how="right")
            .merge(
                vung_loc[["analysis_area", "lon_min", "lat_min", "lon_max", "lat_max"]],
                on="analysis_area",
                how="left",
            )
        )
        df_merged["snapshot"] = snap
        df_merged["flag_missing_listing"] = is_empty_str(df_merged["id"])
        df_merged["id"] = df_merged["id"].where(~df_merged["flag_missing_listing"], df_merged["listing_id"])
        df_merged = df_merged.drop(columns=["listing_id"])
        frames.append(df_merged)

    df = pd.concat(frames, ignore_index=True)
    present = ~df["flag_missing_listing"]

    # =========================================================================
    # BƯỚC 1: KIỂU SAI (WRONG TYPE) TRÊN MỌI CỘT SỐ, NGÀY THÁNG & JSON
    # =========================================================================
    wrong_type_mask = pd.Series(False, index=df.index)

    # 1.1. Kiểm tra kiểu sai trên cột price
    raw_price_non_empty = present & (~is_empty_str(df["price"]))
    df["price_raw_num"] = clean_price_series(df["price"])
    wrong_type_mask |= raw_price_non_empty & df["price_raw_num"].isna()
    df["price_clean"] = np.where(df["snapshot"] == last_snap, df["price_raw_num"], np.nan)

    # 1.2. Kiểm tra kiểu sai trên cột bathrooms và bathrooms_text
    raw_bath_non_empty = present & (~is_empty_str(df["bathrooms"]))
    bath_num = pd.to_numeric(df["bathrooms"].where(raw_bath_non_empty, np.nan), errors="coerce")
    wrong_type_mask |= raw_bath_non_empty & bath_num.isna()

    text_non_empty = present & (~is_empty_str(df["bathrooms_text"]))
    bath_from_text = extract_bathrooms_from_text(df["bathrooms_text"])
    wrong_type_mask |= text_non_empty & bath_from_text.isna() & bath_num.isna()

    df["bathrooms_clean"] = bath_num.fillna(bath_from_text)
    df.loc[df["bathrooms_clean"] < 0, "bathrooms_clean"] = np.nan

    # 1.3. Kiểm tra kiểu sai trên toàn bộ các cột số khác trước khi ép kiểu
    for num_col in NUM_COLS:
        if num_col in df.columns:
            col_non_empty = present & (~is_empty_str(df[num_col]))
            df[num_col] = pd.to_numeric(df[num_col].where(col_non_empty, np.nan), errors="coerce")
            wrong_type_mask |= col_non_empty & df[num_col].isna()

    # 1.4. Kiểm tra kiểu sai trên cột ngày last_scraped và JSON amenities
    if "last_scraped" in df.columns:
        scraped_non_empty = present & (~is_empty_str(df["last_scraped"]))
        scraped_dt = pd.to_datetime(df["last_scraped"].where(scraped_non_empty, np.nan), errors="coerce")
        wrong_type_mask |= scraped_non_empty & scraped_dt.isna()

    amen_non_empty = present & (~is_empty_str(df["amenities"], extra={"[]"}))
    df["amenities_count"] = parse_amenities_count(df["amenities"])
    wrong_type_mask |= amen_non_empty & df["amenities_count"].isna()

    df["flag_wrong_type"] = wrong_type_mask
    # Gộp bằng OR giữa việc thiếu cột bathrooms gốc (phải tách từ text) và lỗi kiểu dữ liệu
    df["flag_bath_or_type_issue"] = present & (bath_num.isna() | df["flag_wrong_type"])

    # Tách cột _clean cho bedrooms, beds và chuẩn hóa chuỗi rỗng thành NaN cho cột phân loại
    df["bedrooms_clean"] = df["bedrooms"].where(df["bedrooms"] >= 0, np.nan)
    df["beds_clean"] = df["beds"].where(df["beds"] >= 0, np.nan)
    for str_col in ["room_type", "property_type", "neighbourhood_cleansed"]:
        if str_col in df.columns:
            df[str_col] = df[str_col].where(~is_empty_str(df[str_col]), np.nan)

    check_expected_listings(df, snapshots, last_snap, expected)

    # =========================================================================
    # BƯỚC 2: GẮN CỜ DỮ LIỆU THIẾU (MISSING)
    # =========================================================================
    df["flag_missing_price_last"] = present & (df["snapshot"] == last_snap) & df["price_clean"].isna()
    df["flag_no_reviews"] = present & (df["number_of_reviews"] == 0)
    df["flag_missing_score"] = present & (
        df["number_of_reviews"].isna()
        | df["review_scores_rating"].isna()
        | df["review_scores_value"].isna()
    )
    df["flag_missing_bed_bath"] = present & (
        df["bedrooms_clean"].isna()
        | df["beds_clean"].isna()
        | df["bathrooms_clean"].isna()
        | df["amenities_count"].isna()
    )

    # =========================================================================
    # BƯỚC 3: GẮN CỜ NGOẠI LAI (OUTLIERS)
    # =========================================================================
    df["flag_outlier_price"] = (
        present
        & (df["snapshot"] == last_snap)
        & df["price_clean"].notna()
        & ((df["price_clean"] <= 0) | (df["price_clean"] > price_max))
    )
    df["flag_outlier_min_nights"] = (
        present
        & df["minimum_nights"].notna()
        & ((df["minimum_nights"] <= 0) | (df["minimum_nights"] > min_nights_cap))
    )
    df["flag_out_of_bounds"] = (
        present
        & df["longitude"].notna()
        & df["latitude"].notna()
        & ~(
            df["longitude"].between(df["lon_min"], df["lon_max"])
            & df["latitude"].between(df["lat_min"], df["lat_max"])
        )
    )

    valid_for_ppp = (
        present
        & (df["snapshot"] == last_snap)
        & (~df["flag_missing_price_last"])
        & (~df["flag_outlier_price"])
        & (df["accommodates"] > 0)
    )
    df["price_per_person"] = np.where(
        valid_for_ppp,
        (df["price_clean"] / df["accommodates"]).round(2),
        np.nan,
    )

    # =========================================================================
    # BƯỚC 4: GẮN CỜ TRÙNG LẶP (DUPLICATES)
    # =========================================================================
    dup_in_snap = present & df.duplicated(subset=["snapshot", "id"], keep="first")
    valid_dedupe_cols = [c for c in dedupe_keys if c in df.columns]
    dup_by_keys = (
        present
        & (~df[valid_dedupe_cols].apply(is_empty_str).any(axis=1))
        & df.duplicated(subset=valid_dedupe_cols, keep="first")
    )
    df["flag_duplicate"] = dup_in_snap | dup_by_keys

    # =========================================================================
    # BƯỚC 5: GẮN CỜ DANH MỤC LỆCH (CATEGORY MISMATCH)
    # =========================================================================
    room_type_changed = df.groupby("id")["room_type"].transform(lambda s: s.dropna().nunique()) > 1
    borough_changed = df.groupby("id")["neighbourhood_cleansed"].transform(lambda s: s.dropna().nunique()) > 1

    df["flag_category_mismatch"] = present & (
        (~df["room_type"].isin(allowed_room_types))
        | df["neighbourhood_cleansed"].isna()
        | df["property_type"].isna()
        | room_type_changed
        | borough_changed
    )

    # =========================================================================
    # BƯỚC 6: TẠO BÁO CÁO QA ĐỘNG (ĐỊNH DẠNG "MỐC: %" VÀ DÙNG MEDIAN CHO R5)
    # =========================================================================
    n_total = len(df)
    n_present = int(present.sum())
    last_df = df[(df["snapshot"] == last_snap) & present]
    n_last = max(len(last_df), 1)
    valid_price_df = last_df[~last_df["flag_missing_price_last"] & ~last_df["flag_outlier_price"]]

    bath_miss_str = fmt_snap_pct(
        df[present].groupby("snapshot")["bathrooms"].apply(lambda s: is_empty_str(s).mean() * 100)
    )
    bed_miss_str = fmt_snap_pct(
        df[present].groupby("snapshot")["bedrooms_clean"].apply(lambda s: s.isna().mean() * 100)
    )
    no_rev_str = fmt_snap_int(
        df[present].groupby("snapshot")["flag_no_reviews"].sum()
    )
    price_miss_str = fmt_snap_pct(
        df[present].groupby("snapshot")["price_raw_num"].apply(lambda s: s.isna().mean() * 100)
    )

    qa_rules = [
        {
            "table": "listings",
            "rule_name": "R0_Missing_Listing_In_Snapshot",
            "category": "Thiếu dữ liệu (Missing)",
            "condition": "id trong listings-loc.csv không xuất hiện trong tệp listings.csv.gz của mốc chụp",
            "reason": f"Kiểm tra tính đầy đủ của {len(listing_loc)} ID được phân công qua {len(snapshots)} mốc chụp",
            "action": "Gắn cờ flag_missing_listing; khôi phục id và loại khỏi mẫu số của các quy tắc QA khác",
            "affected_rows": int(df["flag_missing_listing"].sum()),
            "affected_pct": round(df["flag_missing_listing"].sum() / n_total * 100, 2),
            "kpi_metric": "Tổng số bản ghi listing hiện diện qua các mốc",
            "kpi_before": float(n_total),
            "kpi_after": float(n_present),
        },
        {
            "table": "listings",
            "rule_name": "R1_Wrong_Type_And_Bathrooms_Text",
            "category": "Kiểu sai (Wrong Type)",
            "condition": "bathrooms rỗng cần tách từ bathrooms_text HOẶC các cột số/ngày/JSON có dữ liệu nhưng sai kiểu",
            "reason": f"Tỷ lệ khuyết cột số bathrooms ({bath_miss_str}); kiểm tra ép kiểu trên toàn bộ cột số và JSON",
            "action": "Gắn cờ flag_wrong_type; ép kiểu số/datetime và tách Regex từ bathrooms_text sang bathrooms_clean",
            "affected_rows": int(df["flag_bath_or_type_issue"].sum()),
            "affected_pct": round(df["flag_bath_or_type_issue"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Tỷ lệ dòng hiện diện có số liệu phòng tắm hợp lệ (%)",
            "kpi_before": round((1 - is_empty_str(df.loc[present, "bathrooms"]).mean()) * 100, 2),
            "kpi_after": round((1 - df.loc[present, "bathrooms_clean"].isna().mean()) * 100, 2),
        },
        {
            "table": "listings",
            "rule_name": "R2_Missing_Price_Last_Snapshot",
            "category": "Thiếu dữ liệu (Missing)",
            "condition": f"snapshot == '{last_snap}' và price_clean IS NULL",
            "reason": f"Tỷ lệ thiếu price theo mốc ({price_miss_str}); chỉ dùng price ở mốc cuối ({last_snap})",
            "action": "Gắn cờ flag_missing_price_last; loại trừ khi tính KPI giá nhưng giữ lại khi đếm tổng cung",
            "affected_rows": int(df["flag_missing_price_last"].sum()),
            "affected_pct": round(df["flag_missing_price_last"].sum() / n_last * 100, 2),
            "kpi_metric": f"Số listing mốc cuối ({last_snap}) hợp lệ để tính Giá trung vị",
            "kpi_before": float(len(last_df)),
            "kpi_after": float((~last_df["flag_missing_price_last"]).sum()),
        },
        {
            "table": "listings",
            "rule_name": "R3_Missing_Bed_Bath_Amenities",
            "category": "Thiếu dữ liệu (Missing)",
            "condition": "bedrooms_clean, beds_clean, bathrooms_clean hoặc amenities_count IS NULL",
            "reason": f"Chủ nhà để trống thông tin phòng/tiện ích (tỷ lệ thiếu bedrooms: {bed_miss_str})",
            "action": "Chỉ gắn cờ flag_missing_bed_bath, giữ nguyên NaN ở cột _clean (không điền trung vị hay 1.0)",
            "affected_rows": int(df["flag_missing_bed_bath"].sum()),
            "affected_pct": round(df["flag_missing_bed_bath"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Số giường trung bình (beds_clean) trước vs sau khi lọc dòng thiếu thông tin phòng",
            "kpi_before": round(float(df.loc[present, "beds_clean"].mean()), 3),
            "kpi_after": round(float(df.loc[present & ~df["flag_missing_bed_bath"], "beds_clean"].mean()), 3),
        },
        {
            "table": "listings",
            "rule_name": "R4a_No_Reviews",
            "category": "Thiếu dữ liệu (Missing)",
            "condition": "number_of_reviews == 0",
            "reason": f"Số listing chưa có khách đánh giá theo mốc ({no_rev_str})",
            "action": "Gắn cờ flag_no_reviews (tách biệt với cờ thiếu điểm đánh giá)",
            "affected_rows": int(df["flag_no_reviews"].sum()),
            "affected_pct": round(df["flag_no_reviews"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Số lượng đánh giá trung bình/listing (toàn bộ vs chỉ căn đã có đánh giá)",
            "kpi_before": round(float(df.loc[present, "number_of_reviews"].mean()), 2),
            "kpi_after": round(float(df.loc[present & ~df["flag_no_reviews"], "number_of_reviews"].mean()), 2),
        },
        {
            "table": "listings",
            "rule_name": "R4b_Missing_Review_Scores",
            "category": "Thiếu dữ liệu (Missing)",
            "condition": "review_scores_rating IS NULL hoặc review_scores_value IS NULL hoặc number_of_reviews IS NULL",
            "reason": f"Listing thiếu điểm đánh giá ({int(df['flag_missing_score'].sum())} dòng trên {len(snapshots)} mốc)",
            "action": "Gắn cờ flag_missing_score; loại trừ khi tính KPI điểm rating và điểm value trung bình",
            "affected_rows": int(df["flag_missing_score"].sum()),
            "affected_pct": round(df["flag_missing_score"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Số bản ghi hợp lệ để tính điểm đánh giá (rating & value)",
            "kpi_before": float(n_present),
            "kpi_after": float((present & ~df["flag_missing_score"]).sum()),
        },
        {
            "table": "listings",
            "rule_name": "R5_Outlier_Price_Last_Snapshot",
            "category": "Ngoại lai (Outliers)",
            "condition": f"snapshot == '{last_snap}' và price_clean có giá trị nhưng (price_clean <= 0 hoặc price_clean > {price_max})",
            "reason": f"Giá niêm yết ở mốc cuối ({last_snap}) vượt trần {price_max} {cfg.get('currency', 'GBP')} hoặc <= 0",
            "action": "Gắn cờ flag_outlier_price; loại khỏi phép tính Giá trung vị và Giá trên mỗi khách",
            "affected_rows": int(df["flag_outlier_price"].sum()),
            "affected_pct": round(df["flag_outlier_price"].sum() / n_last * 100, 2),
            "kpi_metric": f"Giá phòng trung vị mốc cuối ({cfg.get('currency', 'GBP')})",
            "kpi_before": round(float(last_df["price_clean"].median()), 2),
            "kpi_after": round(float(valid_price_df["price_clean"].median()), 2),
        },
        {
            "table": "listings",
            "rule_name": "R6_Outlier_Minimum_Nights",
            "category": "Ngoại lai (Outliers)",
            "condition": f"minimum_nights có giá trị nhưng (minimum_nights <= 0 hoặc minimum_nights > {min_nights_cap})",
            "reason": f"Số đêm tối thiểu vượt ngưỡng {min_nights_cap} ngày (không tính dòng rỗng là ngoại lai)",
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
            "category": "Ngoại lai không gian (Spatial Outliers)",
            "condition": "Tọa độ (longitude, latitude) có giá trị nhưng nằm ngoài khung [lon_min, lon_max] x [lat_min, lat_max]",
            "reason": "Kiểm tra tọa độ thực tế so với ranh giới vùng phân tích trong vung-loc.csv",
            "action": "Gắn cờ flag_out_of_bounds",
            "affected_rows": int(df["flag_out_of_bounds"].sum()),
            "affected_pct": round(df["flag_out_of_bounds"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Số bản ghi hiện diện nằm chuẩn trong khung tọa độ vùng",
            "kpi_before": float(n_present),
            "kpi_after": float((present & ~df["flag_out_of_bounds"]).sum()),
        },
        {
            "table": "listings",
            "rule_name": "R8_Duplicate_Records",
            "category": "Trùng lặp (Duplicates)",
            "condition": f"Trùng (snapshot, id) trong cùng mốc hoặc trùng khóa dedupe_keys={dedupe_keys} giữa các mốc",
            "reason": "Đảm bảo mỗi listing là duy nhất trong 1 mốc và được cào mới ở các mốc tiếp theo",
            "action": "Gắn cờ flag_duplicate",
            "affected_rows": int(df["flag_duplicate"].sum()),
            "affected_pct": round(df["flag_duplicate"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Số bản ghi duy nhất hợp lệ",
            "kpi_before": float(n_present),
            "kpi_after": float((present & ~df["flag_duplicate"]).sum()),
        },
        {
            "table": "listings",
            "rule_name": "R9_Category_Mismatch",
            "category": "Danh mục lệch (Category Mismatch)",
            "condition": f"room_type ngoài {allowed_room_types}, thiếu property_type/borough, hoặc đổi room_type/neighbourhood_cleansed giữa các mốc",
            "reason": "Đảm bảo tính nhất quán của loại phòng và quận (borough) qua các mốc khi Nhóm C phân rã KPI",
            "action": "Gắn cờ flag_category_mismatch",
            "affected_rows": int(df["flag_category_mismatch"].sum()),
            "affected_pct": round(df["flag_category_mismatch"].sum() / max(n_present, 1) * 100, 2),
            "kpi_metric": "Tỷ lệ Entire home/apt hợp lệ (%)",
            "kpi_before": round(float((df.loc[present, "room_type"] == "Entire home/apt").mean() * 100), 2),
            "kpi_after": round(
                float((df.loc[present & ~df["flag_category_mismatch"], "room_type"] == "Entire home/apt").mean() * 100), 2
            ),
        },
    ]
    qa_report_df = pd.DataFrame(qa_rules)

    # =========================================================================
    # BƯỚC 7: LƯU 1 TỆP GỘP DUY NHẤT & CẬP NHẬT reports/qa_report.csv
    # =========================================================================
    drop_cols = ["amenities", "price", "price_raw_num", "flag_bath_or_type_issue"]
    df_out = df.drop(columns=[c for c in drop_cols if c in df.columns])

    proc_dir.mkdir(parents=True, exist_ok=True)
    combined_out = proc_dir / "listings_cleaned.csv"
    df_out.to_csv(combined_out, index=False, encoding="utf-8")

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
    ap = argparse.ArgumentParser(description="Chạy làm sạch & QA cho bảng listings")
    ap.add_argument("--config", default="configs/London.yaml", help="Đường dẫn tới file config YAML")
    args = ap.parse_args()

    df_clean, report = process_listings(args.config)
    print(f"\n[OK] Đã lưu tệp gộp duy nhất: data/processed/listings_cleaned.csv ({len(df_clean)} dòng)")
    print("[OK] Đã cập nhật báo cáo QA : reports/qa_report.csv\n")
    print(
        report[report["table"] == "listings"][
            ["rule_name", "affected_rows", "affected_pct", "kpi_metric", "kpi_before", "kpi_after"]
        ].to_string(index=False)
    )