"""Xử lý làm sạch, điền khuyết và kiểm tra chất lượng (QA) riêng cho bảng listings.

Chạy độc lập để kiểm thử (từ thư mục gốc repo):
    python src/qa_listings.py --config configs/London.yaml

Đầu ra khi chạy thành công:
    1. data/processed/listings_cleaned.csv        (Gộp 4 mốc = 4.000 dòng đã làm sạch & gắn cờ)
    2. data/processed/<mốc>/listing_cleaned.csv   (Tách theo từng mốc để tương thích code cũ)
    3. reports/qa_listings_report.csv             (Bảng báo cáo 5 nhóm quy tắc QA kèm KPI trước/sau)
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

# Danh sách các cột cần giữ lại từ listings.csv.gz gốc (bỏ qua các cột >=90% NaN)
KEEP_COLS = {
    # Định danh, thời gian, không gian & kiểm tra tự động
    "id", "last_scraped", "room_type", "neighbourhood_cleansed",
    "latitude", "longitude",
    # Giá & sức chứa
    "price", "accommodates", "minimum_nights", "maximum_nights",
    # Tiện ích & phòng ốc
    "amenities", "bathrooms", "bathrooms_text", "bedrooms", "beds",
    # Đánh giá
    "number_of_reviews", "review_scores_rating", "review_scores_value",
    # Chủ nhà & độ sẵn sàng
    "host_id", "calculated_host_listings_count",
    "has_availability", "availability_365",
}


# ----------------------------------------------------------------------------- Các hàm tiện ích làm sạch
def clean_price_series(s: pd.Series) -> pd.Series:
    """Chuyển chuỗi tiền tệ '$1,234.00' sang float."""
    s_str = s.astype("string").str.strip()
    is_empty = s_str.isna() | s_str.str.lower().isin(NAN_TOKENS)
    cleaned = s_str.where(~is_empty, np.nan).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def extract_bathrooms_from_text(text_series: pd.Series) -> pd.Series:
    """Trích xuất số lượng phòng tắm từ bathrooms_text (bao gồm cả 'Half-bath' -> 0.5)."""
    def parse_one(val) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if not s or s in NAN_TOKENS:
            return np.nan
        # Trường hợp "half-bath", "shared half-bath", "private half-bath"
        if "half-bath" in s or "half bath" in s:
            return 0.5
        # Tìm con số đầu tiên (ví dụ: "1.5 shared baths" -> 1.5)
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else np.nan

    return text_series.map(parse_one)


def parse_amenities_features(amenities_series: pd.Series) -> pd.DataFrame:
    """Trích xuất số lượng tiện ích, cờ có Wifi và cờ có Bếp (Kitchen) từ chuỗi JSON."""
    counts, wifi, kitchen = [], [], []
    for val in amenities_series:
        if pd.isna(val):
            counts.append(0)
            wifi.append(False)
            kitchen.append(False)
            continue
        try:
            items = json.loads(str(val))
            if not isinstance(items, list):
                items = []
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


# ----------------------------------------------------------------------------- Hàm xử lý chính
def process_listings(config_path: str | Path, root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc, lọc, làm sạch, gắn cờ QA cho bảng listings ở cả 4 mốc và tạo báo cáo QA."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    snapshots = [str(d) for d in cfg["snapshots"]]
    last_snap = snapshots[-1]

    qa_cfg = cfg.get("qa", {}) or {}
    price_max = float(qa_cfg.get("price_max_gbp", qa_cfg.get("price_max", 5000.0)))
    min_nights_cap = int(qa_cfg.get("min_nights_cap", 365))
    allowed_room_types = qa_cfg.get("room_types", ["Entire home/apt", "Private room"])

    proc_dir = root / cfg.get("processed_dir", "data/processed")
    # 1. Đọc danh sách 1.000 ID của nhóm và tọa độ 8 vùng A01-A08
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
        # Lọc đúng 1.000 listing của nhóm và gắn analysis_area + khung tọa độ vùng
        df_merged = (
            df_raw.merge(listing_loc, left_on="id", right_on="listing_id", how="right")
            .merge(vung_loc[["analysis_area", "lon_min", "lat_min", "lon_max", "lat_max"]],
                   on="analysis_area", how="left")
        )
        df_merged["snapshot"] = snap
        frames.append(df_merged)

    df = pd.concat(frames, ignore_index=True)

    # =========================================================================
    # BƯỚC 1: PHÁT HIỆN & XỬ LÝ KIỂU SAI (WRONG TYPE) + TRÍCH XUẤT ĐẶC TRƯNG
    # =========================================================================
    # 1.1. Giá phòng (price): lưu bản gốc để kiểm tra lỗi ép kiểu
    raw_price_non_empty = ~(
        df["price"].astype("string").isna()
        | df["price"].astype("string").str.strip().str.lower().isin(NAN_TOKENS)
    )
    df["price_raw_num"] = clean_price_series(df["price"])
    # Chỉ sử dụng price ở mốc cuối (2026-06-19) theo đúng quy định đề bài
    df["price_clean"] = np.where(df["snapshot"] == last_snap, df["price_raw_num"], np.nan)

    # 1.2. Phòng tắm (bathrooms): lấy từ cột số, nếu thiếu thì trích xuất bằng Regex từ bathrooms_text
    bath_num = pd.to_numeric(df["bathrooms"], errors="coerce")
    bath_from_text = extract_bathrooms_from_text(df["bathrooms_text"])
    # Cờ kiểu sai: có chữ trong bathrooms_text nhưng không parse được ra số hợp lệ
    text_non_empty = ~(
        df["bathrooms_text"].astype("string").isna()
        | df["bathrooms_text"].astype("string").str.strip().str.lower().isin(NAN_TOKENS)
    )
    df["flag_wrong_type"] = (
        (raw_price_non_empty & df["price_raw_num"].isna())
        | (text_non_empty & bath_from_text.isna() & bath_num.isna())
    )
    df["bathrooms_clean"] = bath_num.fillna(bath_from_text)

    # 1.3. Chuyển các cột số & boolean chuẩn
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

    # 1.4. Trích xuất tiện ích từ amenities
    amen_df = parse_amenities_features(df["amenities"])
    df["amenities_count"] = amen_df["amenities_count"]
    df["has_wifi"] = amen_df["has_wifi"]
    df["has_kitchen"] = amen_df["has_kitchen"]

    # =========================================================================
    # BƯỚC 2: GẮN CỜ DỮ LIỆU THIẾU (MISSING) & ĐIỀN KHUYẾT (IMPUTATION)
    # =========================================================================
    # 2.1. Thiếu price ở mốc cuối (kỳ vọng đúng 20 dòng ở mốc 2026-06-19)
    df["flag_missing_price_last"] = (df["snapshot"] == last_snap) & df["price_clean"].isna()

    # 2.2. Listing chưa có review (50, 30, 25, 25 dòng) -> thiếu điểm rating
    df["flag_no_reviews"] = (df["number_of_reviews"].fillna(0) == 0) | df["review_scores_rating"].isna()

    # 2.3. Thiếu bedrooms, beds hoặc bathrooms (trước khi điền khuyết)
    df["flag_missing_bed_bath"] = (
        df["bedrooms"].isna() | df["beds"].isna() | df["bathrooms_clean"].isna()
    )

    # Lưu bản gốc trước khi điền khuyết để đối chiếu KPI trước/sau QA
    df["bedrooms_raw"] = df["bedrooms"]
    df["beds_raw"] = df["beds"]

    # Điền khuyết bedrooms và beds bằng trung vị (median) theo (room_type, accommodates)
    df["bedrooms_clean"] = df.groupby(["room_type", "accommodates"])["bedrooms"].transform(
        lambda s: s.fillna(s.median())
    )
    df["bedrooms_clean"] = df["bedrooms_clean"].fillna(df["bedrooms"].median()).fillna(1.0)

    df["beds_clean"] = df.groupby(["room_type", "accommodates"])["beds"].transform(
        lambda s: s.fillna(s.median())
    )
    df["beds_clean"] = df["beds_clean"].fillna(df["beds"].median()).fillna(1.0)

    df["bathrooms_clean"] = df.groupby("room_type")["bathrooms_clean"].transform(
        lambda s: s.fillna(s.median())
    ).fillna(1.0)

    # =========================================================================
    # BƯỚC 3: GẮN CỜ NGOẠI LAI (OUTLIERS)
    # =========================================================================
    # 3.1. Ngoại lai giá ở mốc cuối (price <= 0 hoặc > price_max)
    df["flag_outlier_price"] = (
        (df["snapshot"] == last_snap)
        & df["price_clean"].notna()
        & ((df["price_clean"] <= 0) | (df["price_clean"] > price_max))
    )

    # 3.2. Ngoại lai số đêm tối thiểu (minimum_nights > 365 hoặc <= 0)
    df["flag_outlier_min_nights"] = (
        df["minimum_nights"].isna()
        | (df["minimum_nights"] <= 0)
        | (df["minimum_nights"] > min_nights_cap)
    )

    # 3.3. Ngoại lai không gian: tọa độ nằm ngoài khung hình chữ nhật của vùng A01-A08
    df["flag_out_of_bounds"] = ~(
        df["longitude"].between(df["lon_min"], df["lon_max"])
        & df["latitude"].between(df["lat_min"], df["lat_max"])
    )

    # Tính sẵn Giá trên mỗi khách (price_per_person) ở mốc cuối cho các dòng hợp lệ (Nhóm C - Câu 1)
    valid_for_ppp = (
        (df["snapshot"] == last_snap)
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
    # BƯỚC 4: GẮN CỜ TRÙNG LẶP (DUPLICATES - TRONG VÀ GIỮA CÁC MỐC)
    # =========================================================================
    # Trùng id trong cùng 1 mốc HOẶC trùng cặp (id, last_scraped) giữa các mốc khác nhau
    dup_in_snap = df.duplicated(subset=["snapshot", "id"], keep="first")
    dup_across_snap = df.duplicated(subset=["id", "last_scraped"], keep="first")
    df["flag_duplicate"] = dup_in_snap | dup_across_snap

    # =========================================================================
    # BƯỚC 5: GẮN CỜ DANH MỤC LỆCH (CATEGORY MISMATCH)
    # =========================================================================
    # room_type không nằm trong danh sách chuẩn HOẶC 1 listing bị đổi room_type giữa 4 mốc
    room_type_changed = df.groupby("id")["room_type"].transform("nunique") > 1
    df["flag_category_mismatch"] = (~df["room_type"].isin(allowed_room_types)) | room_type_changed

    # =========================================================================
    # BƯỚC 6: TỔNG HỢP BÁO CÁO QA (reports/qa_listings_report.csv)
    # =========================================================================
    last_df = df[df["snapshot"] == last_snap]
    valid_price_df = last_df[~last_df["flag_missing_price_last"] & ~last_df["flag_outlier_price"]]

    qa_rules = [
        {
            "rule_name": "R1_Wrong_Type_Bathrooms_Price",
            "category": "Kiểu sai (Wrong Type)",
            "condition": "price chứa ký tự tiền tệ/không ép kiểu được hoặc bathrooms rỗng phải tách Regex từ bathrooms_text",
            "reason": "Dữ liệu gốc lưu price dạng chuỗi '$1,200.00' và khuyết cột số bathrooms (6.0% - 8.5%)",
            "action": "Ép kiểu float cho price; dùng Regex tách số (kể cả Half-bath = 0.5) từ bathrooms_text sang bathrooms_clean",
            "affected_rows": int(df["bathrooms"].isna().sum() + df["flag_wrong_type"].sum()),
            "affected_pct": round((df["bathrooms"].isna().sum() + df["flag_wrong_type"].sum()) / len(df) * 100, 2),
            "kpi_metric": "Tỷ lệ đủ dữ liệu phòng tắm (%)",
            "kpi_before": round((1 - df["bathrooms"].isna().mean()) * 100, 2),
            "kpi_after": round((1 - df["bathrooms_clean"].isna().mean()) * 100, 2),
        },
        {
            "rule_name": "R2_Missing_Price_Last_Snapshot",
            "category": "Thiếu dữ liệu (Missing)",
            "condition": f"snapshot == '{last_snap}' và price IS NULL",
            "reason": "Mốc cuối có 20 listing chưa cập nhật giá; các mốc trước nguồn giá không đồng đều (Mốc 2 rỗng 100%)",
            "action": "Chỉ dùng price ở mốc cuối; gắn cờ flag_missing_price_last để loại trừ khi tính KPI giá nhưng giữ lại khi đếm tổng cung",
            "affected_rows": int(df["flag_missing_price_last"].sum()),
            "affected_pct": round(df["flag_missing_price_last"].sum() / len(last_df) * 100, 2),
            "kpi_metric": "Số listing mốc cuối hợp lệ để tính Giá trung vị",
            "kpi_before": int(len(last_df)),
            "kpi_after": int((~last_df["flag_missing_price_last"]).sum()),
        },
        {
            "rule_name": "R3_Missing_Bedrooms_Beds_Imputation",
            "category": "Thiếu dữ liệu (Missing)",
            "condition": "bedrooms IS NULL (14.8% ở mốc 4) hoặc beds IS NULL (7.3% ở mốc 1, 6.0% ở mốc 2)",
            "reason": "Chủ nhà bỏ trống thông tin số phòng ngủ/giường (thường ở căn hộ Studio)",
            "action": "Gắn cờ flag_missing_bed_bath; điền khuyết bằng trung vị (median) theo nhóm (room_type, accommodates)",
            "affected_rows": int(df["flag_missing_bed_bath"].sum()),
            "affected_pct": round(df["flag_missing_bed_bath"].mean() * 100, 2),
            "kpi_metric": "Số phòng ngủ trung bình (bedrooms)",
            "kpi_before": round(float(df["bedrooms_raw"].mean()), 3),
            "kpi_after": round(float(df["bedrooms_clean"].mean()), 3),
        },
        {
            "rule_name": "R4_Missing_Review_Scores",
            "category": "Thiếu dữ liệu (Missing)",
            "condition": "number_of_reviews == 0 hoặc review_scores_rating IS NULL",
            "reason": "Listing mới đăng chưa từng có khách đánh giá (50, 30, 25, 25 căn qua 4 mốc)",
            "action": "Gắn cờ flag_no_reviews; không điền khuyết điểm ảo, loại trừ khi tính KPI điểm đánh giá trung bình",
            "affected_rows": int(df["flag_no_reviews"].sum()),
            "affected_pct": round(df["flag_no_reviews"].mean() * 100, 2),
            "kpi_metric": "Điểm đánh giá trung bình (nếu gán 0 vs bỏ căn chưa có review)",
            "kpi_before": round(float(df["review_scores_rating"].fillna(0).mean()), 3),
            "kpi_after": round(float(df.loc[~df["flag_no_reviews"], "review_scores_rating"].mean()), 3),
        },
        {
            "rule_name": "R5_Outlier_Price_Last_Snapshot",
            "category": "Ngoại lai (Outliers)",
            "condition": f"snapshot == '{last_snap}' và (price_clean <= 0 hoặc price_clean > {price_max})",
            "reason": "Giá phòng niêm yết quá cao bất thường (giá ảo để khóa lịch) hoặc <= 0",
            "action": "Gắn cờ flag_outlier_price; loại khỏi phép tính Giá trung bình và Giá/người ở mốc cuối",
            "affected_rows": int(df["flag_outlier_price"].sum()),
            "affected_pct": round(df["flag_outlier_price"].sum() / len(last_df) * 100, 2),
            "kpi_metric": "Giá phòng trung bình mốc cuối (GBP)",
            "kpi_before": round(float(last_df["price_clean"].mean()), 2),
            "kpi_after": round(float(valid_price_df["price_clean"].mean()), 2),
        },
        {
            "rule_name": "R6_Outlier_Minimum_Nights",
            "category": "Ngoại lai (Outliers)",
            "condition": f"minimum_nights <= 0 hoặc minimum_nights > {min_nights_cap}",
            "reason": "Yêu cầu số đêm tối thiểu vượt quá 1 năm (>365 ngày) không phải lưu trú ngắn hạn",
            "action": "Gắn cờ flag_outlier_min_nights và giới hạn trần (cap) ở mức min_nights_cap khi phân tích",
            "affected_rows": int(df["flag_outlier_min_nights"].sum()),
            "affected_pct": round(df["flag_outlier_min_nights"].mean() * 100, 2),
            "kpi_metric": "Số đêm tối thiểu trung bình (ngày)",
            "kpi_before": round(float(df["minimum_nights"].mean()), 2),
            "kpi_after": round(float(df.loc[~df["flag_outlier_min_nights"], "minimum_nights"].mean()), 2),
        },
        {
            "rule_name": "R7_Outlier_Spatial_Bounds",
            "category": "Ngoại lai không gian (Spatial Outliers)",
            "condition": "longitude ngoài [lon_min, lon_max] hoặc latitude ngoài [lat_min, lat_max] của vùng A01-A08",
            "reason": "Kiểm tra tọa độ thực tế có bị lệch khỏi 8 vùng hình chữ nhật được phân công hay không",
            "action": "Gắn cờ flag_out_of_bounds",
            "affected_rows": int(df["flag_out_of_bounds"].sum()),
            "affected_pct": round(df["flag_out_of_bounds"].mean() * 100, 2),
            "kpi_metric": "Số listing nằm đúng trong 8 vùng phân tích",
            "kpi_before": int(len(df)),
            "kpi_after": int((~df["flag_out_of_bounds"]).sum()),
        },
        {
            "rule_name": "R8_Duplicate_Records",
            "category": "Trùng lặp (Duplicates)",
            "condition": "Trùng (snapshot, id) trong cùng mốc hoặc trùng (id, last_scraped) giữa các mốc",
            "reason": "Đảm bảo mỗi listing chỉ xuất hiện 1 lần ở mỗi mốc và được cào mới ở các quý tiếp theo",
            "action": "Gắn cờ flag_duplicate (nếu có bản ghi trùng hoàn toàn thì giữ bản ghi đầu tiên)",
            "affected_rows": int(df["flag_duplicate"].sum()),
            "affected_pct": round(df["flag_duplicate"].mean() * 100, 2),
            "kpi_metric": "Tổng số bản ghi duy nhất qua 4 mốc",
            "kpi_before": int(len(df)),
            "kpi_after": int((~df["flag_duplicate"]).sum()),
        },
        {
            "rule_name": "R9_Category_Mismatch_RoomType",
            "category": "Danh mục lệch (Category Mismatch)",
            "condition": f"room_type không thuộc {allowed_room_types} hoặc thay đổi loại phòng giữa các mốc",
            "reason": "Kiểm tra tính nhất quán danh mục loại phòng qua 4 quý",
            "action": "Gắn cờ flag_category_mismatch",
            "affected_rows": int(df["flag_category_mismatch"].sum()),
            "affected_pct": round(df["flag_category_mismatch"].mean() * 100, 2),
            "kpi_metric": "Tỷ lệ Entire home/apt hợp lệ (%)",
            "kpi_before": round(float((df["room_type"] == "Entire home/apt").mean() * 100), 2),
            "kpi_after": round(
                float((df.loc[~df["flag_category_mismatch"], "room_type"] == "Entire home/apt").mean() * 100), 2
            ),
        },
    ]
    qa_report_df = pd.DataFrame(qa_rules)

    # =========================================================================
    # BƯỚC 7: LƯU DỮ LIỆU ĐÃ LÀM SẠCH & BÁO CÁO QA
    # =========================================================================
    # Bỏ cột chuỗi amenities dài và các cột trung gian để giảm dung lượng file dưới 50MB
    drop_cols = ["amenities", "listing_id", "bedrooms_raw", "beds_raw"]
    df_out = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 1. Lưu 1 file gộp cả 4 mốc (rất tiện cho bạn code kpi.py và viz.py)
    proc_dir.mkdir(parents=True, exist_ok=True)
    combined_out = proc_dir / "listings_cleaned.csv"
    df_out.to_csv(combined_out, index=False, encoding="utf-8")

    # 2. Lưu tách theo từng thư mục mốc (để tương thích với cấu trúc cũ trong src/qa.py)
    for snap, g in df_out.groupby("snapshot"):
        snap_dir = proc_dir / str(snap)
        snap_dir.mkdir(parents=True, exist_ok=True)
        g.to_csv(snap_dir / "listing_cleaned.csv", index=False, encoding="utf-8")

    # 3. Lưu báo cáo QA của bảng listings
    rep_dir = root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    qa_report_path = rep_dir / "qa_listings_report.csv"
    qa_report_df.to_csv(qa_report_path, index=False, encoding="utf-8")

    return df_out, qa_report_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chạy thử nghiệm làm sạch & QA cho bảng listings")
    ap.add_argument("--config", default="configs/London.yaml", help="Đường dẫn tới file config YAML")
    args = ap.parse_args()

    df_clean, report = process_listings(args.config)
    print(f"\n[OK] Đã xử lý xong {len(df_clean)} dòng listings (4 mốc).")
    print(f"[OK] Đã lưu dữ liệu sạch tại: data/processed/listings_cleaned.csv")
    print(f"[OK] Đã lưu báo cáo QA tại : reports/qa_listings_report.csv\n")
    print("=== TÓM TẮT KẾT QUẢ QA TRÊN BẢNG LISTINGS ===")
    print(
        report[["rule_name", "affected_rows", "affected_pct", "kpi_metric", "kpi_before", "kpi_after"]]
        .to_string(index=False)
    )