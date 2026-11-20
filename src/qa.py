import pandas as pd
import numpy as np
from pathlib import Path
import requests
import yaml
import re
import json

PROJECT_PATH = Path(__file__).resolve().parents[1]
NAN_TOKENS = {"", "nan", "n/a", "na", "null", "none"}

KEEP_COLS = {
        "listings": {"id", "last_scraped", "room_type", "neighbourhood_cleansed",
            "latitude", "longitude", "price", "accommodates", "minimum_nights", "maximum_nights",
            "amenities", "bathrooms", "bathrooms_text", "bedrooms", "beds",
            "number_of_reviews", "review_scores_rating", "review_scores_value",
            "host_id", "calculated_host_listings_count","has_availability", "availability_365"},
        "calendar":{"listing_id", "date", "available", "minimum_nights", "maximum_nights"},
        "reviews": {"listing_id", "id", "date", "reviewer_id", "comments"}
}

NUMERIC_COLS = {"listings":{"minimum_nights", "maximum_nights", "review_scores_rating",
                "accommodates", "latitude", "longitude"},
                "calendar": {"minimum_nights", "maximum_nights"},
                "reviews": {}
        }

def kiem_cau_truc(d: pd.DataFrame, cot_can: set, cot_so: set, moc: str = "") -> None:
        ten_moc = f" [{moc}]" if moc else ""
        thieu = cot_can - set(d.columns)
        if thieu:
                raise ValueError(f"Thiếu cột ở mốc{ten_moc}: {sorted(thieu)}")
        for cot in cot_so:
                if not pd.api.types.is_numeric_dtype(d[cot]):
                        raise TypeError(f"Cột {cot} (mốc {ten_moc}) phải có kiểu số, kiểu hiện tại là {d[cot].dtype}")

def check_input(config_path, root = PROJECT_PATH) -> None:
        with Path(config_path).open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        snapshot = config["snapshots"]
        loai_bang = ["listings", "calendar", "reviews"]
        for date in snapshot:
                for bang in loai_bang:
                        path_file = root / "data" / "raw" / f"{date}" / f"{bang}.csv.gz"
                        if not path_file.exists():
                                raise FileNotFoundError(f"Không thấy tệp thô: {path_file}")
                        d = pd.read_csv(path_file, compression="gzip", nrows=0)
                        kiem_cau_truc(d, KEEP_COLS[bang], set(), f"{date}")
                        if NUMERIC_COLS[bang]:
                                kiem_cau_truc(pd.read_csv(path_file, compression="gzip", usecols=list(NUMERIC_COLS[bang])), NUMERIC_COLS[bang], NUMERIC_COLS[bang], str(date))

def quy_tac(table, rule_name, category, condition, reason, action, affected_rows, affected_pct, kpi_metric, kpi_before, kpi_after):
        return {
                "table": table,
                "rule_name": rule_name,
                "category": category,
                "condition": condition,
                "reason": reason,
                "action": action,
                "affected_rows": affected_rows,
                "affected_pct":affected_pct,
                "kpi_metric": kpi_metric,
                "kpi_before": kpi_before,
                "kpi_after": kpi_after
        }

def clean_price_series(s: pd.Series) -> pd.Series:
        s_str = s.astype("string").str.strip()
        is_empty = s_str.isna() | s_str.str.lower().isin(NAN_TOKENS)
        cleaned = s_str[~is_empty].str.replace(r"[\$,]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

def extract_bathrooms_from_text(text_series: pd.Series) -> pd.Series:
        res = text_series.str.extract(r"^(\d+\.?\d*)\s*(?:shared|private)?\s*(?:baths?|bathrooms?)", flags = re.I, expand = False)
        nua_bath = (text_series.str.contains(r"half(?:-|\s)bath", regex = True))
        res[nua_bath] = "0.5"
        return pd.to_numeric(res, errors = "coerce")

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

def process_listings(config_path, root = PROJECT_PATH):
        with Path(config_path).open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        snapshot = config["snapshots"]
        last_snap = snapshot[-1]
        proc_dir = root / "data" / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        listing_loc = pd.read_csv(
        proc_dir / "listings-loc.csv",
        dtype={"listing_id": str},
        usecols=["listing_id", "analysis_area"],
        encoding="utf-8-sig",
        )
        vung_loc = pd.read_csv(proc_dir / "vung-loc.csv", encoding="utf-8-sig")
        for col in ["lon_min", "lat_min", "lon_max", "lat_max"]:
                vung_loc[col] = pd.to_numeric(vung_loc[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        
        frames = []
        for date in snapshot:
                listing_path = root / "data" / "raw" / f"{date}" / "listings.csv.gz"
                listing_raw = pd.read_csv(listing_path,
                            compression = "gzip",
                            dtype = {"id": str, "host_id": str},
                            usecols = list(KEEP_COLS["listings"]),
                            low_memory=False,
                        )
                listing_merged = listing_raw.merge(listing_loc, left_on = "id", right_on = "listing_id", how = "right").merge(vung_loc, on = "analysis_area", how = "left")
                listing_merged["snapshot"] = date
                frames.append(listing_merged)
        listings = pd.concat(frames, ignore_index=True)
        
# xu ly kieu du lieu
        raw_price_not_empty = ~(
                listings["price"].astype("string").isna()
                | listings["price"].astype("string").str.strip().str.lower().isin(NAN_TOKENS)
        )
        listings["price_raw_num"] = clean_price_series(listings["price"])
        #loc ra price cua moc cuoi vi chi dung price moc cuoi
        listings["price_clean"] = listings.loc[listings["snapshot"] == last_snap, "price_raw_num"]
        
        bath_num = pd.to_numeric(listings["bathrooms"], errors="coerce")
        bath_from_text = extract_bathrooms_from_text(listings["bathrooms_text"])
        text_not_empty = ~(
        listings["bathrooms_text"].astype("string").isna()
        | listings["bathrooms_text"].astype("string").str.strip().str.lower().isin(NAN_TOKENS)
        )
        listings["flag_wrong_type"] = (
        (raw_price_not_empty & listings["price_raw_num"].isna())
        | (text_not_empty & bath_from_text.isna() & bath_num.isna())
        )
        listings["bathrooms_clean"] = bath_num.fillna(bath_from_text)
        
        for num_col in [
        "accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights",
        "number_of_reviews", "review_scores_rating", "review_scores_value",
        "calculated_host_listings_count", "availability_365", "latitude", "longitude",
        ]:
                if num_col in listings.columns:
                        listings[num_col] = pd.to_numeric(listings[num_col], errors="coerce")

                if "has_availability" in listings.columns:
                        listings["has_availability_clean"] = listings["has_availability"].astype(str).str.lower().map(
                        {"t": True, "true": True, "f": False, "false": False}
                )
                else:
                        listings["has_availability_clean"] = np.nan
        listings["last_scraped"] = pd.to_datetime(listings["last_scraped"], format = "%Y-%m-%d")
        
        amen_listings = parse_amenities_features(listings["amenities"])
        listings["amenities_count"] = amen_listings["amenities_count"]
        listings["has_wifi"] = amen_listings["has_wifi"]
        listings["has_kitchen"] = amen_listings["has_kitchen"]
        
# gan co du lieu khuyet thieu
        listings["flag_missing_price_last"] = (listings["snapshot"] == last_snap) & listings["price_clean"].isna()
        listings["flag_no_reviews"] = (listings["number_of_reviews"] == 0)
        listings["flag_missing_scores"] = (listings["number_of_reviews"].isna()
        | listings["review_scores_rating"].isna()
        | listings["review_scores_value"].isna()
    )
        listings["flag_missing_bed"] = (listings["bedrooms"].isna() | listings["beds"].isna())
        listings["flag_missing_bath"] = (listings["bathrooms_clean"].isna())
# fillna
        mask_last = listings["snapshot"] == last_snap
        listings["price_clean_fillna"] = listings["price_clean"]
        tv_nhom = listings[mask_last].groupby(["room_type", "accommodates"])["price_clean"].transform("median")
        listings.loc[mask_last, "price_clean_fillna"] = listings.loc[mask_last, "price_clean"].fillna(tv_nhom)

        # Nhóm không có giá nào -> dùng trung vị toàn mốc cuối
        listings.loc[mask_last, "price_clean_fillna"] = listings.loc[mask_last, "price_clean_fillna"].fillna(listings.loc[mask_last, "price_clean"].median())
        
        bedroom_nhom = listings.groupby(["room_type", "accommodates"])["bedrooms"].transform("median")
        listings["bedrooms_clean"] = listings["bedrooms"].fillna(bedroom_nhom)
        listings["bedrooms_clean"] = listings["bedrooms_clean"].fillna(listings["bedrooms_clean"].median())

        bed_nhom = listings.groupby(["room_type", "accommodates"])["beds"].transform("median")
        listings["beds_clean"] = listings["beds"].fillna(bed_nhom)
        listings["beds_clean"] = listings["beds_clean"].fillna(listings["beds_clean"].median())

        bathroom_nhom = listings.groupby(["room_type"])["bathrooms_clean"].transform("median")
        listings["bathrooms_clean"] = listings["bathrooms_clean"].fillna(bathroom_nhom)
        listings["bathrooms_clean"] = listings["bathrooms_clean"].fillna(listings["bedrooms_clean"].median())
        
        
        
# gan co ngoai lai
        listings["flag_outlier_price"] = (mask_last
        & listings["price_clean"].notna()
        & ((listings["price_clean"] <= 0) | (listings["price_clean"] > config["qa"]["price_max_gbp"]))
        )
        
        listings["flag_outlier_min_nights"] = ((listings["minimum_nights"] <= 0)
        | (listings["minimum_nights"] > config["qa"]["min_nights_cap"])
        )
        listings["flag_out_of_bounds"] = ~(listings["longitude"].between(listings["lon_min"], listings["lon_max"])
        & listings["latitude"].between(listings["lat_min"], listings["lat_max"])
        )
        
        listings["min_nights_cap"] = listings["minimum_nights"]
        listings.loc[listings["flag_outlier_min_nights"], "min_nights_cap"] = config["qa"]["min_nights_cap"]
        
# gan co trung lap
        dup_across_snap = listings.duplicated(subset=config["qa"]["dedupe_keys"], keep="first")
        listings["flag_duplicate"] = dup_across_snap
        
# gan co loai phong co van de
        room_type_changed = listings.groupby("id")["room_type"].transform("nunique") > 1
        listings["flag_category_mismatch"] = room_type_changed
        
        
        last_listings = listings[listings["snapshot"] == last_snap]
        
        qa_ls_rules = [
                quy_tac("listings",
                        "R1_Wrong_Type_Bathrooms_Price", 
                        "Kiểu sai (Wrong Type)", 
                        "price chứa ký tự tiền tệ/không ép kiểu được hoặc bathrooms rỗng phải tách Regex từ bathrooms_text",
                        "Dữ liệu gốc lưu price dạng chuỗi '$1,200.00' và khuyết một số ô ở cột số bathrooms",
                        "Loại bỏ các ký tự tiền tệ và dấu phẩy và ép kiểu float cho price; dùng Regex tách số (kể cả Half-bath = 0.5) từ bathrooms_text sang bathrooms_clean",
                        int(listings["bathrooms"].isna().sum() + listings["flag_wrong_type"].sum()),
                        round((listings["bathrooms"].isna().sum() + listings["flag_wrong_type"].sum()) / len(listings) * 100, 2),
                        "Tỷ lệ đủ dữ liệu phòng tắm (%)",
                        round((1 - listings["bathrooms"].isna().mean()) * 100, 2),
                        round((1 - listings["bathrooms_clean"].isna().mean()) * 100, 2)
                        ),
                quy_tac("listings",
                        "R2_Missing_Price_Last_Snapshot",
                        "Thiếu dữ liệu (Missing)",
                        f"snapshot == '{last_snap}' và price IS NULL",
                        "Mốc cuối có 20 listing chưa cập nhật giá; các mốc trước nguồn giá không đồng đều (Mốc 2 rỗng 100%)",
                        "Chỉ dùng price ở mốc cuối; gắn cờ flag_missing_price_last và điền khuyết vào chỗ trống bằng trung vị giá của nhóm phòng cùng room_type và cùng sức chứa",
                        int(listings["flag_missing_price_last"].sum()),
                        round(listings["flag_missing_price_last"].sum() / len(last_listings) * 100, 2),
                        "Giá trung bình của listings trước và sau khi điền khuyết",
                        round(listings["price_clean"].mean(), 2),
                        round(listings["price_clean_fillna"].mean(), 2)
                        ),
                quy_tac("listings",
                        "R3_Missing_Bedrooms_Beds_Imputation",
                        "Thiếu dữ liệu (Missing)",
                        "bedrooms IS NULL hoặc beds IS NULL",
                        "Chủ nhà bỏ trống thông tin số phòng ngủ/giường (thường ở căn hộ Studio)",
                        "Gắn cờ flag_missing_bed; điền khuyết bằng trung vị (median) theo nhóm (room_type, accommodates)",
                        int(listings["flag_missing_bed"].sum()),
                        round(listings["flag_missing_bed"].mean() * 100, 2),
                        "Số phòng ngủ trung bình (bedrooms) trước và sau khi điền khuyết",
                        round(float(listings["bedrooms"].mean()), 3),
                        round(float(listings["bedrooms_clean"].mean()), 3)
                        ),
                quy_tac("listings",
                        "R4_No_Reviews",
                        "Thiếu dữ liệu (Missing)",
                        "number_of_reviews == 0",
                        "Số listing chưa có khách đánh giá theo mốc",
                        "Gắn cờ flag_no_reviews (tách biệt với cờ thiếu điểm đánh giá)",
                        int(listings["flag_no_reviews"].sum()),
                        round(listings["flag_no_reviews"].mean() * 100, 2),
                        "Số lượng đánh giá trung bình/listing (toàn bộ vs chỉ căn đã có đánh giá)",
                        round(float(listings["number_of_reviews"].mean()), 2),
                        round(float(listings.loc[~listings["flag_no_reviews"], "number_of_reviews"].mean()), 2)
                        ),
                quy_tac("listings",
                        "R5_Missing_Review_Scores",
                        "Thiếu dữ liệu (Missing)",
                        "review_scores_rating IS NULL hoặc review_scores_value IS NULL hoặc number_of_reviews IS NULL",
                        f"Listing thiếu điểm đánh giá ({int(listings['flag_missing_scores'].sum())} dòng trên 4 mốc",
                        "Gắn cờ flag_missing_score; loại trừ khi tính KPI điểm rating và điểm value trung bình",
                        int(listings["flag_missing_scores"].sum()),
                        round(listings["flag_missing_scores"].mean(), 2),
                        "Số bản ghi hợp lệ để tính điểm đánh giá (rating & value)",
                        int(len(listings)),
                        int((~listings["flag_missing_scores"]).sum())
                        ),
                quy_tac("listings",
                        "R6_Outlier_Price_Last_Snapshot",
                        "Ngoại lai (Outliers)",
                        f"snapshot == '{last_snap}' và (price_clean <= 0 hoặc price_clean > {config["qa"]["price_max_gbp"]})",
                        "Giá phòng niêm yết quá cao bất thường (giá ảo để khóa lịch) hoặc <= 0",
                        "Gắn cờ flag_outlier_price; loại khỏi phép tính Giá trung bình và Giá/người ở mốc cuối",
                        int(listings["flag_outlier_price"].sum()),
                        round(listings["flag_outlier_price"].sum() / mask_last.sum() * 100, 2),
                        "Giá phòng trung bình mốc cuối (GBP)",
                        round(float(listings["price_clean"].mean()), 2),
                        round(float(listings.loc[~listings["flag_missing_price_last"] & ~listings["flag_outlier_price"], "price_clean"].mean()), 2)
                        ),
                quy_tac("listings",
                        "R7_Outlier_Minimum_Nights",
                        "Ngoại lai (Outliers)",
                        f"minimum_nights <= 0 hoặc minimum_nights > {config["qa"]["min_nights_cap"]}",
                        "Yêu cầu số đêm tối thiểu vượt quá 1 năm (>365 ngày) không phải lưu trú ngắn hạn",
                        "Gắn cờ flag_outlier_min_nights và giới hạn trần (cap) ở mức min_nights_cap khi phân tích",
                        int(listings["flag_outlier_min_nights"].sum()),
                        round(listings["flag_outlier_min_nights"].mean() * 100, 2),
                        "Số đêm tối thiểu trung bình (ngày) trước và sau khi giới hạn trần",
                        round(float(listings["minimum_nights"].mean()), 2),
                        round(float(listings["min_nights_cap"].mean()), 2)
                        ),
                quy_tac("listings",
                        "R8_Outlier_Spatial_Bounds",
                        "Ngoại lai không gian (Spatial Outliers)",
                        "longitude ngoài [lon_min, lon_max] hoặc latitude ngoài [lat_min, lat_max] của vùng A01-A08",
                        "Kiểm tra tọa độ thực tế có bị lệch khỏi 8 vùng hình chữ nhật được phân công hay không",
                        "Gắn cờ flag_out_of_bounds",
                        int(listings["flag_out_of_bounds"].sum()),
                        round(listings["flag_out_of_bounds"].mean() * 100, 2),
                        "Số listing nằm đúng trong 8 vùng phân tích",
                        int(len(listings)),
                        int((~listings["flag_out_of_bounds"]).sum())
                        ),
                quy_tac("listings",
                        "R9_Duplicate_Records",
                        "Trùng lặp (Duplicates)",
                        "Trùng (id, last_scraped) giữa các mốc",
                        "Đảm bảo mỗi listing chỉ xuất hiện 1 lần ở mỗi mốc",
                        "Gắn cờ flag_duplicate (nếu có bản ghi trùng hoàn toàn thì giữ bản ghi đầu tiên)",
                        int(listings["flag_duplicate"].sum()),
                        round(listings["flag_duplicate"].mean() * 100, 2),
                        "Tổng số bản ghi duy nhất qua 4 mốc",
                        int(len(listings)),
                        int((~listings["flag_duplicate"]).sum())
                        ),
                quy_tac("listings",
                        "R10_Category_Mismatch_RoomType",
                        "Danh mục lệch (Category Mismatch)",
                        "Thay đổi loại phòng giữa các mốc",
                        "Kiểm tra tính nhất quán danh mục loại phòng qua 4 quý",
                        "Gắn cờ flag_category_mismatch",
                        int(listings["flag_category_mismatch"].sum()),
                        round(listings["flag_category_mismatch"].mean() * 100, 2),
                        "Số listings không bị đổi loại phòng",
                        int(len(listings)),
                        int((~listings["flag_category_mismatch"]).sum())
                        )
        ]
        
        qa_report_listings = pd.DataFrame(qa_ls_rules)
        
        combined_out = proc_dir / "listings_cleaned.csv"
        listings.to_csv(combined_out, index=False, encoding="utf-8")
        
        for date in snapshot:
                snap_dir = proc_dir / str(date)
                snap_dir.mkdir(parents=True, exist_ok=True)
                listing_snap = listings.loc[listings["snapshot"] == date]
                listing_snap.to_csv(snap_dir / "listing_cleaned.csv", index=False, encoding="utf-8")
                
        return qa_report_listings

def process_calendar(config_path, root=PROJECT_PATH):

    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    snapshot = config["snapshots"]
    proc_dir = root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # Đọc danh sách ID phân công từ listings-loc.csv
    listing_loc = pd.read_csv(
        proc_dir / "listings-loc.csv", 
        dtype={"listing_id": str}, 
        usecols=["listing_id"]
    )
    
    use_cols = ['listing_id', 'date', 'available', 'minimum_nights', 'maximum_nights']
    chunk_size = 500_000
    ids_set = set(listing_loc["listing_id"])
    all_snapshot_dfs = []

    # Đọc và lọc từng mốc theo Chunk
    for date_snap in snapshot:
        calendar_path = root / "data" / "raw" / str(date_snap) / "calendar.csv.gz"
        if not calendar_path.exists():
            continue
            
        chunks_for_snap = []
        for chunk in pd.read_csv(calendar_path, compression='gzip', dtype={'listing_id': 'str'}, usecols=use_cols, chunksize=chunk_size):
            # Lọc chunk có listing_id
            chunk =chunk[chunk["listing_id"].isin(ids_set)]
            if not chunk.empty:
                chunks_for_snap.append(chunk)
                
        if chunks_for_snap:
            snap_df = pd.concat(chunks_for_snap, ignore_index=True)
        # Merge với listing_loc
        snap_df= snap_df.merge(listing_loc, on= "listing_id", how="right",indicator=True)
        snap_df["flag_not_in_snapshot"]=snap_df["_merge"].eq("right_only")
        snap_df=snap_df.drop(columns="_merge")
        snap_df["snapshot"]=date_snap
        all_snapshot_dfs.append(snap_df)
    if not all_snapshot_dfs:
        raise ValueError("Không tìm thấy dữ liệu calendar hợp lệ ở các mốc!")

    # Concat toàn bộ thành 1 DataFrame tổng
    calendar = pd.concat(all_snapshot_dfs, ignore_index=True)

    # Ép kiểu dữ liệu và làm sạch
    calendar["available"] = calendar["available"].astype(str).str.lower().map({'t': True, 'f': False})
    calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d', errors="coerce")
    
    for num_col in ['minimum_nights', 'maximum_nights']:
        calendar[num_col] = pd.to_numeric(calendar[num_col], errors='coerce')

    # Kiểm tra Null, Ngoại lai, Điền chặn (Capping) & Trùng lặp
    cap_limit = config['qa']['min_nights_cap']
    
    calendar['flag_date_missing'] = calendar['date'].isna() & ~calendar["flag_not_in_snapshot"]
    calendar['flag_available_missing'] = calendar['available'].isna() & ~calendar["flag_not_in_snapshot"]
    
    calendar['flag_min_max_outlier'] = (
        (calendar['minimum_nights'] > cap_limit) | 
        (calendar['minimum_nights'] < 0) | 
        (calendar['maximum_nights'] < 0)
    )
    
    # Điền chặn ngoại lai (Capping)
    calendar['minimum_nights_cap'] = calendar['minimum_nights'].clip(lower=1, upper=cap_limit)
    calendar['maximum_nights_cap'] = calendar['maximum_nights'].clip(lower=1, upper=365)
    
    calendar['flag_min_gt_max'] = (calendar['minimum_nights'] > calendar['maximum_nights'])
    calendar['flag_duplicated'] = calendar.duplicated(subset=['listing_id', 'date', 'snapshot'], keep='first')

    # Tạo Báo cáo QA (Sử dụng hàm quy_tac chuẩn)
    qa_cal_rules = [
        quy_tac(
                "calendar",
                "CA1_Not_In_Snapshot",
                "Thiếu dữ liệu (Missing)",
                "Khi merge với listing_loc ['listing_id'] chỉ có đánh dấu xuất hiện bên phía listing_loc ",
                "Listing được phân công nhưng không có lịch ở mốc đó (bị gỡ hoặc chưa tạo)",
                "Giữ 1 dòng giữ chỗ cho mỗi listing thiếu, gắn cờ flag_not_in_snapshot; loại khỏi mọi tính toán KPI",
                int(calendar['flag_not_in_snapshot'].sum()),
                round((calendar['flag_not_in_snapshot'].sum()) / len(calendar) * 100, 2),
                "Số cặp (listing, mốc) có dữ liệu ",
                int(len(listing_loc) * len(snapshot)),
                int(len(listing_loc) * len(snapshot)  - calendar["flag_not_in_snapshot"].sum())
        ),
        quy_tac(
            "calendar",
            "CA2_Missing_Date_Available",
            "Thiếu dữ liệu (Missing)",
            "date IS NULL hoặc available IS NULL",
            "Khuyết thông tin ngày hoặc trạng thái lịch",
            "Gắn cờ flag_date_missing và flag_available_missing",
            int(calendar['flag_date_missing'].sum() + calendar['flag_available_missing'].sum()),
            round((calendar['flag_date_missing'].sum() + calendar['flag_available_missing'].sum()) / len(calendar) * 100, 2),
            "Tổng số bản ghi calendar hợp lệ",
            int(len(calendar)),
            int(len(calendar) - (calendar['flag_date_missing'].sum() + calendar['flag_available_missing'].sum()))
        ),
        quy_tac(
            "calendar",
            "CA3_Min_Max_Nights_Outliers",
            "Ngoại lai (Outliers)",
            f"minimum_nights < 0 hoặc minimum_nights > {cap_limit}",
            "Số đêm tối thiểu/tối đa âm hoặc vượt trần cấu hình",
            "Gắn cờ flag_min_max_outlier; điền chặn trần minimum_nights_cap và maximum_nights_cap",
            int(calendar['flag_min_max_outlier'].sum()),
            round(calendar['flag_min_max_outlier'].mean() * 100, 2),
            "Số đêm tối thiểu trung bình trước và sau khi điền chặn trần",
            round(float(calendar['minimum_nights'].mean()), 2),
            round(float(calendar['minimum_nights_cap'].mean()), 2)
        ),
        quy_tac(
            "calendar",
            "CA4_Min_Nights_Greater_Than_Max",
            "Mâu thuẫn logic (Logic Inconsistency)",
            "minimum_nights > maximum_nights",
            "Số đêm tối thiểu thiết lập lớn hơn số đêm tối đa",
            "Gắn cờ flag_min_gt_max để lọc/hiệu chỉnh",
            int(calendar['flag_min_gt_max'].sum()),
            round(calendar['flag_min_gt_max'].mean() * 100, 2),
            "Tỷ lệ bản ghi đúng logic min <= max (%)",
            round((1 - calendar['flag_min_gt_max'].mean()) * 100, 2),
            100.0
        ),
        quy_tac(
            "calendar",
            "CA5_Duplicate_Calendar_Entries",
            "Trùng lặp (Duplicates)",
            "Trùng (listing_id, date, snapshot)",
            "Nhiều bản ghi ngày lịch trùng lặp trong cùng một mốc",
            "Gắn cờ flag_duplicated và giữ lại bản ghi đầu tiên",
            int(calendar['flag_duplicated'].sum()),
            round(calendar['flag_duplicated'].mean() * 100, 2),
            "Số lượng bản ghi calendar duy nhất",
            int(len(calendar)),
            int((~calendar['flag_duplicated']).sum())
        )
    ]
    
    qa_report_calendar = pd.DataFrame(qa_cal_rules)
    
    # Xuất file tổng hợp vào data/processed/
    combined_out = proc_dir / "calendar_cleaned.csv"
    calendar.to_csv(combined_out, index=False, encoding="utf-8")
    
    # Lặp qua từng mốc snapshot để tạo thư mục riêng và lưu calendar_cleaned.csv
    for date in snapshot:
        snap_dir = proc_dir / str(date)
        snap_dir.mkdir(parents=True, exist_ok=True)
        calendar_snap = calendar.loc[calendar["snapshot"] == date]
        calendar_snap.to_csv(snap_dir / "calendar_cleaned.csv", index=False, encoding="utf-8")

    return qa_report_calendar

def process_reviews(config_path, root: Path = PROJECT_PATH):
        """Đọc, lọc, làm sạch, gắn cờ QA cho reviews ở cả 4 mốc và tạo báo cáo QA."""


        with Path(config_path).open("r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
        snapshots = config["snapshots"]
        last_snap = snapshots[-1]
        proc_dir = root / "data" / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        review_loc = pd.read_csv(
        proc_dir / "reviews-loc.csv",
        dtype= str,
        encoding="utf-8-sig",
        )
        review_loc = review_loc.rename(columns={"listing_id": "listing_id_loc"})
        cot_loc = ["review_id", "listing_id_loc"] + [c for c in ("Mẫu LLM", "Nhãn tay") if c in review_loc.columns]


        # review_loc = pd.read_csv(proc_dir / cfg.get("reviews_ids", "reviews-loc.csv"), dtype=str,
        #                             encoding="utf-8-sig")
        # review_loc = review_loc.rename(columns={"listing_id": "listing_id_loc"})
        # cot_loc = ["review_id", "listing_id_loc"] + [c for c in ("Mẫu LLM", "Nhãn tay") if c in review_loc.columns]

        review_loc = review_loc[cot_loc].drop_duplicates("review_id")
        frames = []
        for snap in snapshots:
                path = root / "data" / "raw" / f"{snap}" / "reviews.csv.gz"
                raw = pd.read_csv(path, compression="gzip", dtype=str,
                                usecols=lambda c: c in {"listing_id", "id", "date", "reviewer_id", "comments"})
                m = raw.merge(review_loc, left_on="id", right_on="review_id", how="right")
                m["flag_not_in_snapshot"] = m["id"].isna()
                m["id"] = m["id"].fillna(m["review_id"])
                m["snapshot"] = snap
                frames.append(m)
        df = pd.concat(frames, ignore_index=True)
        n_total = len(df)
        snap_dt = pd.to_datetime(df["snapshot"], format = "%Y-%m-%d", errors = "coerce")


        # # ['listing_id',
        #  'id',
        #  'date',
        #  'reviewer_id',
        #  'comments',
        #  'review_id',
        #  'listing_id_loc',
        #  'Mẫu LLM',
        #  'Nhãn tay',
        #  'flag_not_in_snapshot',
        #  'snapshot']
        


        # gán cờ cho Mẫu LLM , Nhãn tay (na = False, not na = True): 
        df['flag_LLM'] = df['Mẫu LLM'].notna()
        df['flag_human'] = df['Nhãn tay'].notna()


        # Thay "<br/>" và '\r' bằng dấu cách để 2 câu không dính vào nhau khi tìm từ khóa : 
        co_br = df['comments'].str.contains(r"\r?<br/>", regex = True)
        n_html = co_br.sum()
        df["comments_fixed"] = (
        df["comments"]
        .str.replace(r"\r?<br/>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)   
        )
        n_html_after = int(df["comments_fixed"].str.contains(r"\r?<br/>", regex=True, na=False).sum())

        # Gắn cờ cho các review có comment quá ngắn (dưới min_chars ký tự):
        df["flag_short_comment"] = df["comments"].notna() & (df["comments"].str.len() < config["qa"]["min_chars"])

        # thiếu:  
        
        df["flag_empty_comment"] = ~df["flag_not_in_snapshot"] & (df["comments"].isna() | df["comments"].str.strip().str.lower().isin(NAN_TOKENS) | df["comments"].str.strip().eq("."))

        # trùng lặp review_id trong df
        
        df["flag_id_duplicates"] = df.duplicated(subset=['review_id', 'date'], keep='first')
        
        # Báo cáo QA:
        use = ~(df["flag_empty_comment"] | df["flag_not_in_snapshot"])
        df["usable_for_llm"] = use
        con_lai = df[~df["flag_not_in_snapshot"]]
        qa_rules = [
                quy_tac("reviews" ,"REV1_Not_In_Snapshot", "Thiếu dữ liệu (Missing)",
                        "review_id trong reviews-loc.csv không có trong tệp reviews của mốc",
                        "Review tìm được ở mốc nào thì mới đưa vào mốc đó",
                        "Giữ 1 dòng, gắn cờ flag_not_in_snapshot",
                        df["flag_not_in_snapshot"].sum(), df["flag_not_in_snapshot"].mean(), "Số review tìm thấy",
                        int(n_total), int(len(con_lai))),
                
                quy_tac("reviews", "REV2_Html_In_Comments", "Kiểu sai (Wrong Type)",
                        "comments chứa thẻ <br/> hoặc khoảng trắng thừa",
                        "Nội dung gốc lưu dạng HTML thô",
                        "Thay <br/> bằng khoảng trắng, gộp khoảng trắng -> comments",
                        n_html, co_br.mean(), "Số review còn thẻ HTML", n_html,
                        n_html_after),


                quy_tac("reviews", "REV3_Empty_Comment", "Thiếu dữ liệu (Missing)",
                        "comments rỗng sau khi làm sạch",
                        "Review không có nội dung thì LLM không phân loại được ồn/an toàn",
                        "Gắn cờ flag_empty_comment; loại khỏi mẫu LLM khi tính KPI",
                        df["flag_empty_comment"].sum(), df["flag_empty_comment"].mean(), "Số review có nội dung",
                        len(df.loc[~df["flag_not_in_snapshot"] ,"comments"]),
                        (~df["flag_empty_comment"]).sum()),

                quy_tac("reviews", "REV4_Short_Comment", "Ngoại lai (Outliers)",
                        f"độ dài comments_clean < {config["qa"]["min_chars"]} ký tự",
                        "Quá ngắn để phân loại",
                        "Gắn cờ flag_short_comment",
                        df["flag_short_comment"].sum(), df["flag_short_comment"].mean(), "Độ dài review trung bình (ký tự)",
                        round(float(con_lai["comments"].str.len().mean()), 1),
                        round(float(con_lai.loc[~con_lai["flag_short_comment"], "comments"].str.len().mean()), 1)),
                quy_tac("reviews", "RE5_Duplicates", "Trùng lặp",
                        "bị trùng lặp review_id giữa các mốc sau khi ghép",
                        "bị trùng lặp review_id giữa các mốc sau khi ghép",
                        "gắn cờ trùng lặp",
                        df["flag_id_duplicates"].sum(), df["flag_id_duplicates"].mean(), "Số review độc nhất",
                        len(df),
                        (~df["flag_id_duplicates"]).sum())
        ]
        
        report = pd.DataFrame(qa_rules)



        # Lưu kết quả:

        cot_ra = [
                'snapshot',
                'listing_id',
                'review_id',
                'date',
                'reviewer_id',
                'comments_fixed',
                'listing_id_loc',
                'Mẫu LLM',
                'Nhãn tay',
                'flag_not_in_snapshot',
                'flag_short_comment',
                'flag_empty_comment',
                "flag_id_duplicates",
                'flag_LLM',
                'flag_human',
                'usable_for_llm'
                ]
        df_out = con_lai[cot_ra]
        combined_out = proc_dir / "reviews_cleaned.csv"
        df_out.to_csv(combined_out, index=False, encoding="utf-8")
        for date in snapshots:
                snap_dir = proc_dir / str(date)
                snap_dir.mkdir(parents=True, exist_ok=True)
                review_snap = df_out.loc[df_out["snapshot"] == date]
                review_snap.to_csv(snap_dir / "reviews_cleaned.csv", index=False, encoding="utf-8")
        return report

def run_qa(config_path, root = PROJECT_PATH):
        check_input(config_path, root)
        print("Cau truc dau vao da OKK")
        
        reports = [process_listings(config_path, root), process_calendar(config_path, root), process_reviews(config_path, root)]
        qa_report = pd.concat(reports, ignore_index=True)
        report_path = root / "reports"
        report_path.mkdir(parents = True, exist_ok = True)
        qa_report.to_csv(report_path / "qa_report.csv", index=False, encoding="utf-8-sig")
        return qa_report
