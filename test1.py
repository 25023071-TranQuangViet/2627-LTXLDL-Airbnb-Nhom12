from pathlib import Path
import pandas as pd
import yaml

with open("configs/London.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

listings_loc = pd.read_csv("data/processed/listings-loc.csv", dtype={"listing_id": str})
reviews_loc = pd.read_csv("data/processed/reviews-loc.csv", dtype={"review_id": str, "listing_id": str})

ma_du_lieu = listings_loc["Mã dữ liệu"].iloc[0]  # DSE2049-C04-G12-london
listing_ids = set(listings_loc["listing_id"])
review_ids = set(reviews_loc["review_id"])

rows_part1 = []
nan90_by_snapshot = {}

for idx, ngay in enumerate(config["snapshots"], start=1):
    raw_dir = Path(f"data/raw/{ngay}")
    f_list = raw_dir / "listings.csv.gz"
    f_cal = raw_dir / "calendar.csv.gz"
    f_rev = raw_dir / "reviews.csv.gz"

    # Kiểm tra tải về và dung lượng bytes
    tai_ve_ok = "co" if (f_list.exists() and f_cal.exists() and f_rev.exists()) else "khong"
    l_bytes = f_list.stat().st_size if f_list.exists() else 0
    c_bytes = f_cal.stat().st_size if f_cal.exists() else 0
    r_bytes = f_rev.stat().st_size if f_rev.exists() else 0

    # 1. Đọc và lọc listings (đọc toàn bộ cột để đếm so_cot_listings và so_cot_nan_90)
    df_l = pd.read_csv(f_list, compression="gzip", dtype={"id": str}, low_memory=False)
    df_l_sub = df_l[df_l["id"].isin(listing_ids)].copy()

    so_cot_listings = df_l_sub.shape[1]
    nan_ratio = df_l_sub.isna().mean()
    cols_nan_90 = nan_ratio[nan_ratio >= 0.90].index.tolist()
    nan90_by_snapshot[ngay] = cols_nan_90

    # 2. Đọc và lọc calendar theo chunksize
    cal_dong = 0
    for chunk in pd.read_csv(f_cal, compression="gzip", usecols=["listing_id"], dtype={"listing_id": str}, chunksize=500_000):
        cal_dong += chunk["listing_id"].isin(listing_ids).sum()

    # 3. Đọc và lọc reviews
    df_r = pd.read_csv(f_rev, compression="gzip", usecols=["id"], dtype={"id": str})
    df_r_sub = df_r[df_r["id"].isin(review_ids)]
    rev_dong = len(df_r_sub)

    # Kiểm tra ở mốc cuối số lượng khớp Mẫu LLM và Nhãn tay
    if idx == 4:
        llm_ids = set(reviews_loc[reviews_loc["Mẫu LLM"] == "x"]["review_id"])
        hand_ids = set(reviews_loc[reviews_loc["Nhãn tay"] == "x"]["review_id"])
        print(f"[Câu 2] Số review khớp Mẫu LLM: {df_r_sub['id'].isin(llm_ids).sum()} / 1000")
        print(f"[Câu 2] Số review khớp Nhãn tay: {df_r_sub['id'].isin(hand_ids).sum()} / 100")

    rows_part1.append({
        "ma_du_lieu": ma_du_lieu,
        "moc": idx,
        "ngay": str(ngay),
        "tai_ve_ok": tai_ve_ok,
        "listings_bytes": l_bytes,
        "calendar_bytes": c_bytes,
        "reviews_bytes": r_bytes,
        "listings_dong": len(df_l_sub),
        "calendar_dong": int(cal_dong),
        "reviews_dong": rev_dong,
        "entire_home": int((df_l_sub["room_type"] == "Entire home/apt").sum()),
        "private_room": int((df_l_sub["room_type"] == "Private room").sum()),
        "khong_co_review": int((df_l_sub["number_of_reviews"] == 0).sum()),
        "thieu_price": int(df_l_sub["price"].isna().sum()),
        "so_cot_listings": so_cot_listings,
        "so_cot_nan_90": len(cols_nan_90),
    })

out_csv = f"check_{ma_du_lieu}.csv"
pd.DataFrame(rows_part1).to_csv(out_csv, index=False)
print(f"\nĐã xuất file: {out_csv}")

print("\n[Câu 3] Danh sách các cột có >= 90% NaN theo từng mốc:")
for ngay, cols in nan90_by_snapshot.items():
    print(f"- Mốc {ngay} ({len(cols)} cột): {', '.join(cols)}")