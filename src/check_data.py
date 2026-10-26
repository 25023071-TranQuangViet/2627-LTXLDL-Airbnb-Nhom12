"""Kiểm tra dữ liệu thô của nhóm trước khi làm QA.

Kiểm tra, với từng mốc chụp:
  - 3 tệp listings / calendar / reviews có tải đủ và đọc được không
  - các cột listings nhóm định dùng (GROUP_COLS): rỗng bao nhiêu %, sai kiểu / vô lý bao nhiêu %
  - 3 cột đầu vào cho LLM (review_id, listing_id, analysis_area) + nội dung review (comments)
  - các số liệu theo biểu mẫu "Báo cáo kiểm tra dữ liệu BTL Airbnb"

Chạy (từ thư mục gốc repo):
    python src/check_data.py --config configs/London.yaml
    python src/check_data.py --config configs/London.yaml --skip-calendar   # nhanh hơn

Dữ liệu thô: data/raw/<yyyy-mm-dd>/{listings,calendar,reviews}.csv.gz
Đầu ra (thư mục reports/):
    column_health.csv      cột x mốc: % rỗng, % không hợp lệ, trạng thái OK/WARN/FAIL/ABSENT
    check_<Mã dữ liệu>.csv đúng mẫu Phần 1 của biểu mẫu giảng viên
    check_summary.md       bản nháp câu trả lời cho 5 câu hỏi Phần 2
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLES = ["listings", "calendar", "reviews"]
CHUNK = 50_000

# Chuỗi được coi là rỗng (NaN) - khớp cách biểu mẫu tính "ô rỗng = NaN"
NAN_TOKENS = {"", "nan", "n/a", "na", "null", "none"}

# Cột listings nhóm định dùng cho KPI / phân tích (sửa ở đây hoặc khai `check.group_cols` trong config)
GROUP_COLS = [
    "id", "room_type", "neighbourhood_cleansed",
    "price",
    "number_of_reviews", "review_scores_rating",
    "amenities", "bathrooms", "bathrooms_text", "bedrooms", "beds",
    "latitude", "longitude",
    "accommodates", "review_scores_value",   # Nhóm C: giá/người và điểm value
]
# Cột calendar cần cho Nhóm C (giá và độ sẵn sàng theo mùa)
CALENDAR_COLS = ["date", "available", "price"]
# Giá trị kỳ vọng theo đề; chỉ dùng để CẢNH BÁO, không làm chương trình dừng
DEFAULT_EXPECTED = {
    "listings": 1000, "calendar": 365000, "reviews": 5000,
    "entire_home": 875, "private_room": 125,
    "no_review": [50, 30, 25, 25], "missing_price_last": 20,
    "llm_sample": 1000, "hand_label": 100,
}


# ----------------------------------------------------------------------------- tiện ích
def is_empty(s: pd.Series, extra: set[str] | None = None) -> pd.Series:
    s = s.astype("string")
    tokens = NAN_TOKENS | (extra or set())
    return (s.isna() | s.str.strip().str.lower().isin(tokens)).fillna(True).astype(bool)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(r"[$,%]", "", regex=True), errors="coerce")


def json_list_ok(s: pd.Series) -> pd.Series:
    def ok(x: str) -> bool:
        try:
            return isinstance(json.loads(x), list)
        except Exception:
            return False
    return s.map(ok)


def build_validators(allowed_room_types: list[str]) -> dict:
    """validator nhận Series chuỗi KHÔNG rỗng, trả Series bool (True = hợp lệ)"""
    return {
        "price": lambda s: to_num(s) > 0,
        "number_of_reviews": lambda s: to_num(s) >= 0,
        "review_scores_rating": lambda s: to_num(s).between(0, 100),
        "bathrooms": lambda s: to_num(s) >= 0,
        "bedrooms": lambda s: to_num(s) >= 0,
        "beds": lambda s: to_num(s) >= 0,
        "latitude": lambda s: to_num(s).between(-90, 90),
        "longitude": lambda s: to_num(s).between(-180, 180),
        "amenities": json_list_ok,
        "room_type": lambda s: s.isin(allowed_room_types),
        "accommodates": lambda s: to_num(s) > 0,
        "review_scores_value": lambda s: to_num(s).between(0, 5),
        # "1 bath", "1.5 shared baths"... ; "Half-bath" KHÔNG khớp -> bị tính là không hợp lệ
        "bathrooms_text": lambda s: s.str.contains(
            r"^\s*\d+(?:\.\d+)?\s*(?:shared |private )?(?:baths?|bathrooms?)", case=False, regex=True),
    }


def health_row(snap: str, table: str, col: str, n: int,
               empty: pd.Series | None, invalid: pd.Series | None,
               warn: float, fail: float, absent: bool = False) -> dict:
    if absent or n == 0:
        return dict(snapshot=snap, table=table, column=col, n_rows=n, empty_n=n, empty_pct=1.0,
                    invalid_n=0, invalid_pct=0.0, bad_pct=1.0,
                    status="ABSENT" if absent else "FAIL")
    e = int(empty.sum())
    i = int(invalid.sum())
    bad = (e + i) / n
    status = "FAIL" if bad >= fail else "WARN" if bad >= warn else "OK"
    return dict(snapshot=snap, table=table, column=col, n_rows=n,
                empty_n=e, empty_pct=round(e / n, 4), invalid_n=i,
                invalid_pct=round(i / n, 4), bad_pct=round(bad, 4), status=status)


def read_filtered(path: Path, key: str, keep_ids: set[str], usecols=None) -> pd.DataFrame:
    """Đọc .csv.gz theo khối, chỉ giữ dòng có `key` nằm trong keep_ids (không tràn RAM)."""
    parts = []
    reader = pd.read_csv(path, compression="gzip", dtype=str, keep_default_na=False,
                         usecols=usecols, chunksize=CHUNK)
    for chunk in reader:
        parts.append(chunk[chunk[key].isin(keep_ids)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def file_ok(path: Path) -> str | None:
    """None nếu ổn, ngược lại trả mô tả lỗi."""
    if not path.exists():
        return "thieu_tep"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            f.readline()
    except Exception as e:  # noqa: BLE001
        return f"khong_doc_duoc ({type(e).__name__})"
    return None


# ----------------------------------------------------------------------------- chính
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=str(ROOT), help="thư mục gốc repo (mặc định: tự suy ra)")
    ap.add_argument("--skip-calendar", action="store_true", help="bỏ qua đếm dòng calendar (file rất lớn)")
    args = ap.parse_args()

    root = Path(args.root)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    chk = cfg.get("check", {}) or {}
    warn, fail = chk.get("warn", 0.05), chk.get("fail", 0.30)
    group_cols = chk.get("group_cols", GROUP_COLS)
    expected = {**DEFAULT_EXPECTED, **(cfg.get("expected", {}) or {})}
    allowed_rt = (cfg.get("qa", {}) or {}).get("room_types", ["Entire home/apt", "Private room"])
    snapshots = [str(d) for d in cfg["snapshots"]]  # YAML có thể parse ngày thành kiểu date
    validators = build_validators(allowed_rt)

    proc = root / cfg.get("processed_dir", "data/processed")
    lst_loc = pd.read_csv(proc / cfg.get("listings_ids", "listings-loc.csv"), dtype=str, encoding="utf-8-sig")
    rev_loc = pd.read_csv(proc / cfg.get("reviews_ids", "reviews-loc.csv"), dtype=str, encoding="utf-8-sig")
    ma = lst_loc["Mã dữ liệu"].dropna().iloc[0] if "Mã dữ liệu" in lst_loc else cfg.get("city", "nhom")
    listing_ids = set(lst_loc["listing_id"])
    review_ids = set(rev_loc["review_id"])
    is_x = lambda s: s.fillna("").astype(str).str.strip().str.lower().eq("x")  # noqa: E731
    sample_ids = set(rev_loc.loc[is_x(rev_loc["Mẫu LLM"]), "review_id"])
    hand_ids = set(rev_loc.loc[is_x(rev_loc["Nhãn tay"]), "review_id"])
    area_of = lst_loc.set_index("listing_id")["analysis_area"]

    print(f"Mã dữ liệu: {ma} | ID listings: {len(listing_ids)} | review: {len(review_ids)} "
          f"| mẫu LLM: {len(sample_ids)} | nhãn tay: {len(hand_ids)}")

    health: list[dict] = []
    check_rows: list[dict] = []
    issues: list[str] = []
    nan90: dict[str, list[str]] = {}
    sample_info: dict[str, tuple[int, int]] = {}
    cal_range: dict[str, tuple[str, str, int]] = {}

    for idx, snap in enumerate(snapshots, start=1):
        d = root / "data" / "raw" / snap
        print(f"\n=== Mốc {idx}: {snap}")
        row = dict(ma_du_lieu=ma, moc=idx, ngay=snap, tai_ve_ok="co",
                   listings_bytes="", calendar_bytes="", reviews_bytes="",
                   listings_dong="", calendar_dong="", reviews_dong="",
                   entire_home="", private_room="", khong_co_review="", thieu_price="",
                   so_cot_listings="", so_cot_nan_90="")
        bad_files: list[str] = []
        for t in TABLES:
            p = d / f"{t}.csv.gz"
            err = file_ok(p)
            if err:
                bad_files.append(f"{t}: {err}")
                issues.append(f"Mốc {snap} - {t}.csv.gz: {err}")
            if p.exists():
                row[f"{t}_bytes"] = p.stat().st_size

        # ---------------- listings
        lp = d / "listings.csv.gz"
        lst = pd.DataFrame()
        if not any(b.startswith("listings") for b in bad_files):
            try:
                lst = read_filtered(lp, "id", listing_ids)
                with gzip.open(lp, "rt", encoding="utf-8") as f:
                    header_cols = pd.read_csv(f, nrows=0).columns.tolist()
                row["so_cot_listings"] = len(header_cols)
                row["listings_dong"] = len(lst)
                if len(lst) != expected["listings"]:
                    issues.append(f"Mốc {snap}: tìm thấy {len(lst)} listing (kỳ vọng {expected['listings']})")
                if len(lst):
                    row["entire_home"] = int((lst["room_type"] == "Entire home/apt").sum())
                    row["private_room"] = int((lst["room_type"] == "Private room").sum())
                    row["khong_co_review"] = int((to_num(lst["number_of_reviews"]) == 0).sum())
                    row["thieu_price"] = int(is_empty(lst["price"]).sum())
                    frac = {c: float(is_empty(lst[c], {"[]"} if c == "amenities" else None).mean())
                            for c in lst.columns}
                    nan90[snap] = sorted(c for c, v in frac.items() if v >= 0.9)
                    row["so_cot_nan_90"] = len(nan90[snap])
                # sức khoẻ các cột nhóm định dùng
                for c in group_cols:
                    if c not in lst.columns:
                        health.append(health_row(snap, "listings", c, len(lst), None, None, warn, fail, absent=True))
                        continue
                    s = lst[c]
                    empty = is_empty(s, {"[]"} if c == "amenities" else None)
                    invalid = pd.Series(False, index=s.index)
                    if c in validators and (~empty).any():
                        v = validators[c](s[~empty].astype("string")).fillna(False).astype(bool)
                        invalid.loc[v.index] = ~v
                    health.append(health_row(snap, "listings", c, len(lst), empty, invalid, warn, fail))
            except Exception as e:  # noqa: BLE001
                bad_files.append(f"listings: loi_xu_ly ({type(e).__name__}: {e})")
                issues.append(f"Mốc {snap} - listings: lỗi xử lý ({type(e).__name__}: {e})")

        # ---------------- reviews + 3 cột LLM
        rp = d / "reviews.csv.gz"
        if not any(b.startswith("reviews") for b in bad_files):
            try:
                rv = read_filtered(rp, "id", review_ids,
                                   usecols=lambda c: c in {"id", "listing_id", "comments"})
                row["reviews_dong"] = len(rv)
                if len(rv) != expected["reviews"]:
                    issues.append(f"Mốc {snap}: tìm thấy {len(rv)} review (kỳ vọng {expected['reviews']})")
                if "comments" not in rv.columns:
                    issues.append(f"Mốc {snap}: reviews không có cột comments")
                    rv["comments"] = ""
                rv = rv.drop_duplicates("id").set_index("id")
                s_df = rev_loc[rev_loc["review_id"].isin(sample_ids)].copy()
                s_df["raw_listing"] = s_df["review_id"].map(rv["listing_id"]) if len(rv) else pd.NA
                s_df["comments"] = s_df["review_id"].map(rv["comments"]) if len(rv) else pd.NA
                s_df["area"] = s_df["listing_id"].map(area_of)
                found = int(s_df["raw_listing"].notna().sum())
                hand_found = int(rev_loc[rev_loc["review_id"].isin(hand_ids)]["review_id"]
                                 .isin(rv.index).sum())
                sample_info[snap] = (found, hand_found)
                if found != expected["llm_sample"]:
                    issues.append(f"Mốc {snap}: mẫu LLM tìm được {found} (kỳ vọng {expected['llm_sample']})")
                if hand_found != expected["hand_label"]:
                    issues.append(f"Mốc {snap}: nhãn tay tìm được {hand_found} (kỳ vọng {expected['hand_label']})")
                n = len(s_df)
                health.append(health_row(snap, "llm", "review_id", n, s_df["raw_listing"].isna(),
                                         pd.Series(False, index=s_df.index), warn, fail))
                mism = s_df["raw_listing"].notna() & (s_df["raw_listing"] != s_df["listing_id"])
                health.append(health_row(snap, "llm", "listing_id", n, s_df["raw_listing"].isna(),
                                         mism, warn, fail))
                health.append(health_row(snap, "llm", "analysis_area", n, is_empty(s_df["area"]),
                                         pd.Series(False, index=s_df.index), warn, fail))
                health.append(health_row(snap, "llm", "comments", n, is_empty(s_df["comments"]),
                                         pd.Series(False, index=s_df.index), warn, fail))
            except Exception as e:  # noqa: BLE001
                bad_files.append(f"reviews: loi_xu_ly ({type(e).__name__}: {e})")
                issues.append(f"Mốc {snap} - reviews: lỗi xử lý ({type(e).__name__}: {e})")

        # ---------------- calendar
        cp = d / "calendar.csv.gz"
        if not args.skip_calendar and not any(b.startswith("calendar") for b in bad_files):
            try:
                cal = read_filtered(cp, "listing_id", listing_ids,
                                    usecols=lambda c: c in {"listing_id", *CALENDAR_COLS})
                rows = len(cal)
                n_listing = cal["listing_id"].nunique() if rows else 0
                row["calendar_dong"] = rows
                if rows != expected["calendar"]:
                    issues.append(f"Mốc {snap}: calendar có {rows} dòng của nhóm (kỳ vọng {expected['calendar']}); "
                                  f"{n_listing} listing có lịch")
                if rows and "date" in cal.columns:
                    dts = pd.to_datetime(cal["date"], errors="coerce")
                    cal_range[snap] = (str(dts.min().date()), str(dts.max().date()), int(dts.nunique()))
                for c in CALENDAR_COLS:
                    if c not in cal.columns:
                        health.append(health_row(snap, "calendar", c, rows, None, None, warn, fail, absent=True))
                        continue
                    s = cal[c]
                    empty = is_empty(s)
                    invalid = pd.Series(False, index=s.index)
                    if (~empty).any():
                        x = s[~empty].astype("string")
                        if c == "available":
                            v = x.isin(["t", "f"])
                        elif c == "price":
                            v = to_num(x) > 0
                        else:
                            v = pd.to_datetime(x, errors="coerce").notna()
                        invalid.loc[v.index] = ~v.fillna(False).astype(bool)
                    health.append(health_row(snap, "calendar", c, rows, empty, invalid, warn, fail))
            except Exception as e:  # noqa: BLE001
                bad_files.append(f"calendar: loi_xu_ly ({type(e).__name__}: {e})")
                issues.append(f"Mốc {snap} - calendar: lỗi xử lý ({type(e).__name__}: {e})")

        if bad_files:
            row["tai_ve_ok"] = "; ".join(bad_files)
        check_rows.append(row)
        print({k: v for k, v in row.items() if k not in ("ma_du_lieu",)})

    # ---- cảnh báo theo kỳ vọng (không dừng chương trình)
    for i, r in enumerate(check_rows):
        if r["entire_home"] != "" and r["entire_home"] != expected["entire_home"]:
            issues.append(f"Mốc {r['ngay']}: Entire home/apt = {r['entire_home']} (kỳ vọng {expected['entire_home']})")
        if r["private_room"] != "" and r["private_room"] != expected["private_room"]:
            issues.append(f"Mốc {r['ngay']}: Private room = {r['private_room']} (kỳ vọng {expected['private_room']})")
        exp_nr = expected["no_review"]
        if r["khong_co_review"] != "" and i < len(exp_nr) and r["khong_co_review"] != exp_nr[i]:
            issues.append(f"Mốc {r['ngay']}: không có review = {r['khong_co_review']} (kỳ vọng {exp_nr[i]})")
    last = check_rows[-1]
    if last["thieu_price"] != "" and last["thieu_price"] != expected["missing_price_last"]:
        issues.append(f"Mốc cuối {last['ngay']}: thiếu price = {last['thieu_price']} (kỳ vọng {expected['missing_price_last']})")

    # ---- xuất tệp
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    h = pd.DataFrame(health)
    h.to_csv(out / "column_health.csv", index=False, encoding="utf-8")
    pd.DataFrame(check_rows).to_csv(out / f"check_{ma}.csv", index=False, encoding="utf-8")

    summary = build_summary(ma, snapshots, check_rows, h, nan90, sample_info, issues, expected, cal_range)
    (out / "check_summary.md").write_text(summary, encoding="utf-8")

    # ---- in ra màn hình
    if len(h):
        pv = h.pivot_table(index=["table", "column"], columns="snapshot", values="status", aggfunc="first")
        pv = pv.reindex(columns=snapshots)
        print("\n=== Trạng thái cột x mốc (OK <5% lỗi, WARN 5-30%, FAIL >=30%, ABSENT = không có cột)")
        print(pv.fillna("-").to_string())
    print("\n" + summary)
    print(f"\nĐã ghi: {out / 'column_health.csv'}, {out / f'check_{ma}.csv'}, {out / 'check_summary.md'}")


def build_summary(ma, snapshots, check_rows, h, nan90, sample_info, issues, expected, cal_range) -> str:
    L: list[str] = [f"# Bản nháp báo cáo kiểm tra dữ liệu - {ma}", ""]
    L.append("## 1. Tệp không tải được / không đọc được")
    bad = [f"- Mốc {r['ngay']}: {r['tai_ve_ok']}" for r in check_rows if r["tai_ve_ok"] != "co"]
    L += bad or ["- Không có (cả 3 tệp của 4 mốc đều tồn tại và đọc được)."]
    L.append("  (Mã lỗi HTTP khi tải, ví dụ 403, lấy từ bước download.py; script này chỉ kiểm tệp đã có trên đĩa.)")
    L += ["", "## 2. Mẫu LLM và nhãn tay tìm được"]
    for s in snapshots:
        if s in sample_info:
            f, hf = sample_info[s]
            L.append(f"- Mốc {s}: mẫu LLM {f}/{expected['llm_sample']}, nhãn tay {hf}/{expected['hand_label']}")
        else:
            L.append(f"- Mốc {s}: không kiểm được")
    L += ["", "## 3. Cột listings có từ 90% NaN trở lên"]
    for s in snapshots:
        cols = nan90.get(s)
        L.append(f"- Mốc {s}: " + ("không kiểm được" if cols is None else ", ".join(cols) if cols else "không có"))
    L += ["", "## 4. Cột nhóm định dùng bị rỗng / lỗi nhiều"]
    probs = h[h["status"].isin(["FAIL", "ABSENT"])] if len(h) else h
    if len(probs):
        for (t, c), g in probs.groupby(["table", "column"]):
            L.append(f"- `{c}` ({t}): " + ", ".join(f"{r.snapshot} [{r.status}, lỗi {r.bad_pct:.0%}]"
                                                      for r in g.itertuples()))
        L.append("  Hướng xử lý (nhóm chọn): bỏ cột / chỉ dùng ở mốc có dữ liệu / thay bằng cột khác "
                 "(ví dụ bathrooms -> bathrooms_text). `price` theo đề chỉ dùng ở mốc cuối.")
    else:
        L.append("- Không có cột nào FAIL/ABSENT.")
    warns = h[h["status"] == "WARN"] if len(h) else h
    if len(warns):
        L.append("- Mức WARN (lỗi 5-30%): " + "; ".join(sorted({f"{r.column}@{r.snapshot}" for r in warns.itertuples()})))
    if cal_range:
        L += ["", "## Phạm vi ngày của calendar (để làm chuỗi theo mùa/tháng)"]
        L += [f"- Mốc {s}: {a} -> {b} ({n} ngày)" for s, (a, b, n) in cal_range.items()]
    L += ["", "## 5. Gợi ý kết luận"]
    L += [f"- {x}" for x in issues] or ["- Mọi số đếm khớp kỳ vọng."]
    ok = (not issues) and (not len(probs))
    L.append("**Gợi ý: đủ để làm.**" if ok else
             "**Gợi ý: xem lại các mục trên; nếu lỗi nằm ở cột bắt buộc ở mốc không thay thế được thì báo giảng viên cấp lại.**")
    return "\n".join(L)


if __name__ == "__main__":
    main()