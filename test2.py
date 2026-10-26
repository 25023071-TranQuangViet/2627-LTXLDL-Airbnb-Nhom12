import json
import pandas as pd

EMPTY = {"", "n/a", "na", "nan", "none", "null", "[]"}

def _num(s):
    return pd.to_numeric(s.str.replace(r"[$,%]", "", regex=True), errors="coerce")

def _json_list(s):
    def ok(x):
        try: return isinstance(json.loads(x), list)
        except Exception: return False
    return s.map(ok)

# validator nhận Series chuỗi KHÔNG rỗng, trả Series bool (True = hợp lệ)
VALIDATORS = {
    "price":                lambda s: _num(s) > 0,
    "number_of_reviews":    lambda s: _num(s) >= 0,
    "review_scores_rating": lambda s: _num(s).between(0, 5) | _num(s).between(0, 100),
    "bedrooms":             lambda s: _num(s) >= 0,
    "beds":                 lambda s: _num(s) >= 0,
    "bathrooms":            lambda s: _num(s) >= 0,
    "latitude":             lambda s: _num(s).between(-90, 90),
    "longitude":            lambda s: _num(s).between(-180, 180),
    "amenities":            _json_list,
}

def profile_columns(frames, cols, warn=0.05, fail=0.30):
    """frames: {snapshot: DataFrame đọc bằng dtype=str, keep_default_na=False}"""
    rows = []
    for snap, df in frames.items():
        n = len(df)
        for c in cols:
            if c not in df.columns:
                rows.append(dict(snapshot=snap, column=c, n_rows=n, empty_pct=1.0,
                                 invalid_pct=0.0, bad_pct=1.0, status="ABSENT"))
                continue
            s = df[c].astype("string")
            empty = s.isna() | s.str.strip().str.lower().isin(EMPTY)
            invalid = pd.Series(False, index=df.index)
            if c in VALIDATORS and (~empty).any():
                v = VALIDATORS[c](s[~empty]).fillna(False).astype(bool)
                invalid.loc[v.index] = ~v
            bad = (empty.sum() + invalid.sum()) / n
            status = "FAIL" if bad >= fail else "WARN" if bad >= warn else "OK"
            rows.append(dict(snapshot=snap, column=c, n_rows=n,
                             empty_pct=round(empty.mean(), 4),
                             invalid_pct=round(invalid.mean(), 4),
                             bad_pct=round(bad, 4), status=status))
    return pd.DataFrame(rows)

def kpi_readiness(health, kpi_cols):
    """kpi_cols: {tên KPI: [cột cần]} -> bảng KPI x snapshot"""
    st = health.pivot(index="column", columns="snapshot", values="status")
    out = {}
    for k, cols in kpi_cols.items():
        sub = st.loc[cols]
        out[k] = sub.apply(lambda col: "NOT_USABLE" if col.isin(["FAIL", "ABSENT"]).any()
                           else "PARTIAL" if (col == "WARN").any() else "USABLE")
    return pd.DataFrame(out).T