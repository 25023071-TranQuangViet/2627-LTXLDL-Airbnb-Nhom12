"""viz.py - Vẽ hình K1-K5 từ các file kpi_*.csv (Inside Airbnb).

Đầu vào  : data/processed/kpi_borough_value.csv, kpi_snapshot.csv, kpi_availability_month.csv
Đầu ra   : figures/K1_...png ... K5_...png
Chạy     : python src/viz.py --config configs/London.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")  # chỉ lưu file, không mở cửa sổ
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Bảng màu sáng sủa, tương ứng với màu cam điểm nhấn (#E8890C)
XANH_SANG = "#4DB6C6"
XAM_SANG = "#B8C2CC"
XAM_NHAT = "#C5CED6"
CAM = "#E8890C"
XANH_DAT = "#15803D"
NET_DUT = "#7A8691"
INK = "#222222"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E9EB",
    "grid.alpha": 0.65,
    "axes.axisbelow": True,
    "axes.titleweight": "bold",
    "axes.titlesize": 10.5,
})


# ----------------------------------------------------------------------------- tiện ích
def so(v, d=1):
    """Định dạng số kiểu Việt Nam: 4.64 -> '4,64'."""
    return f"{v:.{d}f}".replace(".", ",")


def ngay(s):
    """'2025-09-14' -> '14/09/2025'."""
    return f"{s[8:10]}/{s[5:7]}/{s[:4]}"


def ghi_nguon(fig, thanh_pho, cac_moc, dong_them, y=0.0):
    """Chân trang: nguồn, mốc chụp, phạm vi và giới hạn."""
    nguon = (f"Nguồn: Inside Airbnb (CC BY 4.0), {thanh_pho}, {len(cac_moc)} mốc "
             f"{ngay(cac_moc[0])} – {ngay(cac_moc[-1])}.")
    fig.text(0.01, y, nguon + "\n" + dong_them, fontsize=7.5, color="#555555", va="top", linespacing=1.45)


def luu(fig, thu_muc, ten):
    thu_muc.mkdir(parents=True, exist_ok=True)
    p = thu_muc / f"{ten}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] đã lưu {p}")


# ----------------------------------------------------------------------------- K1
def ve_k1(b, tt, moc, tien_te, out):
    """K1: giá trung vị / khách theo borough, có khoảng tin cậy 95%."""
    r = b[b["reliable"]].sort_values("ppp_median")
    city = float(b["city_ppp_median"].iloc[0])
    y = np.arange(len(r))
    top3 = r["ppp_median"] >= r["ppp_median"].nlargest(3).min()

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.hlines(y, r["ppp_ci_low"], r["ppp_ci_high"], color=XANH_SANG, lw=2.2, alpha=0.85)
    ax.scatter(r["ppp_median"], y, marker="o", color=np.where(top3, CAM, XANH_SANG),
               s=52, zorder=3, edgecolor=INK, lw=0.35)
    ax.axvline(city, color=NET_DUT, ls="--", lw=1.1, alpha=0.45)

    ax.set_yticks(y)
    ax.set_yticklabels(r["neighbourhood_cleansed"], fontsize=8.5)
    ax.set_ylim(-0.7, len(r) - 0.3)
    ax.set_xlabel(f"Giá trung vị mỗi khách ({tien_te}/khách/đêm); nét đứt mờ = trung vị {tt} ({so(city, 0)} {tien_te})")

    t = r.nlargest(2, "ppp_median")
    hon = (t["ppp_median"] / city - 1) * 100
    ax.set_title(f"K1 · {t.iloc[0]['neighbourhood_cleansed']} ({so(t.iloc[0]['ppp_median'], 0)} {tien_te}) và "
                 f"{t.iloc[1]['neighbourhood_cleansed']} ({so(t.iloc[1]['ppp_median'], 0)}) cao hơn trung vị {tt} "
                 f"({so(city, 0)}) khoảng {so(hon.min(), 0)}–{so(hon.max(), 0)}%", fontsize=10, loc="left")
    ghi_nguon(fig, tt, moc,
              f"Phạm vi: {len(r)}/{len(b)} borough đủ mẫu; giá chỉ dùng mốc cuối {ngay(moc[-1])}; thanh ngang = KTC 95%.\n"
              "Giới hạn: khoảng tin cậy nhiều borough chồng nhau, chỉ nhóm đầu và đáy thật sự tách biệt. Cam = 3 borough đắt nhất.",
              y=0.02)
    luu(fig, out, "K1_ppp_borough")


# ----------------------------------------------------------------------------- K2
def ve_k2(b, tt, moc, out):
    """K2: điểm value trung bình theo borough - dot-plot + khoảng tin cậy 95%."""
    r = b[b["reliable"]].dropna(subset=["value_mean"]).sort_values("value_mean")
    city = float(b["city_value_mean"].iloc[0])
    y = np.arange(len(r))
    top3 = r["value_mean"] >= r["value_mean"].nlargest(3).min()

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ax.hlines(y, r["value_ci_low"], r["value_ci_high"], color=XAM_SANG, lw=2.2, alpha=0.9)
    ax.scatter(r["value_mean"], y, marker="o", color=np.where(top3, CAM, XANH_SANG),
               s=52, zorder=3, edgecolor=INK, lw=0.35)
    ax.axvline(city, color=NET_DUT, ls="--", lw=1.1, alpha=0.45)

    ax.set_yticks(y)
    ax.set_yticklabels(r["neighbourhood_cleansed"], fontsize=8.5)
    ax.set_ylim(-0.7, len(r) - 0.3)
    ax.set_xlabel(f"Điểm value trung bình (thang 0–5); nét đứt mờ = TB {tt} ({so(city, 2)}); thanh = KTC 95%")
    ax.set_title(f"K2 · Điểm value các borough dao động {so(r['value_mean'].min(), 2)}–{so(r['value_mean'].max(), 2)}/5: "
                 "chênh lệch nhỏ, khoảng tin cậy chồng nhau", fontsize=10, loc="left")
    ghi_nguon(fig, tt, moc,
              f"Phạm vi: {len(r)} borough đủ mẫu (listing có ≥5 review); cam = 3 borough điểm cao nhất.\n"
              "Giới hạn: điểm value là cảm nhận của khách đã đánh giá, không phải của mọi khách.",
              y=0.02)
    luu(fig, out, "K2_value_borough")


# ----------------------------------------------------------------------------- K3
def ve_k3(b, tt, moc, tien_te, out):
    """K3: scatter chấm tròn; chỉ ghi tên nhóm Đạt; màu sắc tách biệt rõ giữa các nhóm."""
    r = b[b["reliable"]]
    cp, cv = float(b["city_ppp_median"].iloc[0]), float(b["city_value_mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    cau_hinh_nhom = [
        ("Đạt (Chắc chắn)", CAM, 0.95),
        ("Đạt (Sát ngưỡng)", XANH_DAT, 0.92),
        ("Không đạt", XAM_NHAT, 0.80),
    ]
    for nhom, mau, alpha in cau_hinh_nhom:
        s = r[r["good_value"] == nhom]
        ax.scatter(s["ppp_median"], s["value_mean"], s=s["n_listing"] * 1.6,
                   marker="o", color=mau, alpha=alpha,
                   edgecolor=INK, lw=0.45, label=f"{nhom} ({len(s)})", zorder=3)

    ax.axvline(cp, color=NET_DUT, ls="--", lw=1.1, alpha=0.45, zorder=1)
    ax.axhline(cv, color=NET_DUT, ls="--", lw=1.1, alpha=0.45, zorder=1)

    # Chỉ gắn tên cho các borough thuộc nhóm Đạt
    for _, t in r[r["good_value"] != "Không đạt"].iterrows():
        ax.annotate(
            f"{t['neighbourhood_cleansed']}",
            (t["ppp_median"], t["value_mean"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color=INK,
            zorder=4,
        )

    ax.set_xlabel(f"Giá trung vị mỗi khách ({tien_te}/khách/đêm)")
    ax.set_ylabel("Điểm value TB (thang 0–5)")
    n_dat = int((r["good_value"] != "Không đạt").sum())
    ax.set_title(f"K3 · {n_dat}/{len(r)} borough vừa rẻ hơn trung vị {tt} ({so(cp, 0)} {tien_te}) "
                 f"vừa có value trên TB ({so(cv, 2)})", fontsize=10, loc="left")
    ax.legend(title="Phân loại (cỡ chấm = số listing)", fontsize=8, title_fontsize=8, loc="lower right")
    ghi_nguon(fig, tt, moc,
              f"Phạm vi: {len(r)}/{len(b)} borough đủ mẫu; giá chỉ dùng mốc cuối {ngay(moc[-1])}; nét đứt mờ = ngưỡng {tt}.\n"
              "'Chắc chắn' = cả KTC 95% vượt ngưỡng; 'Sát ngưỡng' = chỉ điểm ước lượng vượt. Chỉ mô tả, không chứng minh nguyên nhân.",
              y=0.02)
    luu(fig, out, "K3_quadrant")


# ----------------------------------------------------------------------------- K4
def ve_k4(snap, tt, moc, out):
    """K4: 2 khung ngang (1x2): trái = % listing chưa có review (cột từ 0), phải = đường rating TB."""
    n_ls = snap["n_listing"].to_numpy()
    if "n_no_reviews" in snap.columns:
        n_no = snap["n_no_reviews"].to_numpy()
    else:
        n_no = np.round(n_ls * snap["pct_no_reviews"].to_numpy() / 100).astype(int)
    pct_no = n_no / n_ls * 100.0
    rt = snap["rating_mean"].to_numpy()

    x = np.arange(len(snap))
    nhan = [ngay(s) for s in snap["snapshot"]]
    mau_cot = [XAM_SANG] * (len(x) - 1) + [CAM]

    fig, (a_trai, a_phai) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.suptitle(f"K4 · Chất lượng dữ liệu đánh giá qua {len(snap)} mốc, n = {int(n_ls[0]):,} listing cố định"
                 .replace(",", "."), x=0.01, y=1.02, ha="left", fontsize=11.5, fontweight="bold")

    # --- Khung trái: cột % listing chưa có review (trục y từ 0)
    bars = a_trai.bar(x, pct_no, color=mau_cot, width=0.52)
    for idx, bar in enumerate(bars):
        h = bar.get_height()
        a_trai.text(bar.get_x() + bar.get_width() / 2, h + 0.12,
                    f"{so(pct_no[idx], 1)}%\n({int(n_no[idx])} listing)",
                    ha="center", va="bottom", fontsize=8)
    a_trai.set_ylim(0, max(float(pct_no.max()) * 1.25, 1.0))
    a_trai.set_xticks(x)
    a_trai.set_xticklabels(nhan, fontsize=8.5)
    a_trai.set_ylabel("% listing chưa có review (trục từ 0)", fontsize=8.5)
    a_trai.set_title(f"Listing chưa có review giảm từ {so(pct_no[0], 1)}% xuống {so(pct_no[-1], 1)}%",
                     loc="left", fontsize=10)

    # --- Khung phải: đường rating TB
    a_phai.plot(x, rt, marker="o", color=XAM_SANG, lw=2.2, markersize=6)
    a_phai.scatter(x[-1], rt[-1], marker="o", color=CAM, s=68, zorder=4, edgecolor=INK, lw=0.4)
    for i in x:
        a_phai.annotate(so(rt[i], 3), (i, rt[i]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8.5)
    a_phai.set_ylim(min(float(rt.min()) - 0.02, 4.60), max(float(rt.max()) + 0.03, 4.75))
    a_phai.set_xticks(x)
    a_phai.set_xticklabels(nhan, fontsize=8.5)
    a_phai.set_ylabel("Rating TB (thang 0–5; trục cắt)", fontsize=8.5)
    a_phai.set_title(f"Rating TB giảm nhẹ {so(abs(rt[-1] - rt[0]), 3)} điểm ({so(rt[0], 3)} → {so(rt[-1], 3)})",
                     loc="left", fontsize=10)

    ghi_nguon(fig, tt, moc,
              f"Giới hạn: trục rating bị cắt nên độ dốc nhìn lớn hơn thực tế; thay đổi {so(abs(rt[-1] - rt[0]), 3)} điểm là rất nhỏ. "
              f"Đây là cùng {int(n_ls[0]):,} listing ở cả {len(snap)} mốc.".replace(",", "."),
              y=-0.04)
    luu(fig, out, "K4_snapshot")


# ----------------------------------------------------------------------------- K5
def ve_k5(thang, snap, tt, moc, out):
    """K5: heatmap độ sẵn sàng (% ngày còn trống) theo mốc x tháng."""
    d = thang.merge(snap[["snapshot", "n_listing"]], on="snapshot")
    so_ngay = pd.to_datetime(d["date_thang"] + "-01").dt.days_in_month
    d["du"] = d["n_days"] >= 0.95 * d["n_listing"] * so_ngay
    n_loai = int((~d["du"]).sum())
    pv = d[d["du"]].pivot(index="snapshot", columns="date_thang", values="availability_pct")
    pv.index = [f"mốc {ngay(s)}" for s in pv.index]
    pv.columns = [f"{c[5:]}/{c[:4]}" for c in pv.columns]

    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    im = ax.imshow(pv.to_numpy(dtype=float), cmap="YlOrBr", vmin=40, vmax=75, aspect="auto")
    ax.set_xticks(np.arange(pv.shape[1]))
    ax.set_xticklabels(pv.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pv.shape[0]))
    ax.set_yticklabels(pv.index, fontsize=8.5)
    ax.grid(False)

    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            val = pv.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.5, color=INK)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% ngày còn trống", fontsize=8.5)

    dau, cuoi = pv.iloc[0], pv.iloc[-1]
    ax.set_title(f"K5 · Tháng nhiều ngày trống nhất đổi theo mốc: {so(dau.max(), 0)}% (T{dau.idxmax()}) ở mốc đầu, "
                 f"{so(cuoi.max(), 0)}% (T{cuoi.idxmax()}) ở mốc cuối", fontsize=10, loc="left")
    ghi_nguon(fig, tt, moc,
              f"Mỗi ô ≈ 30.000 listing-ngày. Tháng chưa đủ ≥95% dữ liệu đã loại ({n_loai} tháng).\n"
              "Giới hạn: 'ngày trống' gồm cả ngày chủ nhà chặn lịch, không đồng nghĩa chưa có khách. Mỗi mốc phủ 365 ngày khác nhau.",
              y=-0.12)
    luu(fig, out, "K5_availability_month")


# ----------------------------------------------------------------------------- CHẠY CHÍNH
def ve_tat_ca(config_path, proc=None, out=None, root=ROOT):
    root = Path(root)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) if Path(config_path).exists() else {}
    tt = cfg.get("city", "London")
    tien_te = cfg.get("currency", "GBP")
    proc = Path(proc) if proc else root / "data" / "processed"
    out = Path(out) if out else root / "figures"

    b = pd.read_csv(proc / "kpi_borough_value.csv")
    snap = pd.read_csv(proc / "kpi_snapshot.csv")
    thang = pd.read_csv(proc / "kpi_availability_month.csv")
    snap["snapshot"] = snap["snapshot"].astype(str)
    thang["snapshot"] = thang["snapshot"].astype(str)
    moc = sorted(snap["snapshot"])

    ve_k1(b, tt, moc, tien_te, out)
    ve_k2(b, tt, moc, out)
    ve_k3(b, tt, moc, tien_te, out)
    ve_k4(snap, tt, moc, out)
    ve_k5(thang, snap, tt, moc, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vẽ hình K1-K5")
    ap.add_argument("--config", default="configs/London.yaml")
    ap.add_argument("--proc", default=None, help="thư mục chứa kpi_*.csv (mặc định data/processed)")
    ap.add_argument("--out", default=None, help="thư mục lưu hình (mặc định figures)")
    a = ap.parse_args()
    ve_tat_ca(a.config, a.proc, a.out)