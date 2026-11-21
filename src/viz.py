"""viz.py - Vẽ hình K1-K5, K7-K8, K9-K10 từ các file kpi_*.csv (Inside Airbnb).

Đầu vào  : data/processed/kpi_borough_value.csv, kpi_snapshot.csv, kpi_availability_month.csv,
           kpi_complaints_borough.csv (K7-K8; nếu chưa có -> vẽ bản MINH HỌA có watermark),
           kpi_entire_availability.csv & kpi_private_availability.csv (K9-K10)
Đầu ra   : figures/K1_...png ... K5_...png, K7_...png, K8_...png, K9_K10_availability_room_type.png
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
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

# Bảng màu sáng sủa, tương ứng với màu cam điểm nhấn (#E8890C)
XANH_SANG = "#4DB6C6"
XAM_SANG = "#B8C2CC"
XAM_NHAT = "#C5CED6"
CAM = "#E8890C"
XANH_DAT = "#15803D"
NET_DUT = "#7A8691"
INK = "#222222"

NGUONG_NGAY = 0.95

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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


def thang_vn(ym):
    """'2025-10' -> '10/2025'."""
    return f"{ym[5:]}/{ym[:4]}"


def ghi_nguon(fig, thanh_pho, cac_moc, dong_them, y=0.0):
    """Chân trang: nguồn, mốc chụp, phạm vi và giới hạn."""
    nguon = (f"Nguồn: Inside Airbnb (CC BY 4.0), {thanh_pho}, {len(cac_moc)} mốc "
             f"{ngay(cac_moc[0])} – {ngay(cac_moc[-1])}.")
    fig.text(0.01, y, nguon + "\n" + dong_them, fontsize=7.5, color="#555555", va="top", linespacing=1.45)


def luu(fig, thu_muc, ten):
    thu_muc.mkdir(parents=True, exist_ok=True)
    p = thu_muc / f"{ten}.pdf"
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



# ----------------------------------------------------------------------------- K5
def ve_k5(thang, snap, tt, moc, out):
    """K5: heatmap độ sẵn sàng (% ngày còn trống) theo mốc x tháng."""
    d = thang.merge(snap[["snapshot", "n_listing"]], on="snapshot")
    so_ngay = pd.to_datetime(d["date_thang"] + "-01").dt.days_in_month
    d["du"] = d["n_days"] >= NGUONG_NGAY * d["n_listing"] * so_ngay
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
              f"Mỗi ô ≈ 30.000 listing-ngày. Tháng chưa đủ ≥{int(round(NGUONG_NGAY * 100))}% dữ liệu đã loại ({n_loai} tháng).\n"
              "Giới hạn: 'ngày trống' gồm cả ngày chủ nhà chặn lịch, không đồng nghĩa chưa có khách. Mỗi mốc phủ 365 ngày khác nhau.",
              y=-0.12)
    luu(fig, out, "K5_availability_month")


# ----------------------------------------------------------------------------- K7 / K8
def _gia_lap_khieu_nai(ten_borough, seed=42):
    """Số liệu NGẪU NHIÊN chỉ để dựng khung hình khi chưa có kpi_complaints_borough.csv."""
    rng = np.random.default_rng(seed)
    n = rng.integers(12, 90, len(ten_borough))
    noise = np.clip(rng.normal(8, 4, len(ten_borough)), 0.5, 20)
    safety = np.clip(rng.normal(4, 3, len(ten_borough)), 0.2, 15)
    return pd.DataFrame({
        "neighbourhood_cleansed": list(ten_borough), "n_reviews": n,
        "noise_pct": noise.round(1), "safety_pct": safety.round(1), "reliable": True,
    })


def _ve_khieu_nai(c, cot, cot_dem, ma, ten_ngan, mo_ta, tt, moc, out, mo_phong):
    """Thanh ngang sắp giảm dần: % review nhắc tới 1 khía cạnh / số review đã gán nhãn LLM."""
    r = (
        c[c["reliable"]]
        .dropna(subset=[cot])
        .sort_values([cot, "n_reviews"], ascending=[False, False])
        .reset_index(drop=True)
    )
    y = np.arange(len(r))[::-1]
    top3 = (np.arange(len(r)) < 3) & (r[cot] > 0)
    tieu_de = "[MINH HỌA] " if mo_phong else ""

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ax.barh(y, r[cot], height=0.68, color=np.where(top3, CAM, XAM_SANG), zorder=3)
    for yi, v in zip(y, r[cot]):
        ax.text(v + 0.15, yi, f"{so(v, 1)}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{b} (n={int(n)})" for b, n in zip(r["neighbourhood_cleansed"], r["n_reviews"])],
                       fontsize=8.5)
    ax.set_ylim(-0.7, len(r) - 0.3)
    ax.set_xlim(0, max(float(r[cot].max()) * 1.15, 1.0))
    ax.set_xlabel(f"% review {mo_ta} / số review đã gán nhãn LLM (trục từ 0)")

    t = r.iloc[0]
    ax.set_title(f"{ma} · {tieu_de}Tỷ lệ review {mo_ta} theo borough"
                 + ("" if mo_phong else f": cao nhất {t['neighbourhood_cleansed']} ({so(t[cot], 1)}%)"),
                 fontsize=10, loc="left")

    if mo_phong:
        ax.text(0.5, 0.5, "DỮ LIỆU MÔ PHỎNG", transform=ax.transAxes, ha="center", va="center",
                fontsize=38, color="#C0392B", alpha=0.18, rotation=25, fontweight="bold", zorder=5)
        ghi_nguon(fig, tt, moc, "", y=0.02)
        fig.text(0.01, -0.035, "MINH HỌA: số liệu ngẫu nhiên, KHÔNG phải kết quả thật (chưa có kpi_complaints_borough.csv). "
                 "Sẽ thay bằng số thật khi chạy kpi.py.", fontsize=7.5, color="#C0392B", va="top")
    else:
        tong_ca = int(r[cot_dem].sum()) if cot_dem in r.columns else None
        tong_rev = int(r["n_reviews"].sum())
        them_tong = f" (tổng {tong_ca}/{tong_rev} review)" if tong_ca is not None else ""
        ghi_nguon(fig, tt, moc,
                  f"Phạm vi: {len(r)}/{len(c)} borough đủ nhãn LLM{them_tong} (n = số review đã gán nhãn); cam = top 3 > 0%.\n"
                  "Giới hạn: review chỉ gồm khách đã đánh giá; nhãn do LLM gán nên có sai số; tỷ lệ khi n nhỏ (vd. n=10) nhạy mạnh với 1 review.",
                  y=0.02)
    luu(fig, out, f"{ma}_{ten_ngan}")


def ve_k7(c, tt, moc, out, mo_phong=False):
    """K7: % review chê tiếng ồn theo borough."""
    _ve_khieu_nai(c, "noise_pct", "n_noise", "K6", "tieng_on_borough", "chê tiếng ồn", tt, moc, out, mo_phong)


def ve_k8(c, tt, moc, out, mo_phong=False):
    """K8: % review nhắc tới vấn đề an toàn theo borough."""
    _ve_khieu_nai(c, "safety_pct", "n_safety", "K7", "an_toan_borough", "nhắc tới an toàn", tt, moc, out, mo_phong)


# ----------------------------------------------------------------------------- K9 / K10
def doc_k9_k10(proc: Path, snap: pd.DataFrame) -> pd.DataFrame:
    """Đọc K9–K10 trực tiếp từ kpi_entire_availability.csv & kpi_private_availability.csv.

    Dùng tổng n_listing từ kpi_snapshot.csv để lọc tháng đạt >= NGUONG_NGAY (95% số ngày).
    """
    ent = pd.read_csv(proc / "kpi_entire_availability.csv")
    pri = pd.read_csv(proc / "kpi_private_availability.csv")
    n_listing = int(snap["n_listing"].iloc[-1])

    df = ent[["date_thang", "n_days", "availability_pct"]].merge(
        pri[["date_thang", "n_days", "availability_pct"]],
        on="date_thang", how="outer", suffixes=("_entire", "_private"),
    ).sort_values("date_thang").reset_index(drop=True)

    so_ngay = pd.to_datetime(df["date_thang"] + "-01").dt.days_in_month
    df["n_days_total"] = df["n_days_entire"].fillna(0) + df["n_days_private"].fillna(0)
    df["expected_days"] = so_ngay * n_listing
    df["coverage_pct"] = (df["n_days_total"] / df["expected_days"] * 100).round(1)
    df["is_full_month"] = df["n_days_total"] >= (NGUONG_NGAY * df["expected_days"])
    df["entire_avail_pct"] = df["availability_pct_entire"].round(1)
    df["private_avail_pct"] = df["availability_pct_private"].round(1)
    df["k10_diff_pp"] = (df["private_avail_pct"] - df["entire_avail_pct"]).round(1)
    return df.rename(columns={"date_thang": "month_ym"})


def ve_k9_k10(df_k910: pd.DataFrame, tt: str, moc: list[str], out: Path):
    """K9–K10: biểu đồ 2 tầng — trên là độ sẵn sàng 2 loại phòng, dưới là chênh lệch Private − Entire."""
    plot_df = df_k910[df_k910["is_full_month"]].reset_index(drop=True)
    excl_df = df_k910[~df_k910["is_full_month"]].reset_index(drop=True)
    if plot_df.empty:
        print("[VIZ] Không có tháng nào đạt ngưỡng ngày để vẽ K9–K10.")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.14},
    )
    x = np.arange(len(plot_df))
    entire = plot_df["entire_avail_pct"].to_numpy()
    private = plot_df["private_avail_pct"].to_numpy()
    diffs = plot_df["k10_diff_pp"].to_numpy()

    # --- Tầng trên: K9 (hai đường độ sẵn sàng)
    ax1.plot(x, private, color=CAM, marker="o", markersize=5.2, lw=2.2, label="Private room", zorder=3)
    ax1.plot(x, entire, color=XANH_SANG, marker="s", markersize=5.0, lw=2.2, label="Entire home/apt", zorder=3)
    ax1.fill_between(x, entire, private, where=(private >= entire),
                     interpolate=True, color=CAM, alpha=0.14, zorder=2)
    ax1.fill_between(x, entire, private, where=(private < entire),
                     interpolate=True, color=XAM_NHAT, alpha=0.35, zorder=2)

    noi_bat = {
        0,
        len(plot_df) - 1,
        int(np.nanargmax(private)),
        int(np.nanargmin(entire)),
        int(np.nanargmin(private)),
    }
    for idx in sorted(noi_bat):
        pv, ev = private[idx], entire[idx]
        p_va, p_dy = ("bottom", 1.8) if pv >= ev else ("top", -2.2)
        e_va, e_dy = ("top", -2.2) if pv >= ev else ("bottom", 1.8)
        ax1.text(idx, pv + p_dy, f"{so(pv, 1)}%", ha="center", va=p_va, fontsize=8, color=INK, fontweight="bold")
        ax1.text(idx, ev + e_dy, f"{so(ev, 1)}%", ha="center", va=e_va, fontsize=8, color=INK)

    y_min = max(0, float(min(np.nanmin(entire), np.nanmin(private))) - 10)
    y_max = min(100, float(max(np.nanmax(entire), np.nanmax(private))) + 10)
    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel("% ngày còn trống", fontsize=8.5)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#E6E9EB")

    n_duong = int((diffs > 0).sum())
    i_max = int(np.nanargmax(diffs))
    thang_max = thang_vn(plot_df["month_ym"].iloc[i_max])
    if (diffs < 0).any():
        thang_dao = thang_vn(plot_df.loc[diffs < 0, "month_ym"].iloc[0])
        tieu_de = (f"K9 · Private còn trống hơn Entire {n_duong}/{len(plot_df)} tháng, "
                   f"chênh lớn nhất +{so(diffs[i_max], 1)} điểm ở T{thang_max}, đảo chiều từ T{thang_dao}")
    else:
        tieu_de = (f"K9 · Private còn trống hơn Entire {n_duong}/{len(plot_df)} tháng, "
                   f"chênh lớn nhất +{so(diffs[i_max], 1)} điểm ở T{thang_max}")
    ax1.set_title(tieu_de, fontsize=10, loc="left")

    # --- Tầng dưới: K10 (cột chênh lệch Private − Entire)
    mau_cot = [CAM if v >= 0 else XAM_SANG for v in diffs]
    ax2.bar(x, diffs, color=mau_cot, width=0.62, zorder=3)
    ax2.axhline(0, color=NET_DUT, lw=1.1, zorder=4)
    for xi, val in zip(x, diffs):
        if pd.isna(val):
            continue
        va = "bottom" if val >= 0 else "top"
        dy = 0.55 if val >= 0 else -0.65
        nhan_val = f"+{so(val, 1)}" if val > 0 else (f"−{so(abs(val), 1)}" if val < 0 else so(0.0, 1))
        ax2.text(xi, val + dy, nhan_val, ha="center", va=va, fontsize=7.8, color=INK, fontweight="bold")

    d_min, d_max = float(np.nanmin(diffs)), float(np.nanmax(diffs))
    ax2.set_ylim(min(d_min - 3.2, -6.5), max(d_max + 3.5, 8.0))
    ax2.set_ylabel("Chênh Private − Entire\n(điểm %)", fontsize=8.5)
    ax2.set_xlabel("Tháng dương lịch (calendar gộp theo mốc chụp muộn nhất)", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([thang_vn(m) for m in plot_df["month_ym"]], rotation=45, ha="right", fontsize=8)
    ax2.legend(
        handles=[
            Patch(facecolor=CAM, label="Private còn trống hơn Entire (> 0)"),
            Patch(facecolor=XAM_SANG, label="Private thấp hơn Entire (< 0)"),
        ],
        loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="#E6E9EB",
    )

    nguong_pct = int(round(NGUONG_NGAY * 100))
    if not excl_df.empty:
        ds_loai = ", ".join(f"{thang_vn(r['month_ym'])} ({so(r['coverage_pct'], 1)}% số ngày)" for _, r in excl_df.iterrows())
        dong_loai = f"Tháng chưa đủ ≥{nguong_pct}% số ngày đã loại ({len(excl_df)} tháng): {ds_loai}."
    else:
        dong_loai = f"Tất cả các tháng đều đạt ≥{nguong_pct}% số ngày."

    ghi_nguon(
        fig, tt, moc,
        f"Phạm vi: {len(plot_df)} tháng đủ ngày ({thang_vn(plot_df['month_ym'].iloc[0])} – {thang_vn(plot_df['month_ym'].iloc[-1])}); "
        f"đọc từ kpi_entire_availability.csv & kpi_private_availability.csv. {dong_loai}\n"
        "Giới hạn: 'ngày trống' gồm cả ngày chủ nhà tự khóa lịch; loại phòng lấy theo mốc cuối nên không phản ánh listing đổi loại (nếu có).",
        y=-0.03,
    )
    luu(fig, out, "K9_availability_room_type")

def ve_gia_trung_vi_loai_phong(df_rt: pd.DataFrame, tt: str, moc: list[str], tien_te: str, out: Path):
    """Vẽ 2 biểu đồ cột ngang riêng biệt cho Entire home/apt và Private room so với trung vị toàn thành phố."""
    cau_hinh_loai = [
        {
            "cot_gia": "trung_vi_en",
            "cot_city": "city_median_en",
            "ten_loai": "Entire home/apt",
            "ten_file": "K8_gia_trung_vi_entire_home",
        },
        {
            "cot_gia": "trung_vi_pri",
            "cot_city": "city_median_pri",
            "ten_loai": "Private room",
            "ten_file": "K8_gia_trung_vi_private_room",
        },
    ]

    for cfg in cau_hinh_loai:
        cot_gia = cfg["cot_gia"]
        cot_city = cfg["cot_city"]
        ten_loai = cfg["ten_loai"]

        # Loại bỏ các quận không có dữ liệu loại phòng này và sắp xếp tăng dần
        r = df_rt.dropna(subset=[cot_gia]).sort_values(cot_gia, ascending=True).reset_index(drop=True)
        if r.empty:
            continue

        # Lấy ngưỡng giá trung vị toàn thành phố
        city_med = float(r[cot_city].iloc[0]) if cot_city in r.columns else float(r[cot_gia].median())
        y = np.arange(len(r))

        # Tô màu cam cho các quận có giá vượt ngưỡng trung vị toàn thành phố, màu xanh sáng cho các quận thấp hơn/bằng
        mau_cot = [CAM if v > city_med else XANH_SANG for v in r[cot_gia]]

        fig, ax = plt.subplots(figsize=(8.4, 7.2))
        bars = ax.barh(y, r[cot_gia], height=0.68, color=mau_cot, edgecolor=INK, lw=0.3, zorder=3)

        # Đường kẻ nét đứt thẳng đứng biểu diễn giá trung vị toàn thành phố
        ax.axvline(city_med, color=INK, ls="--", lw=1.4, alpha=0.85, zorder=4)

        # Ghi nhãn giá trị ở cuối mỗi thanh ngang
        max_val = float(r[cot_gia].max())
        for yi, v in zip(y, r[cot_gia]):
            ax.text(v + max_val * 0.012, yi, f"{so(v, 0)}", va="center", fontsize=7.8, color=INK)

        # Ghi chú thích trực tiếp cho đường ngưỡng thành phố ở phía trên cùng biểu đồ
        ax.text(
            city_med + max_val * 0.015,
            len(r) - 0.35,
            f"Trung vị {tt}: {so(city_med, 0)} {tien_te}",
            color=INK,
            fontsize=8.5,
            fontweight="bold",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=NET_DUT, alpha=0.9),
            zorder=5,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(r["neighbour_hood"], fontsize=8.5)
        ax.set_ylim(-0.7, len(r) + 0.3)
        ax.set_xlim(0, max_val * 1.15)
        ax.set_xlabel(f"Giá trung vị ({tien_te}/đêm) — Loại phòng: {ten_loai}", fontsize=9)

        so_quan_vuot = int((r[cot_gia] > city_med).sum())
        ax.set_title(
            f"Giá trung vị {ten_loai} theo quận: {so_quan_vuot}/{len(r)} quận cao hơn ngưỡng {tt} ({so(city_med, 0)} {tien_te})",
            fontsize=10,
            loc="left",
        )

        ax.legend(
            handles=[
                Patch(facecolor=CAM, edgecolor=INK, lw=0.3, label=f"Cao hơn trung vị {tt} (> {so(city_med, 0)} {tien_te})"),
                Patch(facecolor=XANH_SANG, edgecolor=INK, lw=0.3, label=f"Thấp hơn hoặc bằng trung vị {tt}"),
            ],
            loc="lower right",
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="#E6E9EB",
        )

        ghi_nguon(
            fig,
            tt,
            moc,
            f"Phạm vi: {len(r)} quận có dữ liệu {ten_loai} ở mốc cuối {ngay(moc[-1])}; đường nét đứt = giá trung vị toàn {tt}.\n"
            "Giới hạn: giá trung vị chỉ tính trên các listing hợp lệ sau QA ở mốc cuối, chưa phản ánh biến động theo mùa.",
            y=0.02,
        )
        luu(fig, out, cfg["ten_file"])

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
    ve_k5(thang, snap, tt, moc, out)

    # K7-K8: dùng kết quả LLM nếu đã có; nếu chưa có thì vẽ bản minh họa có watermark
    f_kn = proc / "kpi_complaints_borough.csv"
    if f_kn.exists():
        c, mo_phong = pd.read_csv(f_kn), False
    else:
        print("[VIZ] chưa có kpi_complaints_borough.csv -> K7/K8 vẽ bản MINH HỌA (số ngẫu nhiên)")
        c, mo_phong = _gia_lap_khieu_nai(b["neighbourhood_cleansed"].tolist()), True
    ve_k7(c, tt, moc, out, mo_phong)
    ve_k8(c, tt, moc, out, mo_phong)

    # K9-K10: đọc trực tiếp từ 2 file KPI độ sẵn sàng theo loại phòng
    if (proc / "kpi_entire_availability.csv").exists() and (proc / "kpi_private_availability.csv").exists():
        df_k910 = doc_k9_k10(proc, snap)
        ve_k9_k10(df_k910, tt, moc, out)
    else:
        print("[VIZ] chưa có kpi_entire_availability.csv / kpi_private_availability.csv -> bỏ qua K9-K10")
        
    f_rt_med = proc / "kpi_room_type_median.csv"
    if f_rt_med.exists():
        df_rt = pd.read_csv(f_rt_med)
        ve_gia_trung_vi_loai_phong(df_rt, tt, moc, tien_te, out)
    else:
        print("[VIZ] chưa có kpi_room_type_median.csv -> bỏ qua biểu đồ giá trung vị từng loại phòng")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vẽ hình K1-K5, K7-K8, K9-K10")
    ap.add_argument("--config", default="configs/London.yaml")
    ap.add_argument("--proc", default=None, help="thư mục chứa kpi_*.csv (mặc định data/processed)")
    ap.add_argument("--out", default=None, help="thư mục lưu hình (mặc định figures)")
    a = ap.parse_args()
    ve_tat_ca(a.config, a.proc, a.out)
