"""
Maps for the "sea memory" report (Arabic labels):
  1. maps_study_area.png   - the study area, the three sectors, Dakhla, mean SST 2002-2026
  2. maps_patterns.png     - the 6 sea patterns: mean SST map of each, with its peak months
  3. maps_today_vs_analog.png - the latest day vs its most similar past day (temperature, food, currents)
Reuses the caches written by front_regimes.py and sea_memory_v2.py.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import front_regimes as fr  # noqa: E402
import sea_memory_v2 as sm  # noqa: E402

OUT = fr.OUT
MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "ماي", "يونيو", "يوليوز", "غشت", "شتنبر", "أكتوبر", "نونبر", "دجنبر"]
DAKHLA = (23.70, -15.93)


def ar(s):
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    return get_display(reshape(s))


def load():
    years = range(2002, dt.date.today().year + 1)
    all_days, maps = [], []
    for y in years:
        try:
            d, lats, lons, v = fr.mur_year(y)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] MUR {y}: {e}")
            continue
        all_days += d
        maps.append(v)
    sst = np.concatenate(maps)
    days = [dt.date.fromisoformat(x) for x in all_days]
    chl = np.full(sst.shape, np.nan, "float32")
    u = np.full(sst.shape, np.nan, "float32")
    v = np.full(sst.shape, np.nan, "float32")
    for y in years:
        idx = [i for i, d in enumerate(days) if d.year == y]
        if not idx:
            continue
        dy = [days[i] for i in idx]
        c = sm.cm_year("chl", sm.CHL_IDS, ["CHL"], y)
        if c is not None:
            chl[idx] = sm.on_grid(c, "CHL", dy, lats, lons)
        cu = sm.cm_year("cur", sm.CUR_IDS, ["uo", "vo"], y, depth=True)
        if cu is not None:
            u[idx] = sm.on_grid(cu, "uo", dy, lats, lons)
            v[idx] = sm.on_grid(cu, "vo", dy, lats, lons)
    return days, lats, lons, sst, chl, u, v


def base_ax(ax, lats, lons, land):
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.set_aspect(1 / np.cos(np.radians(23.5)))
    ax.contourf(lons, lats, land.astype(float), levels=[0.5, 1.5], colors=["#D3D1C7"])
    ax.plot(DAKHLA[1], DAKHLA[0], "o", color="#2C2C2A", ms=4)
    ax.text(DAKHLA[1] + 0.08, DAKHLA[0], ar("الداخلة"), fontsize=9, va="center", color="#2C2C2A")
    ax.tick_params(labelsize=7)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.cluster.vq import kmeans2

    for f in font_manager.findSystemFonts():
        if "NotoSansArabic" in f or "NotoNaskhArabic" in f:
            font_manager.fontManager.addfont(f)
    names = {f.name for f in font_manager.fontManager.ttflist}
    fam = next((n for n in ("Noto Sans Arabic", "Noto Naskh Arabic") if n in names), "DejaVu Sans")
    print("font", fam)
    plt.rcParams["font.family"] = [fam, "DejaVu Sans"]

    days, lats, lons, sst, chl, u, v = load()
    land = np.isfinite(sst).mean(0) < 0.05
    oc = ~land

    # ---- 1. study area
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    m = np.nanmean(sst, 0)
    pc = ax.pcolormesh(lons, lats, np.where(oc, m, np.nan), cmap="RdYlBu_r", shading="auto")
    base_ax(ax, lats, lons, land)
    for la, name in ((22.5, "جنوب الداخلة"), (23.5, "الداخلة"), (24.5, "شمال الداخلة")):
        ax.text(lons.min() + 0.15, la, ar(name), fontsize=10, color="#2C2C2A",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888780", lw=0.5))
    for la in (23.0, 24.0):
        ax.axhline(la, color="#444441", lw=0.8, ls="--")
    cb = fig.colorbar(pc, ax=ax, shrink=0.8)
    cb.set_label(ar("متوسط حرارة سطح البحر °C"), fontsize=9)
    ax.set_title(ar("منطقة الدراسة: الماء البارد قرب الساحل والدافئ في عرض البحر (متوسط 2002–2026)"), fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "maps_study_area.png", dpi=150)
    plt.close(fig)

    # ---- 2. the 6 patterns (same method as front_regimes.py)
    X = sst[:, oc]
    good = np.isfinite(X).mean(1) > 0.98
    Xg = np.where(np.isfinite(X[good]), X[good], np.nanmean(X[good], 1, keepdims=True))
    Xa = Xg - Xg.mean(1, keepdims=True)
    mu = Xa.mean(0)
    _, _, Vt = np.linalg.svd(Xa - mu, full_matrices=False)
    Z = (Xa - mu) @ Vt[:12].T
    _, lab = kmeans2(Z[:, :6], 6, minit="++", seed=0)
    gdays = [d for d, g in zip(days, good) if g]
    sst_g = sst[good]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6), constrained_layout=True)
    vmin, vmax = np.nanpercentile(sst[:, oc], [3, 97])
    for c in range(6):
        ax = axs.flat[c]
        sel = lab == c
        mm = np.nanmean(sst_g[sel], 0)
        months = np.bincount([d.month for d, s in zip(gdays, sel) if s], minlength=13)[1:]
        peak = [MONTHS[i] for i in np.argsort(months)[::-1][:3]]
        pc = ax.pcolormesh(lons, lats, np.where(oc, mm, np.nan), cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
        base_ax(ax, lats, lons, land)
        ax.set_title(ar(f"الشكل {c + 1}: غالباً في {'، '.join(peak)}"), fontsize=10.5)
    fig.colorbar(pc, ax=axs, shrink=0.7, label=ar("حرارة سطح البحر °C"))
    fig.suptitle(ar("أشكال البحر الستة أمام الداخلة: متوسط حرارة الماء في أيام كل شكل"), fontsize=13)
    fig.savefig(OUT / "maps_patterns.png", dpi=140)
    plt.close(fig)

    # ---- 3. latest day vs its most similar past day (temperature + food + currents)
    def frac(cube):
        return np.array([np.isfinite(cube[i][oc]).mean() for i in range(len(days))])
    ok3 = (frac(sst) > 0.98) & (frac(chl) > 0.6) & (frac(u) > 0.6)
    Zs, _ = sm.block(sst, ok3)
    Zc, _ = sm.block(chl, ok3, log=True)
    Zu, _ = sm.block(u, ok3, demean=False)
    Zv, _ = sm.block(v, ok3, demean=False)
    Zall = np.hstack([Zs, Zc, Zu * np.sqrt(0.5), Zv * np.sqrt(0.5)])
    for y in sorted({d.year for d in days}):
        m = np.array([d.year == y for d in days])
        print(y, "sst", round(float((frac(sst)[m] > 0.98).mean()), 2), "chl", round(float(frac(chl)[m].mean()), 2),
              "cur", round(float(frac(u)[m].mean()), 2), "all3", int(ok3[m].sum()))
    idx = np.nonzero(ok3)[0]
    t = idx[-1]
    yrs = np.array([d.year for d in days])
    doy = np.array([d.timetuple().tm_yday for d in days])
    cal = np.abs(((doy[idx] - doy[t] + 182) % 365) - 182)
    pool = idx[(yrs[idx] != yrs[t]) & (cal <= 45)]
    dist = np.sqrt(((Zall[pool] - Zall[t]) ** 2).sum(1))
    a = pool[int(np.argmin(dist))]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6))
    for row, (i, name) in enumerate(((t, "اليوم المختار"), (a, "أشبه يوم في الماضي"))):
        ax = axs[row, 0]
        an = sst[i] - np.nanmean(sst[i][oc])
        ax.pcolormesh(lons, lats, np.where(oc, an, np.nan), cmap="RdBu_r", vmin=-2.5, vmax=2.5, shading="auto")
        base_ax(ax, lats, lons, land)
        ax.set_title(ar(f"{name} ({days[i]}): شكل الحرارة (أبرد/أدفأ من المتوسط)"), fontsize=10)
        ax = axs[row, 1]
        ax.pcolormesh(lons, lats, np.where(oc, np.log10(np.clip(chl[i], 0.05, 30)), np.nan), cmap="viridis", vmin=-1, vmax=1.2, shading="auto")
        base_ax(ax, lats, lons, land)
        ax.set_title(ar("الغذاء (الكلوروفيل)"), fontsize=10)
        ax = axs[row, 2]
        sp = np.hypot(u[i], v[i])
        ax.pcolormesh(lons, lats, np.where(oc, sp, np.nan), cmap="Blues", vmin=0, vmax=0.5, shading="auto")
        s = 4
        ax.quiver(lons[::s], lats[::s], u[i][::s, ::s], v[i][::s, ::s], color="#2C2C2A", scale=6, width=0.003)
        base_ax(ax, lats, lons, land)
        ax.set_title(ar("التيارات (الاتجاه والسرعة)"), fontsize=10)
    fig.suptitle(ar(f"«ذاكرة البحر»: بحر {days[t]} يشبه أكثر بحر {days[a]} (بالحرارة والغذاء والتيارات معاً)"), fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "maps_today_vs_analog.png", dpi=130)
    plt.close(fig)
    (OUT / "maps_info.txt").write_text(f"today={days[t]} analog={days[a]}\n", encoding="utf-8")
    print("today", days[t], "analog", days[a])


if __name__ == "__main__":
    main()
