"""
Text-free map panels for the "sea memory" report (Arabic captions are added outside matplotlib,
because matplotlib cannot shape Arabic script). Writes docs/analysis/fronts/panels/*.png + panels.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sea_memory_maps as smm  # noqa: E402
import sea_memory_v2 as sm  # noqa: E402

OUT = smm.OUT / "panels"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.vq import kmeans2

    OUT.mkdir(parents=True, exist_ok=True)
    days, lats, lons, sst, chl, u, v = smm.load()
    land = np.isfinite(sst).mean(0) < 0.05
    oc = ~land
    meta = {}

    def panel(name, field, cmap, vmin, vmax, quiv=None, sectors=False, size=(4.4, 3.4)):
        fig, ax = plt.subplots(figsize=size)
        ax.pcolormesh(lons, lats, np.where(oc, field, np.nan), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlim(lons.min(), lons.max())
        ax.set_ylim(lats.min(), lats.max())
        ax.set_aspect(1 / np.cos(np.radians(23.5)))
        ax.contourf(lons, lats, land.astype(float), levels=[0.5, 1.5], colors=["#D3D1C7"])
        if quiv is not None:
            s = 4
            ax.quiver(lons[::s], lats[::s], quiv[0][::s, ::s], quiv[1][::s, ::s], color="#2C2C2A", scale=6, width=0.004)
        if sectors:
            for la in (23.0, 24.0):
                ax.axhline(la, color="#444441", lw=0.9, ls="--")
        ax.plot(smm.DAKHLA[1], smm.DAKHLA[0], "o", color="#111111", ms=5, mec="white", mew=1)
        ax.tick_params(labelsize=7)
        ax.set_xticks([-19, -18, -17, -16, -15])
        ax.set_xticklabels(["19W", "18W", "17W", "16W", "15W"])
        ax.set_yticks([22, 23, 24, 25])
        ax.set_yticklabels(["22N", "23N", "24N", "25N"])
        fig.tight_layout(pad=0.3)
        fig.savefig(OUT / f"{name}.png", dpi=160)
        plt.close(fig)

    def colorbar(name, cmap, vmin, vmax, ticks, unit):
        fig, ax = plt.subplots(figsize=(4.4, 0.55))
        import matplotlib as mpl
        cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(cmap), norm=mpl.colors.Normalize(vmin, vmax),
                                       orientation="horizontal", ticks=ticks)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(unit, fontsize=8)
        fig.tight_layout(pad=0.2)
        fig.savefig(OUT / f"cb_{name}.png", dpi=160)
        plt.close(fig)

    # 1. study area
    m = np.nanmean(sst, 0)
    lo, hi = float(np.nanpercentile(m[oc], 1)), float(np.nanpercentile(m[oc], 99))
    panel("study_area", m, "RdYlBu_r", lo, hi, sectors=True, size=(6.4, 4.8))
    colorbar("study", "RdYlBu_r", lo, hi, list(np.round(np.linspace(lo, hi, 5), 1)), "degC")

    # 2. six patterns
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
    vmin, vmax = [float(x) for x in np.nanpercentile(sst[:, oc], [3, 97])]
    pats = []
    for c in range(6):
        sel = lab == c
        months = np.bincount([d.month for d, s in zip(gdays, sel) if s], minlength=13)[1:]
        panel(f"pattern_{c + 1}", np.nanmean(sst_g[sel], 0), "RdYlBu_r", vmin, vmax)
        pats.append({"id": c + 1, "days": int(sel.sum()), "peak_months": [int(i) + 1 for i in np.argsort(months)[::-1][:3]],
                     "years": len({d.year for d, s in zip(gdays, sel) if s})})
    colorbar("patterns", "RdYlBu_r", vmin, vmax, list(np.round(np.linspace(vmin, vmax, 6), 1)), "degC")
    meta["patterns"] = pats

    # 3. example: latest day with all three vs most similar day, same season, other years
    def frac(cube):
        return np.array([np.isfinite(cube[i][oc]).mean() for i in range(len(days))])
    ok3 = (frac(sst) > 0.98) & (frac(chl) > 0.6) & (frac(u) > 0.6)
    Zs, _ = sm.block(sst, ok3)
    Zc, _ = sm.block(chl, ok3, log=True)
    Zu, _ = sm.block(u, ok3, demean=False)
    Zv, _ = sm.block(v, ok3, demean=False)
    Zall = np.hstack([Zs, Zc, Zu * np.sqrt(0.5), Zv * np.sqrt(0.5)])
    idx = np.nonzero(ok3)[0]
    t = idx[-1]
    yrs = np.array([d.year for d in days])
    doy = np.array([d.timetuple().tm_yday for d in days])
    cal = np.abs(((doy[idx] - doy[t] + 182) % 365) - 182)
    pool = idx[(yrs[idx] != yrs[t]) & (cal <= 45)]
    dist = np.sqrt(((Zall[pool] - Zall[t]) ** 2).sum(1))
    order = np.argsort(dist)
    a = pool[order[0]]
    for tag, i in (("today", t), ("analog", a)):
        panel(f"ex_{tag}_sst", sst[i] - np.nanmean(sst[i][oc]), "RdBu_r", -2.5, 2.5)
        panel(f"ex_{tag}_chl", np.log10(np.clip(chl[i], 0.05, 30)), "viridis", -1, 1.2)
        panel(f"ex_{tag}_cur", np.hypot(u[i], v[i]), "Blues", 0, 0.5, quiv=(u[i], v[i]))
    colorbar("ex_sst", "RdBu_r", -2.5, 2.5, [-2, -1, 0, 1, 2], "degC vs map mean")
    colorbar("ex_chl", "viridis", -1, 1.2, [-1, -0.5, 0, 0.5, 1], "log10 chlorophyll (mg/m3)")
    colorbar("ex_cur", "Blues", 0, 0.5, [0, 0.1, 0.2, 0.3, 0.4, 0.5], "current speed (m/s)")
    meta["example"] = {"today": str(days[t]), "analog": str(days[a]), "pool_days": int(len(pool))}
    years_all3 = sorted({days[i].year for i in idx})
    meta["years_with_all_three"] = [years_all3[0], years_all3[-1]]
    (OUT / "panels.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
