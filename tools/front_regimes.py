"""
Do Dakhla's sea "states" repeat across years, and does the date matter?

Archive analysis for the "sea memory" (analog days) idea, satellite + reanalysis only (no catch data):

  1. SST every 2nd day since 2002 (NASA MUR, ~5 km sampling) over 22-25N, 19.5-14.6W.
  2. Cross-shore profiles at every latitude row: width of the cold upwelled band
     (distance offshore where SST first exceeds the nearshore SST + 1 degC) and the
     upwelling contrast dT = offshore (150-200 km) minus nearshore (5-15 km).
  3. Winds (ERA5 via Open-Meteo archive) -> Coastal Upwelling Index, mean of the 7 days before.
  4. Relationship wind -> cold band width / dT, by wind class.
  5. Analog test: for each day, the most similar SST *pattern* among other years, versus the
     same calendar date in other years. If the best analog is usually far from the same date,
     "the date does not matter, the pattern does".

Outputs: docs/analysis/fronts/summary.json and PNG charts.
"""
import datetime as dt
import json
import math
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "analysis" / "fronts"
CACHE = ROOT / "data" / "cache" / "mur_fronts"
UA = {"User-Agent": "Mozilla/5.0 (dakhla-radar; +https://github.com/Negjir/dakhla-radar)"}
BASES = ["https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41",
         "https://upwell.pfeg.noaa.gov/erddap/griddap/jplMURSST41"]
LAT0, LAT1, LON0, LON1, SP, TSTEP = 22.0, 25.0, -19.5, -14.6, 5, 2
WIND_PTS = [(22.5, -16.95, 205, "south"), (23.5, -16.30, 215, "dakhla"), (24.3, -15.80, 220, "north")]


def get(url, timeout=600, tries=4):
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(10 * (k + 1))
    raise RuntimeError(err)


def mur_year(year):
    import xarray as xr

    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{year}.nc"
    if not f.exists():
        a = dt.date(year, 1, 1) if year > 2002 else dt.date(2002, 6, 2)
        b = min(dt.date(year, 12, 31), dt.date.today() - dt.timedelta(days=3))
        q = (f".nc?analysed_sst[({a}T09:00:00Z):{TSTEP}:({b}T09:00:00Z)]"
             f"[({LAT0}):{SP}:({LAT1})][({LON0}):{SP}:({LON1})]")
        data, err = None, None
        for base in BASES:
            try:
                data = get(base + q)
                break
            except Exception as e:  # noqa: BLE001
                err = e
                print(f"[skip] {year} {base.split('/')[2]}: {e}")
        if data is None:
            raise RuntimeError(err)
        f.write_bytes(data)
    ds = xr.open_dataset(f)
    v = ds["analysed_sst"].values.astype("float32")
    if np.nanmean(v) > 200:
        v -= 273.15
    days = [str(t)[:10] for t in ds["time"].values]
    return days, ds["latitude"].values, ds["longitude"].values, v


def winds(start, end):
    """Daily Coastal Upwelling Index per point from ERA5 hourly 10 m winds (Open-Meteo archive)."""
    from forecast import coastal_upwelling_index

    out = {}
    for la, lo, cdir, name in WIND_PTS:
        per = {}
        y = start.year
        while y <= end.year:
            a, b = max(start, dt.date(y, 1, 1)), min(end, dt.date(y + 4, 12, 31))
            url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}&start_date={a}&end_date={b}"
                   "&hourly=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms&timezone=UTC")
            h = json.loads(get(url))["hourly"]
            for t, s, w in zip(h["time"], h["wind_speed_10m"], h["wind_direction_10m"]):
                if s is not None and w is not None:
                    per.setdefault(t[:10], []).append(coastal_upwelling_index(s, w, cdir, la))
            print(f"[ok] winds {name} {a}..{b}")
            y += 5
            time.sleep(2)
        out[name] = {k: float(np.mean(v)) for k, v in per.items()}
    return out


def profiles(lats, lons, sst):
    """Per day and latitude row: cold-band width (km) and upwelling contrast dT (degC)."""
    nd, ny, nx = sst.shape
    width = np.full((nd, ny), np.nan, "float32")
    dT = np.full((nd, ny), np.nan, "float32")
    for j in range(ny):
        for d in range(nd):
            row = sst[d, j]
            ok = np.isfinite(row)
            if ok.sum() < 20:
                continue
            land = np.nonzero(~ok)[0]
            if not len(land) or land[0] == 0:
                continue
            ic = land[0] - 1                                  # last ocean cell before the first land (scanning from the west)
            dist = (lons[ic] - lons[: ic + 1]) * 111.32 * math.cos(math.radians(lats[j]))
            prof = row[: ic + 1]
            near = prof[(dist >= 5) & (dist <= 15)]
            off = prof[(dist >= 150) & (dist <= 200)]
            if not len(near) or not len(off) or not np.isfinite(near).any() or not np.isfinite(off).any():
                continue
            tn, to = np.nanmean(near), np.nanmean(off)
            dT[d, j] = to - tn
            order = np.argsort(dist)
            ds_, ps_ = dist[order], prof[order]
            hit = np.nonzero((ds_ > 5) & (ps_ > tn + 1.0))[0]
            width[d, j] = ds_[hit[0]] if len(hit) else np.nan
    return width, dT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    years = range(2002, dt.date.today().year + 1)
    all_days, maps = [], []
    for y in years:
        try:
            d, lats, lons, v = mur_year(y)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] MUR {y}: {e}")
            continue
        all_days += d
        maps.append(v)
        print(f"[ok] MUR {y}: {len(d)} days")
    sst = np.concatenate(maps)
    days = [dt.date.fromisoformat(x) for x in all_days]
    print("SST cube", sst.shape)

    width, dT = profiles(lats, lons, sst)
    sector = {"south": (lats >= 22.0) & (lats < 23.0), "dakhla": (lats >= 23.0) & (lats < 24.0), "north": (lats >= 24.0) & (lats <= 25.0)}

    W = winds(days[0], days[-1])
    def cui7(name, day):
        v = [W[name].get(str(day - dt.timedelta(days=k))) for k in range(1, 8)]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else np.nan

    rows = []
    for i, day in enumerate(days):
        r = {"date": str(day), "month": day.month}
        for s, m in sector.items():
            r[f"w_{s}"] = float(np.nanmedian(width[i, m])) if np.isfinite(width[i, m]).any() else None
            r[f"dT_{s}"] = float(np.nanmedian(dT[i, m])) if np.isfinite(dT[i, m]).any() else None
            r[f"cui_{s}"] = cui7(s, day)
        rows.append(r)

    # ---- wind -> front relationship, by wind class
    rel = {}
    for s in sector:
        x = np.array([r[f"cui_{s}"] for r in rows], float)
        yw = np.array([np.nan if r[f"w_{s}"] is None else r[f"w_{s}"] for r in rows], float)
        yd = np.array([np.nan if r[f"dT_{s}"] is None else r[f"dT_{s}"] for r in rows], float)
        ok = np.isfinite(x) & np.isfinite(yd)
        okw = np.isfinite(x) & np.isfinite(yw)
        q = np.nanpercentile(x[ok], [33, 67])
        cls = {}
        for name, lo, hi in [("weak", -1e9, q[0]), ("moderate", q[0], q[1]), ("strong", q[1], 1e9)]:
            m = ok & (x >= lo) & (x < hi)
            mw = okw & (x >= lo) & (x < hi)
            cls[name] = {"cui_range": [round(float(max(lo, np.nanmin(x[ok]))), 0), round(float(min(hi, np.nanmax(x[ok]))), 0)],
                         "days": int(m.sum()),
                         "dT_median": round(float(np.nanmedian(yd[m])), 2),
                         "cold_band_km_median": round(float(np.nanmedian(yw[mw])), 1) if mw.any() else None,
                         "cold_band_km_iqr": [round(float(v), 1) for v in np.nanpercentile(yw[mw], [25, 75])] if mw.any() else None}
        rel[s] = {"corr_cui_dT": round(float(np.corrcoef(x[ok], yd[ok])[0, 1]), 2),
                  "corr_cui_width": round(float(np.corrcoef(x[okw], yw[okw])[0, 1]), 2) if okw.sum() > 30 else None,
                  "classes": cls}

    # ---- analog test on SST *patterns* (each map minus its own mean), PCA space
    ocean = np.isfinite(sst).mean(0) > 0.95
    X = sst[:, ocean]
    good = np.isfinite(X).mean(1) > 0.98
    X = X[good]
    X = np.where(np.isfinite(X), X, np.nanmean(X, 1, keepdims=True))
    X = X - X.mean(1, keepdims=True)
    gdays = [d for d, g in zip(days, good) if g]
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = 12
    Z = (X - mu) @ Vt[:k].T
    expl = float((S[:k] ** 2).sum() / (S ** 2).sum())
    yrs = np.array([d.year for d in gdays])
    doy = np.array([d.timetuple().tm_yday for d in gdays])
    offs, ratio, best_same = [], [], 0
    rng = np.random.default_rng(1)
    sample = rng.choice(len(gdays), size=min(1500, len(gdays)), replace=False)
    for i in sample:
        other = yrs != yrs[i]
        dist = np.sqrt(((Z - Z[i]) ** 2).sum(1))
        dist[~other] = np.inf
        j = int(np.argmin(dist))
        cal = np.abs(((doy - doy[i] + 182) % 365) - 182)
        same = other & (cal <= 3)
        if not same.any():
            continue
        d_same = float(np.median(dist[same]))
        offs.append(int(cal[j]))
        ratio.append(float(dist[j] / d_same))
        best_same += int(cal[j] <= 15)
    offs = np.array(offs)
    analog = {"days_tested": int(len(offs)), "pca_components": k, "variance_explained": round(expl, 2),
              "best_analog_calendar_offset_days_median": int(np.median(offs)),
              "share_best_analog_within_15_days_of_date": round(best_same / len(offs), 2),
              "share_best_analog_more_than_30_days_from_date": round(float((offs > 30).mean()), 2),
              "share_best_analog_more_than_60_days_from_date": round(float((offs > 60).mean()), 2),
              "best_analog_distance_vs_same_date_median": round(float(np.median(ratio)), 2)}

    # ---- pattern families (k-means on PCA) and when they occur
    from scipy.cluster.vq import kmeans2
    np.random.seed(0)
    cent, lab = kmeans2(Z[:, :6], 6, minit="++", seed=0)
    fam = []
    for c in range(6):
        m = lab == c
        months = np.bincount([d.month for d, mm in zip(gdays, m) if mm], minlength=13)[1:]
        years_c = sorted({d.year for d, mm in zip(gdays, m) if mm})
        fam.append({"family": c + 1, "days": int(m.sum()), "months": [int(v) for v in months], "years": len(years_c)})

    summary = {"made": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "period": [str(days[0]), str(days[-1])],
               "sst": "NASA JPL MUR SST v4.1 (ERDDAP), every 2nd day, ~5 km sampling, 22-25N 19.5-14.6W",
               "wind": "ERA5 10 m wind via Open-Meteo archive -> Coastal Upwelling Index (Ekman), mean of the 7 days before",
               "cold_band": "distance offshore where SST first exceeds nearshore (5-15 km) SST + 1 degC",
               "dT": "SST 150-200 km offshore minus SST 5-15 km offshore (upwelling strength)",
               "relationship": rel, "analog_test": analog, "families": fam}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "daily.csv").open("w", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None or (isinstance(r[k], float) and not np.isfinite(r[k])) else str(round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in keys) + "\n")

    # ---- charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, s in zip(axs, sector):
        x = np.array([r[f"cui_{s}"] for r in rows], float)
        y = np.array([np.nan if r[f"dT_{s}"] is None else r[f"dT_{s}"] for r in rows], float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=3, alpha=.25, color="#378ADD")
        bins = np.nanpercentile(x[ok], np.linspace(0, 100, 11))
        mids = [np.nanmedian(y[ok & (x >= a) & (x < b)]) for a, b in zip(bins[:-1], bins[1:])]
        ax.plot((bins[:-1] + bins[1:]) / 2, mids, color="#D85A30", lw=2.5)
        ax.set_title(f"{s}: r = {rel[s]['corr_cui_dT']}")
        ax.set_xlabel("Upwelling wind index, 7 days before")
        ax.set_ylabel("Offshore minus nearshore SST (degC)")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "wind_vs_upwelling.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(offs, bins=np.arange(0, 190, 10), color="#7F77DD")
    ax.axvline(15, color="#D85A30", ls="--")
    ax.set_xlabel("Days between the date and its best analog's calendar date (other years)")
    ax.set_ylabel("Number of days tested")
    ax.set_title(f"Best analog more than 30 days from the same date: {int(analog['share_best_analog_more_than_30_days_from_date'] * 100)}%")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "analog_offset.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for f_ in fam:
        ax.plot(range(1, 13), f_["months"], marker="o", label=f"Pattern {f_['family']} ({f_['days']} days, {f_['years']} years)")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Days")
    ax.set_title("When each sea pattern occurs")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "patterns_by_month.png", dpi=130)
    print(json.dumps({"relationship": rel, "analog_test": analog}, indent=1))


if __name__ == "__main__":
    main()
