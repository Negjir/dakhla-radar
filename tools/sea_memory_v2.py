"""
Sea memory v2: compare days by temperature + food (chlorophyll) + currents, not temperature alone.

Same area and dates as tools/front_regimes.py (MUR SST every 2nd day since 2002), plus:
  * chlorophyll: Copernicus GlobColour L4 gap-free, 4 km, daily (multi-year, then near-real-time);
  * surface currents u, v: Copernicus GLORYS 1/12 deg daily (reanalysis, interim, then analysis/forecast).

Tests, without catch data:
  A. Do the closest past days change when food and currents are added?
  B. "Same future" test: for each day, take the 10 closest past days (other years) and use what the sea
     looked like 6 days after them to predict the sea 6 days later. Lower error = the matching finds days
     that really behave alike. Compared: same calendar date, temperature only, temperature + food + currents.

Outputs: docs/analysis/fronts/summary_v2.json and charts.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import front_regimes as fr  # noqa: E402

OUT = fr.OUT
CACHE = ROOT / "data" / "cache" / "cmems_fronts"
BOX = dict(minimum_longitude=fr.LON0, maximum_longitude=fr.LON1, minimum_latitude=fr.LAT0, maximum_latitude=fr.LAT1)
CHL_IDS = ["cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
           "cmems_obs-oc_glo_bgc-plankton_myint_l4-gapfree-multi-4km_P1D",
           "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"]
CUR_IDS = ["cmems_mod_glo_phy_my_0.083deg_P1D-m",
           "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
           "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"]
LEAD = 3   # steps of 2 days -> 6 days ahead
K = 10


def cm_year(kind, ids, variables, year, depth=False):
    """One year of a Copernicus product over the box, saved once (tries each dataset id in turn)."""
    import copernicusmarine as cm
    import xarray as xr

    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{kind}_{year}.nc"
    if f.exists():
        return xr.open_dataset(f)
    a = f"{max(year, 2002)}-01-01T00:00:00"
    b = f"{year}-12-31T23:59:59"
    parts = []
    for ds_id in ids:
        try:
            kw = dict(dataset_id=ds_id, variables=variables, start_datetime=a, end_datetime=b, **BOX)
            if depth:
                kw.update(minimum_depth=0, maximum_depth=1)
            try:
                ds = cm.open_dataset(coordinates_selection_method="inside", **kw)
            except TypeError:
                ds = cm.open_dataset(**kw)
            if "depth" in ds.dims:
                ds = ds.isel(depth=0)
            ds = ds[variables].load()
            if ds.sizes.get("time", 0):
                parts.append(ds)
                print(f"[ok] {kind} {year} {ds_id}: {ds.sizes['time']} days")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {kind} {year} {ds_id}: {str(e).splitlines()[0][:150]}")
    if not parts:
        return None
    ds = xr.concat(parts, "time") if len(parts) > 1 else parts[0]
    _, keep = np.unique(ds["time"].values, return_index=True)
    ds = ds.isel(time=np.sort(keep))
    ds.to_netcdf(f)
    return ds


def on_grid(ds, var, days, lats, lons):
    """Variable from a yearly dataset, nearest day (<=1 day), interpolated onto the SST grid."""
    la = "latitude" if "latitude" in ds.coords else "lat"
    lo = "longitude" if "longitude" in ds.coords else "lon"
    da = ds[var].sortby(la)
    times = da["time"].values.astype("datetime64[D]")
    out = np.full((len(days), len(lats), len(lons)), np.nan, "float32")
    for i, d in enumerate(days):
        k = np.searchsorted(times, np.datetime64(d))
        best = None
        for j in (k - 1, k):
            if 0 <= j < len(times) and abs((times[j] - np.datetime64(d)).astype(int)) <= 1:
                best = j
        if best is None:
            continue
        out[i] = da.isel(time=best).interp({la: lats, lo: lons}, method="linear").values
    return out


def block(cube, good, log=False, demean=True, n=8):
    """PCA scores of one variable's daily maps (pattern), scaled so the block has unit total variance."""
    X = cube.reshape(cube.shape[0], -1).astype("float64")
    if log:
        X = np.log10(np.clip(X, 0.01, None))
    cols = np.isfinite(X[good]).mean(0) > 0.9
    X = X[:, cols]
    rowmean = np.nanmean(X, 1, keepdims=True)
    X = np.where(np.isfinite(X), X, rowmean)
    if demean:
        X = X - rowmean
    mu = X[good].mean(0)
    U, S, Vt = np.linalg.svd(X[good] - mu, full_matrices=False)
    Z = (X - mu) @ Vt[:n].T
    scale = np.sqrt((Z[good] ** 2).sum(1).mean())
    return Z / scale, float((S[:n] ** 2).sum() / (S ** 2).sum())


def main():
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
    print("SST", sst.shape)

    chl = np.full(sst.shape, np.nan, "float32")
    u = np.full(sst.shape, np.nan, "float32")
    v = np.full(sst.shape, np.nan, "float32")
    cover = {"chl": [], "cur": []}
    for y in years:
        idx = [i for i, d in enumerate(days) if d.year == y]
        if not idx:
            continue
        dy = [days[i] for i in idx]
        c = cm_year("chl", CHL_IDS, ["CHL"], y)
        if c is not None:
            chl[idx] = on_grid(c, "CHL", dy, lats, lons)
            cover["chl"].append(y)
        cu = cm_year("cur", CUR_IDS, ["uo", "vo"], y, depth=True)
        if cu is not None:
            u[idx] = on_grid(cu, "uo", dy, lats, lons)
            v[idx] = on_grid(cu, "vo", dy, lats, lons)
            cover["cur"].append(y)
    print("coverage", cover)

    oc = np.isfinite(sst).mean(0) > 0.95
    def ok_frac(c):
        return np.array([np.isfinite(c[i][oc]).mean() for i in range(len(days))])
    good = (ok_frac(sst) > 0.98) & (ok_frac(chl) > 0.6) & (ok_frac(u) > 0.6)
    print("days with all three:", int(good.sum()), "of", len(days))

    Zs, es = block(sst, good)
    Zc, ec = block(chl, good, log=True)
    Zu, eu = block(u, good, demean=False)
    Zv, ev = block(v, good, demean=False)
    Z_sst = Zs
    Z_all = np.hstack([Zs, Zc, Zu * np.sqrt(0.5), Zv * np.sqrt(0.5)])   # SST, food, currents: one third each
    Z_eval = np.hstack([Zs, Zc])        # what we try to predict 6 days later: temperature + food pattern

    gi = np.nonzero(good)[0]
    yrs = np.array([d.year for d in days])
    doy = np.array([d.timetuple().tm_yday for d in days])
    has_future = np.zeros(len(days), bool)
    for i in gi:
        j = i + LEAD
        if j < len(days) and good[j] and (days[j] - days[i]).days == 2 * LEAD:
            has_future[i] = True
    cand = np.nonzero(has_future)[0]
    rng = np.random.default_rng(3)
    test = rng.choice(cand, size=min(1500, len(cand)), replace=False)

    err = {"climatology": [], "same_date": [], "sst_only": [], "sst_food_currents": [], "persistence": []}
    overlap, offs_all, offs_sst = [], [], []
    clim = Z_eval[cand].mean(0)
    for i in test:
        pool = cand[yrs[cand] != yrs[i]]
        truth = Z_eval[i + LEAD]
        cal = np.abs(((doy[pool] - doy[i] + 182) % 365) - 182)
        same = pool[cal <= 7]
        e = lambda pred: float(np.sqrt(((pred - truth) ** 2).sum()))  # noqa: E731
        err["climatology"].append(e(clim))
        err["persistence"].append(e(Z_eval[i]))
        if len(same):
            err["same_date"].append(e(Z_eval[same + LEAD].mean(0)))
        res = {}
        for name, Z in (("sst_only", Z_sst), ("sst_food_currents", Z_all)):
            dist = np.sqrt(((Z[pool] - Z[i]) ** 2).sum(1))
            best = pool[np.argsort(dist)[:K]]
            res[name] = best
            err[name].append(e(Z_eval[best + LEAD].mean(0)))
        overlap.append(len(set(res["sst_only"]) & set(res["sst_food_currents"])) / K)
        for name, store in (("sst_only", offs_sst), ("sst_food_currents", offs_all)):
            b = res[name][0]
            store.append(int(abs(((doy[b] - doy[i] + 182) % 365) - 182)))

    med = {k: round(float(np.median(v_)), 3) for k, v_ in err.items() if v_}
    base = med["climatology"]
    skill = {k: round(1 - med[k] / base, 3) for k in med}
    summary = {
        "made": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "period": [str(days[0]), str(days[-1])],
        "years_with_chlorophyll": cover["chl"], "years_with_currents": cover["cur"],
        "days_with_all_three": int(good.sum()), "days_tested": int(len(test)),
        "variance_explained": {"sst": round(es, 2), "chl": round(ec, 2), "u": round(eu, 2), "v": round(ev, 2)},
        "lead_days": 2 * LEAD, "analogs_per_day": K,
        "forecast_error_median": med,
        "skill_vs_climatology": skill,
        "share_same_10_closest_days_sst_vs_all": round(float(np.mean(overlap)), 2),
        "best_analog_more_than_30_days_from_date": {"sst_only": round(float((np.array(offs_sst) > 30).mean()), 2),
                                                      "sst_food_currents": round(float((np.array(offs_all) > 30).mean()), 2)},
        "best_analog_calendar_offset_median": {"sst_only": int(np.median(offs_sst)), "sst_food_currents": int(np.median(offs_all))},
        "notes": ["Errors are measured on the temperature + chlorophyll pattern 6 days later (PCA space).",
                  "Skill = 1 - error / error of the long-term average state. Higher is better.",
                  "Persistence = 'the sea in 6 days will look like today'.",
                  "No catch data is used: this tests whether the matching finds days that evolve alike, not where fish were."],
    }
    (OUT / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [("same_date", "Same date\nother years"), ("sst_only", "Similar days\n(temperature)"),
             ("sst_food_currents", "Similar days\n(temp + food + currents)"), ("persistence", "Sea stays\nas today")]
    vals = [skill.get(k, np.nan) for k, _ in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    cols = ["#B4B2A9", "#85B7EB", "#185FA5", "#D3D1C7"]
    bars = ax.bar([n for _, n in names], vals, color=cols)
    for b_, v_ in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v_ + 0.01, f"{v_:.2f}", ha="center")
    ax.axhline(0, color="#444441", lw=1)
    ax.set_ylabel("Skill predicting the sea 6 days later\n(0 = long-term average, higher is better)")
    ax.set_title("Which past days best predict the sea's next week?")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "analog_skill_v2.png", dpi=130)


if __name__ == "__main__":
    main()
