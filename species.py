"""
Species layer for Madar Offshore (large pelagic vessels).

One index cannot serve every species: sardine follows cold upwelled water, sardinella the warm side,
mackerel follows its prey, horse mackerel the shelf edge and the bottom by day. So each species gets:

  * a habitat gate: temperature range and depth range (trapezoids, config.json -> "species");
  * its own indicator weights, different by night and by day;
  * time lags: food = chlorophyll ~8 days ago (zooplankton needs time to grow on a bloom),
    "biological" upwelling = winds 4-10 days ago, "physical" upwelling = winds 0-3 days ago.

score(species, period) = gate^0.7 x weighted sum of indicators   (0..1, absolute, not rescaled per day)

The numbers in config.json are starting values from the literature on the Canary Current upwelling
system. They are NOT calibrated yet: every captain report (reports.csv, column "species") is used both
as a boost here and, later, to fit the weights.
"""
import datetime as dt
import json
import math
import urllib.request

import numpy as np

import radar

CFG = radar.CFG

DEFAULT_SPECIES = [
    {"id": "sardine", "ar": "السردين", "fr": "Sardine", "en": "Sardine", "sci": "Sardina pilchardus",
     "aliases": ["sardine", "sardina", "سردين", "السردين"],
     "temp": [13, 15, 21.5, 24], "depth": [15, 25, 90, 130],
     "night": {"front_cold": 0.30, "food_lag": 0.25, "upwell_bio": 0.25, "food_now": 0.20},
     "day": {"front_cold": 0.20, "food_lag": 0.30, "upwell_bio": 0.25, "food_now": 0.10, "bottom_ok": 0.15},
     "layer": {"night": "0-30", "day": "30-80"}},
    {"id": "sardinella", "ar": "السردينيلا", "fr": "Sardinelle", "en": "Sardinella", "sci": "Sardinella aurita / S. maderensis",
     "aliases": ["sardinella", "sardinelle", "سردينيلا", "السردينيلا"],
     "temp": [19.5, 21.5, 27, 29], "depth": [10, 20, 80, 120],
     "night": {"front_warm": 0.30, "weak_upwell": 0.25, "food_lag": 0.20, "food_now": 0.25},
     "day": {"front_warm": 0.20, "weak_upwell": 0.25, "food_lag": 0.25, "food_now": 0.15, "bottom_ok": 0.15},
     "layer": {"night": "0-25", "day": "20-60"}},
    {"id": "mackerel", "ar": "الماكريل", "fr": "Maquereau", "en": "Mackerel", "sci": "Scomber colias",
     "aliases": ["mackerel", "maquereau", "ماكريل", "الماكريل", "كبايلا"],
     "temp": [14, 16, 21, 23.5], "depth": [30, 60, 200, 300],
     "night": {"front": 0.30, "prey": 0.35, "food_lag": 0.15, "shelf_edge": 0.20},
     "day": {"front": 0.20, "prey": 0.35, "shelf_edge": 0.30, "bottom_ok": 0.15},
     "layer": {"night": "0-40", "day": "50-150"}},
    {"id": "horse_mackerel", "ar": "الشرن", "fr": "Chinchard", "en": "Horse mackerel", "sci": "Trachurus trachurus / T. trecae",
     "aliases": ["horse_mackerel", "horse mackerel", "chinchard", "شرن", "الشرن"],
     "temp": [12, 14, 20, 22.5], "depth": [50, 80, 250, 400], "day_temp_on": "bottom", "night_temp_on": "bottom",
     "night": {"shelf_edge": 0.30, "food_lag": 0.30, "front": 0.20, "prey": 0.20},
     "day": {"shelf_edge": 0.40, "bottom_ok": 0.30, "food_lag": 0.20, "prey": 0.10},
     "layer": {"night": "20-80", "day": "80-250"}},
]
SPECIES = CFG.get("species") or DEFAULT_SPECIES
STRIDE = 2   # species grids are stored at every 2nd cell (~6 km) to keep the page light


# ----------------------------------------------------------------- inputs with time lags
def wind_history(today, demo):
    """Daily Coastal Upwelling Index per coast point for the last 11 days + today (Open-Meteo)."""
    import forecast

    pts = forecast.COAST_POINTS
    if demo:
        rng = np.random.default_rng(int(today.strftime("%Y%m%d")) + 7)
        out = []
        for i, p in enumerate(pts):
            out.append({str(today - dt.timedelta(days=k)): float(120 + 80 * math.sin(k / 2.5 + i) + rng.normal(0, 20)) for k in range(0, 12)})
        return out
    lats = ",".join(str(p[0]) for p in pts)
    lons = ",".join(str(p[1]) for p in pts)
    url = ("https://api.open-meteo.com/v1/forecast?latitude=" + lats + "&longitude=" + lons
           + "&hourly=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms&past_days=11&forecast_days=1&timezone=UTC")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode())
    if isinstance(data, dict):
        data = [data]
    out = []
    for p, d in zip(pts, data):
        h, per = d["hourly"], {}
        for t, s, w in zip(h["time"], h["wind_speed_10m"], h["wind_direction_10m"]):
            if s is not None and w is not None:
                per.setdefault(t[:10], []).append(forecast.coastal_upwelling_index(s, w, p[2], p[0]))
        out.append({k: float(np.mean(v)) for k, v in per.items()})
    return out


def lagged(cui, today, lags):
    """Mean upwelling index per coast point over the given day lags (e.g. range(4, 11))."""
    vals = []
    for pt in cui:
        v = [pt[str(today - dt.timedelta(days=k))] for k in lags if str(today - dt.timedelta(days=k)) in pt]
        vals.append(float(np.mean(v)) if v else 0.0)
    return np.array(vals)


def regrid(da, lats, lons):
    la = "latitude" if "latitude" in da.coords else "lat"
    lo = "longitude" if "longitude" in da.coords else "lon"
    v = da.sortby(la).interp({la: np.sort(lats), lo: lons}, method="linear").sortby(la, ascending=False).values.astype("float32")
    while v.ndim > 2:
        v = v[0]
    return v


def chl_lagged(day, box, lats, lons, demo, lag_days=8):
    """Chlorophyll about `lag_days` before the data day: the bloom that today's zooplankton grew on."""
    if demo:
        _, _, _, c, _ = radar.load_demo(day - dt.timedelta(days=lag_days), box)
        return (c if c.shape == (len(lats), len(lons)) else None), str(day - dt.timedelta(days=lag_days))
    try:
        da, used = radar.fetch_any(radar.CHL_ERDDAP, radar.CHL_CANDIDATES, day - dt.timedelta(days=lag_days), box,
                                   f"CHL-{lag_days}d", erddap_first=False)
        return regrid(da, lats, lons), used
    except SystemExit as e:
        print("[warn] no lagged chlorophyll:", str(e).splitlines()[0])
        return None, None


def bottom_temperature(day, box, lats, lons, demo):
    """Sea-floor temperature (Copernicus global model ~9 km) on the radar grid, or None."""
    if demo:
        return None, None
    try:
        import coastal

        da, used = radar.fetch(coastal.BOTTOM_T, day, box, "Bottom temperature")
        v = regrid(da, lats, lons)
        if np.nanmean(v) > 200:
            v = v - 273.15
        radar.SOURCES_EXTRA["BOTTOM_T"] = f"Copernicus model {used}"
        return v, used
    except (Exception, SystemExit) as e:  # noqa: BLE001
        print("[warn] no bottom temperature:", str(e).splitlines()[0])
        return None, None


# ----------------------------------------------------------------- indicators
def trap(x, a, b, c, d):
    x = np.asarray(x, dtype="float32")
    out = np.minimum(np.clip((x - a) / max(b - a, 1e-6), 0, 1), np.clip((d - x) / max(d - c, 1e-6), 0, 1))
    return np.where(np.isfinite(x), out, np.nan)


def chl_ok(chl):
    if chl is None:
        return None
    has = np.isfinite(chl) & (chl > 0)
    lc = np.log10(np.where(has, chl, 0.1))
    v = np.exp(-((lc - CFG.get("chl_best_log10", 0.25)) ** 2) / 0.35)
    return np.where(has, v, np.nan)


def indicators(lats, lons, sst, chl, chl_lag, depth, bottom_t, up_bio, up_phys):
    from scipy.ndimage import gaussian_filter
    import forecast

    ocean = np.isfinite(sst)
    step_km = abs(lats[1] - lats[0]) * 111
    fill = np.where(ocean, sst, np.nanmean(sst))
    sigma = max(1.0, 30 / step_km)                       # ~30 km neighbourhood
    anom = sst - gaussian_filter(fill, sigma)
    cold = 1 / (1 + np.exp(anom / 0.4))                   # 1 = colder than surroundings
    front = np.clip(radar.SCORE_PARTS["front_abs"] / CFG.get("strong_front_c_per_km", 0.10), 0, 1) if radar.SCORE_PARTS else np.zeros_like(sst)

    ref = CFG.get("cui_strong", 200.0)
    ub = np.clip(forecast.upwelling_boost(lats, lons, ocean, up_bio), 0, 1)
    up = np.clip(forecast.upwelling_boost(lats, lons, ocean, up_phys), 0, 1)

    shelf = None
    if depth is not None:
        d = np.where(np.isfinite(depth), depth, 0)
        gy, gx = np.gradient(d, step_km)
        slope = np.hypot(gx, gy)                          # metres of depth per km
        slope = gaussian_filter(slope, 1.0)
        rng = trap(depth, 60, 100, 300, 500)              # the shelf break, not the deep slope
        shelf = np.clip(slope / (np.nanpercentile(slope[ocean], 95) or 1), 0, 1) * np.nan_to_num(rng)

    ind = {
        "front": front, "front_cold": front * cold, "front_warm": front * (1 - cold),
        "food_now": chl_ok(chl), "food_lag": chl_ok(chl_lag),
        "upwell_bio": ub, "weak_upwell": 1 - up,
        "shelf_edge": shelf,
    }
    ctx = {"ocean": ocean, "depth": depth, "bottom_t": bottom_t, "sst": sst, "up_ref": ref}
    return ind, ctx


def habitat(sp, period, ind, ctx):
    t_src = ctx["bottom_t"] if (sp.get(period + "_temp_on") == "bottom" and ctx["bottom_t"] is not None) else ctx["sst"]
    gate = np.nan_to_num(trap(t_src, *sp["temp"]))
    if ctx["depth"] is not None:
        gate = gate * np.nan_to_num(trap(ctx["depth"], *sp["depth"]))
    return gate


def score_species(sp, period, ind, ctx, prey):
    from scipy.ndimage import gaussian_filter  # noqa: F401

    gate = habitat(sp, period, ind, ctx)
    weights = dict(sp[period])
    used, missing, total = {}, [], np.zeros_like(gate)
    for k, w in weights.items():
        if k == "prey":
            v = prey
        elif k == "bottom_ok":
            v = trap(ctx["bottom_t"], *sp["temp"]) if ctx["bottom_t"] is not None else None
        else:
            v = ind.get(k)
        if v is None or not np.isfinite(v[ctx["ocean"]]).any() or np.isfinite(v[ctx["ocean"]]).mean() < 0.15:
            missing.append(k)
            continue
        used[k] = (w, np.nan_to_num(v, nan=0.5))
    wsum = sum(w for w, _ in used.values()) or 1
    for k, (w, v) in used.items():
        total = total + (w / wsum) * v
    s = np.power(gate, 0.7) * total
    s[~ctx["ocean"]] = 0
    return s, {k: v for k, (_, v) in used.items()}, {k: w / wsum for k, (w, _) in used.items()}, missing


def pick(lats, lons, s, parts, wts, ctx, sp, period, n=3):
    from scipy.ndimage import maximum_filter

    port = CFG["port"]
    peaks = (s == maximum_filter(s, size=7)) & (s > 0.15)
    cand = []
    for j, i in zip(*np.nonzero(peaks)):
        la, lo = float(lats[j]), float(lons[i])
        if radar.in_closed(la, lo, CFG.get("closed_zones", [])):
            continue
        dnm = radar.nm_between(port["lat"], port["lon"], la, lo)
        if dnm > CFG.get("max_range_nm", 150):
            continue
        cand.append((float(s[j, i]), j, i, dnm))
    cand.sort(reverse=True)
    out = []
    for v, j, i, dnm in cand:
        la, lo = float(lats[j]), float(lons[i])
        if any(math.hypot(la - z["lat"], (lo - z["lon"]) * 0.92) < 0.3 for z in out):
            continue
        contrib = sorted(((k, wts[k] * float(parts[k][j, i])) for k in parts), key=lambda x: -x[1])
        dep = ctx["depth"][j, i] if ctx["depth"] is not None else np.nan
        bt = ctx["bottom_t"][j, i] if ctx["bottom_t"] is not None else np.nan
        out.append({"id": "ABC"[len(out)], "lat": round(la, 4), "lon": round(lo, 4),
                    "suit": round(min(0.95, v) * 100), "distance_nm": round(dnm),
                    "bearing_deg": round(radar.bearing(port["lat"], port["lon"], la, lo)),
                    "depth_m": int(round(float(dep))) if np.isfinite(dep) else None,
                    "sst": round(float(ctx["sst"][j, i]), 1),
                    "bottom_t": round(float(bt), 1) if np.isfinite(bt) else None,
                    "f": {k: round(float(parts[k][j, i]), 2) for k in parts},
                    "top": [k for k, _ in contrib[:3]]})
        if len(out) == n:
            break
    return out


def pack(a, ocean):
    sub, oc = a[::STRIDE, ::STRIDE], ocean[::STRIDE, ::STRIDE]
    return [None if (not o or not np.isfinite(v)) else int(round(float(v) * 99)) for v, o in zip(sub.ravel(), oc.ravel())]


def report_boost(s, lats, lons, reports, sp):
    names = {a.lower() for a in sp.get("aliases", [])} | {sp["id"]}
    LON, LAT = np.meshgrid(lons, lats)
    coslat = math.cos(math.radians(float(np.mean(lats))))
    n = 0
    for r in reports:
        if str(r.get("species", "")).strip().lower() not in names:
            continue
        w = min(1.0, r["tonnes"] / 40) * math.exp(-r["age"] / 5)
        d2 = (LAT - r["lat"]) ** 2 + ((LON - r["lon"]) * coslat) ** 2
        s = s + 0.15 * w * np.exp(-d2 / 0.02)
        n += 1
    return np.clip(s, 0, 1), n


# ----------------------------------------------------------------- main entry (called by radar.py)
def build(lats, lons, sst, chl, depth, data_day, box, demo, reports):
    from scipy.ndimage import gaussian_filter

    day = dt.date.fromisoformat(data_day)
    chl_lag, lag_used = chl_lagged(day, box, lats, lons, demo)
    bottom_t, _ = bottom_temperature(day, box, lats, lons, demo)
    try:
        cui = wind_history(day, demo)
        up_bio, up_phys = lagged(cui, day, range(4, 11)), lagged(cui, day, range(0, 4))
    except Exception as e:  # noqa: BLE001
        print("[warn] wind history:", e)
        up_bio = up_phys = np.zeros(len(__import__("forecast").COAST_POINTS))
    ind, ctx = indicators(lats, lons, sst, chl, chl_lag, depth, bottom_t, up_bio, up_phys)
    step_km = abs(lats[1] - lats[0]) * 111

    out = {"stride": STRIDE, "lags": {"chl_lag_day": lag_used, "upwell_bio_days": "4-10", "upwell_phys_days": "0-3",
                                      "upwell_bio": round(float(np.mean(up_bio)), 1), "upwell_phys": round(float(np.mean(up_phys)), 1),
                                      "bottom_t": bottom_t is not None},
           "list": []}
    for period in ("night", "day"):
        # prey for mackerel / horse mackerel: where small pelagics (sardine, sardinella) find habitat + food, spread ~20 km
        small = []
        for sp in SPECIES:
            if sp["id"] in ("sardine", "sardinella"):
                g = habitat(sp, period, ind, ctx)
                food = ind["food_lag"] if ind["food_lag"] is not None else ind["food_now"]
                small.append(g * (np.nan_to_num(food, nan=0.5) if food is not None else 0.5))
        prey = gaussian_filter(np.maximum.reduce(small), max(1.0, 20 / step_km)) if small else None
        if prey is not None:
            prey = prey / (np.nanpercentile(prey[ctx["ocean"]], 98) or 1)
            prey = np.clip(prey, 0, 1)
        for sp in SPECIES:
            s, parts, wts, missing = score_species(sp, period, ind, ctx, prey)
            s, nrep = report_boost(s, lats, lons, reports, sp)
            zones = pick(lats, lons, s, parts, wts, ctx, sp, period)
            entry = next((e for e in out["list"] if e["id"] == sp["id"]), None)
            if entry is None:
                entry = {k: sp.get(k) for k in ("id", "ar", "fr", "en", "sci", "temp", "depth", "layer")}
                entry["periods"] = {}
                out["list"].append(entry)
            entry["periods"][period] = {"zones": zones, "weights": {k: round(v, 2) for k, v in wts.items()},
                                        "missing": missing, "reports": nrep, "grid": pack(s, ctx["ocean"])}
            print(f"[ok] species {sp['id']:15s} {period:5s} zones={len(zones)} best={zones[0]['suit'] if zones else '-'}% missing={missing}")
    return out
