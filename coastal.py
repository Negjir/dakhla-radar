"""
Dakhla Coastal Radar - satellite-only fishing zones for small artisanal boats
(group 1: species that follow water masses and bait: meagre/courbine, bluefish/tassergal,
mullet, bonito).

For each landing point (Lassarga, Ntireft, Labouirda, Imoutlane):
  * a 500 m grid on the coastal strip (0.4 to MAX_KM km from shore)
  * thermal fronts from NASA MUR SST (1 km, daily) via NOAA ERDDAP (no account needed)
  * chlorophyll level + fronts from Copernicus Marine ocean colour (300 m -> 1 km -> 4 km fallback)
  * coastal upwelling from forecast winds (Open-Meteo)
  * tide times and sea state (Open-Meteo Marine, approximate)
Outputs: docs/coastal/index.html, one map per point, WhatsApp messages, coastal.json.

Usage:
  python coastal.py          # real data
  python coastal.py --demo   # synthetic data to test the pipeline
"""
import argparse
import datetime as dt
import html
import json
import math
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

import radar  # config + helpers (fetch for Copernicus, pos/dm formatting)

CFG = radar.CFG
C = CFG.get("coastal", {})
OUT = radar.OUT / "coastal"

POINTS = C.get("points") or [
    {"id": "lassarga", "ar": "لاسارگا", "fr": "Lassarga", "lat": 23.60, "lon": -16.00, "approx": True},
    {"id": "ntireft", "ar": "انتيرفت", "fr": "Ntireft", "lat": 24.081, "lon": -15.567},
    {"id": "labouirda", "ar": "لبويردة", "fr": "Labouirda", "lat": 23.208, "lon": -16.095},
    {"id": "imoutlane", "ar": "إيموتلان", "fr": "Imoutlane", "lat": 24.367, "lon": -15.267},
]
MAX_KM = C.get("max_km", 8)
RANGE_KM = C.get("range_km", 15)       # max trip from the landing point
BOX = C.get("box_deg", 0.16)          # half-size of the area around each point
CELL = C.get("cell_deg", 0.005)       # ~500 m
SPECIES = C.get("species_ar", "الكوربين، التاسرگال، البوري، البونيت")
ERDDAP = C.get("erddap", "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41")
CHL_CANDIDATES = C.get("chl_datasets") or [
    ["cmems_obs-oc_glo_bgc-plankton_nrt_l3-olci-300m_P1D", "CHL"],
    ["cmems_obs-oc_atl_bgc-plankton_nrt_l3-multi-1km_P1D", "CHL"],
    ["cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"],
]
COAST_DIR = C.get("coast_dir_deg", 210)   # direction the coast runs toward (southwards)


# ----------------------------------------------------------------- data
def all_box():
    lat0 = min(p["lat"] for p in POINTS) - BOX - 0.05
    lat1 = max(p["lat"] for p in POINTS) + BOX + 0.05
    lon0 = min(p["lon"] for p in POINTS) - BOX - 0.05
    lon1 = max(p["lon"] for p in POINTS) + BOX + 0.05
    return lat0, lat1, lon0, lon1


def mur_sst():
    """Latest MUR SST (0.01 deg) for the whole coast, from ERDDAP. Returns lats(desc), lons, sst(C), date."""
    import xarray as xr

    lat0, lat1, lon0, lon1 = all_box()
    url = f"{ERDDAP}.nc?analysed_sst[(last)][({lat0:.3f}):({lat1:.3f})][({lon0:.3f}):({lon1:.3f})]"
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        with urllib.request.urlopen(url, timeout=180) as r:
            f.write(r.read())
        path = f.name
    ds = xr.open_dataset(path)
    da = ds["analysed_sst"].isel(time=0).sortby("latitude", ascending=False)
    sst = da.values.astype("float32")
    if np.nanmean(sst) > 200:
        sst -= 273.15
    day = str(np.datetime_as_string(ds["time"].values[0], unit="D"))
    print(f"[ok] MUR SST {day} {sst.shape}")
    return da["latitude"].values, da["longitude"].values, sst, day


def ostia_fallback(day):
    lat0, lat1, lon0, lon1 = all_box()
    box = {"lat_min": lat0, "lat_max": lat1, "lon_min": lon0, "lon_max": lon1}
    da, used = radar.fetch(radar.SST_CANDIDATES, day, box, "SST (fallback OSTIA)")
    la = "latitude" if "latitude" in da.coords else "lat"
    da = da.sortby(la, ascending=False)
    sst = da.values.astype("float32")
    if np.nanmean(sst) > 200:
        sst -= 273.15
    return da[la].values, da["longitude" if "longitude" in da.coords else "lon"].values, sst, used


def chlorophyll(day):
    lat0, lat1, lon0, lon1 = all_box()
    box = {"lat_min": lat0, "lat_max": lat1, "lon_min": lon0, "lon_max": lon1}
    try:
        da, used = radar.fetch(CHL_CANDIDATES, day, box, "CHL")
    except SystemExit as e:
        print("[warn] no chlorophyll today:", e)
        return None
    la = "latitude" if "latitude" in da.coords else "lat"
    lo = "longitude" if "longitude" in da.coords else "lon"
    return da.sortby(la, ascending=False), la, lo, used


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def conditions(p):
    """Wind upwelling (past 3 days + 2 forecast), tides and sea state near the point."""
    lat, lon = p["lat"], p["lon"] - 0.04   # a little offshore
    w = get_json("https://api.open-meteo.com/v1/forecast?latitude=%.3f&longitude=%.3f"
                 "&hourly=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms&past_days=3&forecast_days=2&timezone=Africa%%2FCasablanca"
                 % (lat, lon))
    m = get_json("https://marine-api.open-meteo.com/v1/marine?latitude=%.3f&longitude=%.3f"
                 "&hourly=sea_level_height_msl,wave_height&forecast_days=2&timezone=Africa%%2FCasablanca" % (lat, lon))
    return parse_conditions(w, m, lat)


def parse_conditions(w, m, lat):
    from forecast import coastal_upwelling_index

    h = w["hourly"]
    from zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("Africa/Casablanca")).replace(tzinfo=None)   # Open-Meteo times are local
    cui = []
    for t, s, d in zip(h["time"], h["wind_speed_10m"], h["wind_direction_10m"]):
        if s is None or d is None:
            continue
        age_h = (now - dt.datetime.fromisoformat(t)).total_seconds() / 3600
        if -6 <= age_h <= 72:      # last 3 days, weighted towards 1-2 days ago
            wgt = math.exp(-((age_h - 30) / 30) ** 2)
            cui.append((coastal_upwelling_index(s, d, COAST_DIR, lat), wgt))
    up = sum(c * w_ for c, w_ in cui) / max(1e-6, sum(w_ for _, w_ in cui))
    today = now.strftime("%Y-%m-%d")
    ws = [s for t, s in zip(h["time"], h["wind_speed_10m"]) if t.startswith(today) and s is not None]
    wd = [d for t, d in zip(h["time"], h["wind_direction_10m"]) if t.startswith(today) and d is not None]
    mh = m.get("hourly", {})
    tides = []
    sl = mh.get("sea_level_height_msl") or []
    tt = mh.get("time") or []
    for i in range(1, len(sl) - 1):
        if None in (sl[i - 1], sl[i], sl[i + 1]):
            continue
        if sl[i] > sl[i - 1] and sl[i] >= sl[i + 1]:
            tides.append(("high", tt[i]))
        elif sl[i] < sl[i - 1] and sl[i] <= sl[i + 1]:
            tides.append(("low", tt[i]))
    waves = [v for t, v in zip(tt, mh.get("wave_height") or []) if t.startswith(today) and v is not None]
    return {"upwelling": up,
            "wind_kn": round(max(ws) * 1.944) if ws else None,
            "wind_dir": round(float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(wd))), np.mean(np.cos(np.radians(wd))))) % 360)) if wd else None,
            "wave_max": round(max(waves), 1) if waves else None,
            "tides": [(k, t) for k, t in tides if t.startswith(today)][:4],
            "series": series(tt, sl, h["time"], h["wind_speed_10m"], h["wind_direction_10m"], today)}


def series(tt, sl, wt, wsp, wdir, today):
    """Hourly curves for the console: 30 h from today 00:00 (sea level m, wind kn, wind dir)."""
    i0 = next((i for i, t in enumerate(tt) if t.startswith(today)), 0)
    j0 = next((i for i, t in enumerate(wt) if t.startswith(today)), 0)
    rnd = lambda v, k=2: None if v is None else round(v, k)
    return {"t0": today + "T00:00",
            "sea": [rnd(v) for v in sl[i0:i0 + 30]],
            "wind": [rnd(v * 1.944 if v is not None else None, 1) for v in wsp[j0:j0 + 30]],
            "dir": [rnd(v, 0) for v in wdir[j0:j0 + 30]]}


# ----------------------------------------------------------------- demo data
def demo_all(day):
    lat0, lat1, lon0, lon1 = all_box()
    lats = np.arange(lat1, lat0, -0.01)
    lons = np.arange(lon0, lon1, 0.01)
    LON, LAT = np.meshgrid(lons, lats)
    coast = np.interp(LAT, [23.0, 23.2, 23.6, 24.08, 24.37, 24.6], [-16.15, -16.08, -15.98, -15.55, -15.25, -15.05])
    d = (coast - LON) * 102
    rng = np.random.default_rng(int(day.replace("-", "")))
    from scipy.ndimage import gaussian_filter

    n = gaussian_filter(rng.standard_normal(LAT.shape), 4) * 8
    sst = 19.5 + (24.5 - LAT) * 0.5 - 2.4 * np.exp(-np.clip(d, 0, None) / np.clip(6 + 3 * n, 2, None)) + 0.6 * n
    for _ in range(14):   # cold filaments / warm intrusions crossing the coastal strip
        cy = rng.uniform(lat0, lat1)
        cx = np.interp(cy, [23.0, 23.2, 23.6, 24.08, 24.37, 24.6], [-16.15, -16.08, -15.98, -15.55, -15.25, -15.05]) - rng.uniform(0.02, 0.09)
        rad = rng.uniform(0.015, 0.04)
        sst += rng.choice([-1, 1]) * rng.uniform(0.6, 1.3) * np.exp(-(((LON - cx) / rad) ** 2 + ((LAT - cy) / (rad * 1.6)) ** 2))
    sst[d < 0] = np.nan
    chl = 10 ** (0.1 + 0.8 * np.exp(-np.clip(d, 0, None) / 7) + 0.25 * n)
    chl[d < 0] = np.nan
    chl[(gaussian_filter(rng.standard_normal(LAT.shape), 6) > 0.12)] = np.nan   # clouds
    return lats, lons, sst.astype("float32"), chl.astype("float32")


def demo_conditions(p, day):
    rng = np.random.default_rng(int(day.replace("-", "")) + int(p["lat"] * 100))
    base = dt.datetime.fromisoformat(day + "T00:00")
    hi = 3 + rng.random() * 3
    amp = 0.8 + 0.3 * rng.random()
    w0, d0 = 9 + rng.random() * 6, 15 + rng.random() * 20
    ser = {"t0": day + "T00:00", "sea": [round(amp * math.cos(2 * math.pi * (h - hi) / 12.42), 2) for h in range(30)],
           "wind": [round(w0 + 7 * math.exp(-((h - 15) / 4.5) ** 2) + rng.random(), 1) for h in range(30)],
           "dir": [round(d0 + 8 * math.sin(h / 5)) for h in range(30)]}
    return {"series": ser, "upwelling": 120 + 120 * rng.random(), "wind_kn": int(10 + rng.random() * 12), "wind_dir": int(10 + rng.random() * 30),
            "wave_max": round(0.8 + rng.random(), 1),
            "tides": [("high", (base + dt.timedelta(hours=hi)).isoformat()[:16]), ("low", (base + dt.timedelta(hours=hi + 6.2)).isoformat()[:16]),
                      ("high", (base + dt.timedelta(hours=hi + 12.4)).isoformat()[:16]), ("low", (base + dt.timedelta(hours=hi + 18.6)).isoformat()[:16])]}


# ----------------------------------------------------------------- scoring
def regrid(lats, lons, field, glat, glon):
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator((lats[::-1], lons), field[::-1], bounds_error=False, fill_value=np.nan, method="linear")
    G1, G2 = np.meshgrid(glat, glon, indexing="ij")
    return f(np.stack([G1, G2], -1)).astype("float32")


def km_between(la1, lo1, la2, lo2):
    dy = (la2 - la1) * 111.2
    dx = (lo2 - lo1) * 111.2 * math.cos(math.radians((la1 + la2) / 2))
    return math.hypot(dx, dy)


def bearing(la1, lo1, la2, lo2):
    dy = la2 - la1
    dx = (lo2 - lo1) * math.cos(math.radians(la2))
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def score_point(p, sst_all, chl_all, cond):
    from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter, sobel

    glat = np.arange(p["lat"] + BOX, p["lat"] - BOX, -CELL)
    glon = np.arange(p["lon"] - BOX - 0.05, p["lon"] + BOX, CELL)
    sst = regrid(*sst_all, glat, glon)
    ocean = np.isfinite(sst)
    if ocean.mean() < 0.05:
        return None
    km_cell = CELL * 111.2
    dist = distance_transform_edt(ocean) * km_cell
    # distance suitability: boats work close to shore
    D = np.exp(-((dist - 3.0) / 2.6) ** 2)
    D[(dist < 0.4) | (dist > MAX_KM)] = 0
    D[~ocean] = 0

    def fronts(field, sigma):
        f = np.where(np.isfinite(field), field, np.nanmedian(field))
        f = gaussian_filter(f, sigma)
        g = np.hypot(sobel(f, 0), sobel(f, 1)) / (8 * km_cell)
        g[~ocean] = 0
        ref = np.nanpercentile(g[D > 0], 95) if (D > 0).any() else 1
        return np.clip(g / (ref or 1), 0, 1)

    Fs = fronts(sst, 2.0)
    med = np.nanmedian(sst[D > 0]) if (D > 0).any() else np.nanmedian(sst)
    up = np.clip(cond["upwelling"] / CFG.get("cui_strong", 200.0), 0, 1.3)
    cold = 1 / (1 + np.exp(-(med - sst) / 0.35)) * up
    cold[~ocean] = 0

    cover = 0.0
    Fc = np.zeros_like(sst)
    chl = None
    if chl_all is not None:
        chl = regrid(*chl_all, glat, glon)
        valid = np.isfinite(chl) & ocean & (D > 0)
        cover = valid.sum() / max(1, ((D > 0) & ocean).sum())
        if cover > 0.15:
            lc = np.log10(np.clip(chl, 0.02, 80))
            level = np.exp(-((lc - 0.3) / 0.45) ** 2)
            Fc = 0.5 * fronts(lc, 1.5) + 0.5 * np.where(np.isfinite(lc), level, 0)
            Fc[~np.isfinite(chl)] = 0

    wc = 0.30 * min(1, cover * 1.4)
    G1, G2 = np.meshgrid(glat, glon, indexing="ij")
    trip = np.hypot((G1 - p["lat"]) * 111.2, (G2 - p["lon"]) * 111.2 * math.cos(math.radians(p["lat"])))
    fuel = 1 - 0.4 * np.clip(trip / RANGE_KM, 0, 1)   # nearer = less fuel (mild)
    fuel[trip > RANGE_KM] = 0
    S = np.sqrt(D) * fuel * (0.45 * Fs + wc * Fc + 0.25 * cold) / (0.45 + wc + 0.25)
    S[~ocean] = 0

    peaks = (S == maximum_filter(S, size=5)) & (S > 0)
    cand = sorted(((float(S[j, i]), float(glat[j]), float(glon[i]), float(dist[j, i])) for j, i in zip(*np.nonzero(peaks))), reverse=True)
    zones = []
    for v, la, lo, dk in cand:
        if all(km_between(la, lo, z["lat"], z["lon"]) > 2.5 for z in zones):
            zones.append({"id": "ABC"[len(zones)], "lat": round(la, 4), "lon": round(lo, 4), "score": v,
                          "km": round(km_between(p["lat"], p["lon"], la, lo), 1), "brg": round(bearing(p["lat"], p["lon"], la, lo)),
                          "shore_km": round(dk, 1)})
        if len(zones) == 3:
            break
    for z in zones:   # what made this spot stand out (for the console)
        j, i = int(np.argmin(abs(glat - z["lat"]))), int(np.argmin(abs(glon - z["lon"])))
        z["f"] = {"front": round(float(Fs[j, i]), 2), "chl": round(float(Fc[j, i]), 2) if cover > 0.15 else None,
                  "cold": round(float(np.clip(cold[j, i], 0, 1)), 2)}
        z["sst"] = round(float(sst[j, i]), 1)
        z["chl_mg"] = round(float(chl[j, i]), 2) if chl is not None and np.isfinite(chl[j, i]) else None
    # confidence: contrast of the best zone + data quality
    # confidence = how much the spot stands out from the rest of the strip x data quality
    inside = S[(D > 0.2) & (fuel > 0)]
    mean = float(inside.mean()) if inside.size else 1
    quality = 0.65 + 0.35 * min(1, cover * 1.4)          # no chlorophyll (clouds) -> lower
    for z in zones:
        c = z["score"] / (mean or 1) * quality
        z["ratio"] = round(c, 2)
        z["conf"] = "high" if c > 2.6 else "medium" if c > 1.8 else "low"
    return {"glat": glat, "glon": glon, "S": S, "sst": sst, "chl": chl, "ocean": ocean, "zones": zones, "cover": round(cover, 2), "dist": dist}


# ----------------------------------------------------------------- outputs
def grid_json(r):
    """Compact grids for the web console: score 0-99, SST x10, log10(chl) x100; null = land / no data."""
    S = r["S"] / (r["S"].max() or 1)

    def pack(a, k, mask):
        return [None if (not m or not np.isfinite(v)) else int(round(float(v) * k)) for v, m in zip(a.ravel(), mask.ravel())]

    out = {"n": int(r["S"].shape[0]), "m": int(r["S"].shape[1]), "lat0": round(float(r["glat"][0]), 4), "lon0": round(float(r["glon"][0]), 4),
           "cell": CELL, "score": pack(S, 99, r["ocean"]), "sst": pack(r["sst"], 10, r["ocean"])}
    if r["chl"] is not None:
        out["chl"] = pack(np.log10(np.clip(r["chl"], 0.01, 100)), 100, r["ocean"])
    return out


def console(data):
    """docs/coastal/index.html = the console template with today's data inlined (works offline and on file://)."""
    tpl = (Path(__file__).parent / "coastal_console.html").read_text(encoding="utf-8")
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    head = ('<!doctype html>\n<html lang="ar"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>\n')
    (OUT / "index.html").write_text(head + tpl.replace("/*DATA*/null", js) + "\n</body></html>\n", encoding="utf-8")



CONF_AR = {"high": "مرتفعة", "medium": "متوسطة", "low": "منخفضة"}
COMP_AR = ["شمال", "شمال شرق", "شرق", "جنوب شرق", "جنوب", "جنوب غرب", "غرب", "شمال غرب"]


def message(p, r, cond, day, sst_day):
    lines = [f"🎣 رادار الساحل · {p['ar']} · {day}", f"الأنواع: {SPECIES}", ""]
    if not r or not r["zones"]:
        lines.append("لا توجد جبهات واضحة قرب الساحل اليوم.")
    else:
        for z in r["zones"]:
            lines.append(f"{z['id']}: {radar.pos(z['lat'], z['lon'])}")
            lines.append(f"   {z['km']} كم {COMP_AR[round(z['brg'] / 45) % 8]} من {p['ar']} · {z['shore_km']} كم من الشاطئ · ثقة {CONF_AR[z['conf']]}")
    lines.append("")
    if cond.get("tides"):
        lines.append("المد (تقريبي): " + " · ".join(("عالٍ " if k == "high" else "منخفض ") + t[11:16] for k, t in cond["tides"]))
    sea = []
    if cond.get("wind_kn") is not None:
        sea.append(f"ريح حتى {cond['wind_kn']} عقدة")
    if cond.get("wave_max") is not None:
        sea.append(f"موج حتى {cond['wave_max']} م")
    if sea:
        lines.append("البحر: " + " · ".join(sea))
    lines += ["", f"مصدر: أقمار صناعية (حرارة {sst_day}{', كلوروفيل' if r and r['cover'] > 0.15 else ''}). المؤشر يدل على أنسب ظروف الماء، ولا يضمن وجود السمك."]
    return "\n".join(lines)


def draw(p, r, path, day, demo):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(6.4, 7.2), dpi=120, facecolor="#06121d")
    ext = [r["glon"].min(), r["glon"].max(), r["glat"].min(), r["glat"].max()]
    asp = 1 / math.cos(math.radians(p["lat"]))
    ax.set_facecolor("#071a2c")
    cm = LinearSegmentedColormap.from_list("s", ["#071a2c", "#0f3a55", "#136b6f", "#d39a3a", "#ffb547", "#ff6b4a"])
    from scipy.ndimage import gaussian_filter as _g
    S = np.ma.masked_where(_g(r["ocean"].astype(float), 1.2) < 0.5, r["S"] / (r["S"].max() or 1))
    ax.imshow(S, extent=ext, origin="upper", cmap=cm, vmin=0, vmax=1, interpolation="bilinear", aspect=asp)
    sst = np.where(r["ocean"], r["sst"], np.nan)
    try:
        cs = ax.contour(r["glon"], r["glat"], sst, levels=8, colors="white", linewidths=0.5, alpha=0.45)
        ax.clabel(cs, fontsize=6, fmt="%.1f°")
    except Exception:
        pass
    ax.contour(r["glon"], r["glat"], r["dist"], levels=[MAX_KM], colors="#4fe3d6", linewidths=0.8, linestyles="--")
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(r["ocean"].astype(float), 1.2)
    ax.contourf(r["glon"], r["glat"], sm, levels=[-1, 0.5], colors=["#2a2418"], zorder=3)
    ax.contour(r["glon"], r["glat"], sm, levels=[0.5], colors="#e6d2aa", linewidths=1.2, zorder=4)
    th = np.linspace(0, 2 * math.pi, 200)
    ax.plot(p["lon"] + RANGE_KM / (111.2 * math.cos(math.radians(p["lat"]))) * np.sin(th), p["lat"] + RANGE_KM / 111.2 * np.cos(th),
            color="#8aa3b4", lw=0.7, ls=":")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.plot(p["lon"], p["lat"], marker="*", ms=16, color="#4fe3d6", mec="#06121d", zorder=6)
    ax.text(p["lon"] + 0.01, p["lat"] + 0.008, p["fr"], color="#4fe3d6", fontsize=10, weight="bold", zorder=6)
    for z in r["zones"]:
        ax.plot([p["lon"], z["lon"]], [p["lat"], z["lat"]], ":", color="white", lw=0.8)
        ax.plot(z["lon"], z["lat"], "o", ms=18, mfc="#06121d", mec="white", mew=2, zorder=7)
        ax.text(z["lon"], z["lat"], z["id"], color="white", ha="center", va="center", fontsize=10, weight="bold", zorder=8)
        ax.text(z["lon"] + 0.012, z["lat"] - 0.004, f"{z['km']} km", color="white", fontsize=8,
                bbox=dict(fc="#06121d", ec="none", alpha=0.7, pad=1.5), zorder=8)
    # 2 km scale bar
    x0, y0 = ext[0] + 0.01, ext[2] + 0.012
    dlon = 2 / (111.2 * math.cos(math.radians(p["lat"])))
    ax.plot([x0, x0 + dlon], [y0, y0], color="white", lw=3)
    ax.text(x0 + dlon / 2, y0 + 0.006, "2 km", color="white", ha="center", fontsize=8)
    ax.tick_params(colors="#8aa3b4", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#1c3448")
    src = "DEMO DATA" if demo else "NASA MUR SST · Copernicus Marine"
    ax.set_title(f"{p['fr']} · {day}\n{src}", color="#e6eef3", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def page(items, day, demo):
    tabs = "".join(f'<button onclick="show(\'{i["p"]["id"]}\')" id="b-{i["p"]["id"]}">{html.escape(i["p"]["ar"])}</button>' for i in items)
    secs = "".join(f'''<section id="s-{i["p"]["id"]}" hidden><img src="{i["p"]["id"]}.png?v={day}" alt="خريطة {html.escape(i["p"]["ar"])}">
<pre id="m-{i["p"]["id"]}">{html.escape(i["msg"])}</pre>
<button class="copy" onclick="navigator.clipboard.writeText(document.getElementById('m-{i["p"]["id"]}').innerText).then(()=>this.textContent='تم النسخ')">نسخ الرسالة</button></section>''' for i in items)
    warn = "<p class='w'>بيانات تجريبية (demo).</p>" if demo else ""
    first = items[0]["p"]["id"] if items else ""
    doc = f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>رادار الساحل · الداخلة</title><style>body{{margin:0;background:#06121d;color:#e6eef3;font-family:system-ui,sans-serif;padding:16px}}
main{{max-width:760px;margin:auto}}nav{{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}}nav button{{border:1px solid #1c3448;background:#0b1c2b;color:#e6eef3;border-radius:999px;padding:10px 16px;font-size:15px}}
nav button.on{{background:#4fe3d6;color:#06121d;font-weight:700}}img{{width:100%;border-radius:12px;border:1px solid #1c3448}}
pre{{white-space:pre-wrap;background:#0a2a22;border:1px solid #1f5b49;border-radius:12px;padding:12px;font:15px/1.8 system-ui}}
.copy{{background:#4fe3d6;border:0;border-radius:999px;padding:12px 18px;font-weight:700;font-size:15px}}.w{{color:#ffb547}}.s{{color:#8aa3b4;font-size:13px;line-height:1.7}}</style></head>
<body><main><h1>رادار الساحل · الداخلة</h1><p class="s">قوارب الصيد التقليدي · {html.escape(SPECIES)} · {day}</p>{warn}
<nav>{tabs}</nav>{secs}
<p class="s">مؤشر يومي من الأقمار الصناعية: جبهات حرارة الماء (NASA MUR، 1 كم)، والكلوروفيل (Copernicus Marine)، وصعود المياه الباردة من الرياح، على شريط حتى {MAX_KM} كم من الشاطئ.
يدل على أنسب ظروف الماء للأسماك التي تتبع الكتل المائية، ولا يضمن وجود السمك. المد وحالة البحر تقريبيان ولا يصلحان للملاحة.</p></main>
<script>function show(id){{document.querySelectorAll('section').forEach(s=>s.hidden=s.id!=='s-'+id);document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.id==='b-'+id));try{{localStorage.setItem('pt',id)}}catch(e){{}}}}
let p='{first}';try{{p=localStorage.getItem('pt')||p}}catch(e){{}}if(!document.getElementById('s-'+p))p='{first}';show(p)</script></body></html>"""
    (OUT / "index.html").write_text(doc, encoding="utf-8")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--date")
    a = ap.parse_args()
    day = a.date or dt.date.today().isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.demo:
        lats, lons, sst, chl = demo_all(day)
        sst_all, chl_all, sst_day = (lats, lons, sst), (lats, lons, chl), day
    else:
        try:
            lats, lons, sst, sst_day = mur_sst()
        except Exception as e:  # noqa: BLE001
            print("[warn] MUR via ERDDAP failed, using OSTIA:", e)
            lats, lons, sst, sst_day = ostia_fallback(dt.date.fromisoformat(day))
        sst_all = (lats, lons, sst)
        ch = chlorophyll(dt.date.fromisoformat(day))
        chl_all = None
        if ch is not None:
            da, la, lo, _ = ch
            chl_all = (da[la].values, da[lo].values, da.values.astype("float32"))

    items, out = [], []
    for p in POINTS:
        try:
            cond = demo_conditions(p, day) if a.demo else conditions(p)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] conditions for {p['id']}: {e}")
            cond = {"upwelling": 0, "tides": [], "wind_kn": None, "wave_max": None}
        r = score_point(p, sst_all, chl_all, cond)
        if r is None:
            print(f"[skip] {p['id']}: no ocean pixels")
            continue
        msg = message(p, r, cond, day, sst_day)
        draw(p, r, OUT / f"{p['id']}.png", day, a.demo)
        items.append({"p": p, "msg": msg})
        out.append({"point": p, "zones": r["zones"], "chl_cover": r["cover"], "conditions": cond, "msg": msg, "grid": grid_json(r)})
        print(msg, "\n")
    data = {"day": day, "sst_day": sst_day, "demo": a.demo, "made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "max_km": MAX_KM, "range_km": RANGE_KM, "species": C.get("species", ["courbine", "tassergal", "mulet", "bonite"]), "points": out}
    console(data)
    (OUT / "coastal.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT / "messages.txt").write_text("\n\n————————\n\n".join(i["msg"] for i in items), encoding="utf-8")


if __name__ == "__main__":
    main()
