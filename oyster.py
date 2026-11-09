"""
Madar Oysters (مدار المحار) - daily water watch for the shellfish zones of Dakhla Bay.

What it gives every morning, per shellfish zone:
  * sea-surface temperature, today and the last 30 days (NASA MUR, 1 km; Copernicus OSTIA as backup)
  * chlorophyll (food for oysters, and the first sign of an algal bloom), today and the last 30 days
    (Copernicus ocean colour, 1 km gap-free)
  * the newest clear Sentinel-2 picture of the whole bay at 20 m: true colour, a chlorophyll index (NDCI)
    and turbidity, so a bloom or a muddy plume can be seen where it is
  * wind for the next 48 h (work on the parcs, and upwelling that pushes cold rich water into the bay)
  * a traffic light with the reasons in plain words.

Satellites do NOT measure toxins. The signal is an early warning to sample more often; harvest and sale
decisions follow only the official INRH / Department of Maritime Fisheries bulletins.

  python oyster.py           # real data (Copernicus account in env, see README)
  python oyster.py --demo    # synthetic data, to test the page without any account
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

import radar  # http_get, Copernicus helpers, config

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "oyster"
CFG = radar.CFG.get("oyster", {})

# Bay of Dakhla: 37 x 13.5 km, centre 23d45'54"N 15d49'58"W (Ramsar sheet), open to the ocean in the south-west.
BOX = CFG.get("box", [23.56, 23.98, -15.99, -15.66])          # lat0, lat1, lon0, lon1
RES = 0.0002                                                    # ~20 m
# Shellfish zones named in the official closure notices (Aug 2025). Positions are APPROXIMATE:
# farmers correct them in config.json -> oyster.zones.
ZONES = CFG.get("zones") or [
    {"id": "lassarga", "ar": "لاسارگا", "fr": "La Sarga", "lat": 23.640, "lon": -15.915, "approx": True},
    {"id": "dunablanca", "ar": "دونا بلانكا", "fr": "Duna Blanca", "lat": 23.650, "lon": -15.850, "approx": True},
    {"id": "centre", "ar": "وسط الخليج", "fr": "Centre de la baie", "lat": 23.735, "lon": -15.860, "approx": True},
    {"id": "boutalha", "ar": "بوطلحة", "fr": "Boutalha", "lat": 23.840, "lon": -15.795, "approx": True},
    {"id": "tiniguir", "ar": "تنيكير", "fr": "Tiniguir", "lat": 23.905, "lon": -15.745, "approx": True},
]
ZONE_KM = 2.5            # radius used around each zone for the Sentinel-2 statistics
DAYS = 30

CHL_SERIES = CFG.get("chl_datasets") or [
    ["cmems_obs-oc_atl_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D", "CHL"],
    ["cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"],
]
SST_BACKUP = radar.SST_CANDIDATES
# Multi-year ocean colour, for "what is normal at this time of year" (same 3 weeks, previous years).
CHL_CLIM = CFG.get("chl_clim_datasets") or [
    ["cmems_obs-oc_atl_bgc-plankton_my_l4-gapfree-multi-1km_P1D", "CHL"],
    ["cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D", "CHL"],
]
CLIM_YEARS = 5

# Thresholds (Pacific oyster, Crassostrea gigas). Editable in config.json -> oyster.limits.
LIM = {"sst_warm": 24.0, "sst_hot": 26.0, "sst_drop": 2.0,
       "chl_ratio_watch": 1.6, "chl_ratio_alert": 2.5, "chl_high": 5.0, "chl_alert_min": 3.0,
       "ndci_watch": 0.10, "ndci_alert": 0.20, "turb_watch": 20.0, "turb_mask": 25.0, "wind_work_kn": 25,
       "s2_fresh_days": 5}
LIM.update(CFG.get("limits", {}))


# ------------------------------------------------------------------ helpers
def km(la1, lo1, la2, lo2):
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_valid(lats, lons, arr2d, lat, lon, max_km=4.0):
    """Value of the closest valid pixel to (lat, lon) within max_km, else None."""
    la = np.asarray(lats)[:, None]
    lo = np.asarray(lons)[None, :]
    d = np.hypot((la - lat) * 111.0, (lo - lon) * 111.0 * math.cos(math.radians(lat)))
    d = np.where(np.isfinite(arr2d), d, np.inf)
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return float(arr2d[j, i]) if d[j, i] <= max_km else None


def rnd(v, k=2):
    return None if v is None or not np.isfinite(v) else round(float(v), k)


# ------------------------------------------------------------------ real data
def sst_series(day):
    """30 days of SST (deg C) per zone. NASA MUR through ERDDAP, OSTIA (Copernicus) as backup."""
    lat0, lat1, lon0, lon1 = BOX
    for base in ["https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41",
                 "https://upwell.pfeg.noaa.gov/erddap/griddap/jplMURSST41"]:
        try:
            q = (f".json?analysed_sst[last-{DAYS + 4}:1:last][({lat0}):1:({lat1})][({lon0}):1:({lon1})]")
            js = json.loads(radar.http_get(base + q, timeout=240).decode())
            rows = js["table"]["rows"]
            times = sorted({r[0][:10] for r in rows})
            lats = sorted({r[1] for r in rows})
            lons = sorted({r[2] for r in rows})
            ti = {t: k for k, t in enumerate(times)}
            yi = {v: k for k, v in enumerate(lats)}
            xi = {v: k for k, v in enumerate(lons)}
            cube = np.full((len(times), len(lats), len(lons)), np.nan, "float32")
            for t, y, x, v in rows:
                if v is not None:
                    cube[ti[t[:10]], yi[y], xi[x]] = v
            if np.nanmean(cube) > 200:
                cube -= 273.15
            print(f"[ok] SST series: NASA MUR {times[0]} .. {times[-1]}")
            return times, np.array(lats), np.array(lons), cube, "NASA MUR 1 km"
        except Exception as e:  # noqa: BLE001
            print("[skip] MUR series:", base, e)
    times, lats, lons, cube = cm_series(SST_BACKUP, day)
    if np.nanmean(cube) > 200:
        cube = cube - 273.15
    return times, lats, lons, cube, "Copernicus OSTIA"


def cm_series(candidates, day):
    """Daily cube (time, lat, lon) for the bay box from the first Copernicus dataset that answers."""
    import copernicusmarine as cm

    lat0, lat1, lon0, lon1 = BOX
    pad = 0.05
    errors = []
    for dataset_id, var in candidates:
        try:
            ds = cm.open_dataset(dataset_id=dataset_id, variables=[var],
                                 minimum_longitude=lon0 - pad, maximum_longitude=lon1 + pad,
                                 minimum_latitude=lat0 - pad, maximum_latitude=lat1 + pad,
                                 start_datetime=f"{day - dt.timedelta(days=DAYS + 8)}T00:00:00",
                                 end_datetime=f"{day}T23:59:59")
            da = ds[var]
            if "depth" in da.dims:
                da = da.isel(depth=0)
            la = "latitude" if "latitude" in da.coords else "lat"
            lo = "longitude" if "longitude" in da.coords else "lon"
            da = da.sortby(la)
            times = [str(np.datetime_as_string(t, unit="D")) for t in da["time"].values]
            cube = da.values.astype("float32")
            print(f"[ok] {var}: {dataset_id} {times[0]} .. {times[-1]}")
            return times, da[la].values, da[lo].values, cube
        except Exception as e:  # noqa: BLE001
            errors.append(f"{dataset_id}: {e}")
            print("[skip]", dataset_id, e)
    raise RuntimeError("; ".join(errors))


def chl_climatology(day):
    """Median chlorophyll per zone for the same 3 weeks of the previous years (seasonal normal)."""
    import copernicusmarine as cm

    lat0, lat1, lon0, lon1 = BOX
    pad = 0.05
    for dataset_id, var in CHL_CLIM:
        vals = {z["id"]: [] for z in ZONES}
        years = 0
        for y in range(1, CLIM_YEARS + 3):
            try:
                c = day.replace(year=day.year - y)
            except ValueError:
                c = day.replace(year=day.year - y, day=28)
            try:
                ds = cm.open_dataset(dataset_id=dataset_id, variables=[var],
                                     minimum_longitude=lon0 - pad, maximum_longitude=lon1 + pad,
                                     minimum_latitude=lat0 - pad, maximum_latitude=lat1 + pad,
                                     start_datetime=f"{c - dt.timedelta(days=10)}T00:00:00",
                                     end_datetime=f"{c + dt.timedelta(days=10)}T23:59:59")
            except Exception as e:  # noqa: BLE001
                print("[skip] clim", dataset_id, c.year, e)
                continue
            da = ds[var]
            la = "latitude" if "latitude" in da.coords else "lat"
            lo = "longitude" if "longitude" in da.coords else "lon"
            da = da.sortby(la)
            cube = da.values.astype("float32")
            for z in ZONES:
                for k in range(cube.shape[0]):
                    v = nearest_valid(da[la].values, da[lo].values, cube[k], z["lat"], z["lon"])
                    if v is not None:
                        vals[z["id"]].append(v)
            years += 1
            if years >= CLIM_YEARS:
                break
        if years >= 2:
            print(f"[ok] chlorophyll normal: {dataset_id}, {years} years")
            return {k: (rnd(float(np.median(v))) if len(v) >= 10 else None) for k, v in vals.items()}, years, dataset_id
    raise RuntimeError("no multi-year chlorophyll")


def zone_series(times, lats, lons, cube):
    """{zone_id: [values per day]} using the nearest valid pixel of each day."""
    out = {}
    for z in ZONES:
        out[z["id"]] = [rnd(nearest_valid(lats, lons, cube[k], z["lat"], z["lon"])) for k in range(len(times))]
    return out


def sentinel2(day, max_days=20, max_cloud=40):
    """Newest Sentinel-2 L2A pass with clear water over the bay. Returns bands dict, date, scene ids."""
    import rasterio
    from pystac_client import Client
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    lat0, lat1, lon0, lon1 = BOX
    w, h = int(round((lon1 - lon0) / RES)), int(round((lat1 - lat0) / RES))
    tr = from_bounds(lon0, lat0, lon1, lat1, w, h)
    items = list(Client.open("https://earth-search.aws.element84.com/v1").search(
        collections=["sentinel-2-l2a"], bbox=[lon0, lat0, lon1, lat1],
        datetime=f"{day - dt.timedelta(days=max_days)}/{day}",
        query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=60).items())
    by_day = {}
    for it in items:
        by_day.setdefault(it.datetime.date(), []).append(it)
    bands = ["blue", "green", "red", "rededge1", "nir", "scl"]
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                       GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")
    for d in sorted(by_day, reverse=True):          # newest pass first
        out = {b: np.full((h, w), np.nan, "float32") for b in bands}
        ids = []
        with env:
            for it in by_day[d]:
                off = 0.0
                if str(it.properties.get("s2:processing_baseline", "00.00")) >= "04.00" and \
                        not it.properties.get("earthsearch:boa_offset_applied", False):
                    off = -1000.0
                try:
                    tile = {}
                    for b in bands:
                        with rasterio.open(it.assets[b].href) as src, WarpedVRT(
                                src, crs="EPSG:4326", transform=tr, width=w, height=h, nodata=0,
                                resampling=Resampling.nearest if b == "scl" else Resampling.bilinear) as v:
                            a = v.read(1).astype("float32")
                        a[a == 0] = np.nan
                        tile[b] = a if b == "scl" else (a + off) / 10000.0
                except Exception as e:  # noqa: BLE001
                    print("[skip]", it.id, e)
                    continue
                fill = np.isnan(out["red"]) & np.isfinite(tile["red"])
                for b in bands:
                    out[b][fill] = tile[b][fill]
                ids.append(it.id)
        water = out["scl"] == 6
        clear_water = water.mean()
        print(f"[s2] {d}: clear water {100 * clear_water:.1f}% of the box")
        if clear_water > 0.25:
            return out, d.isoformat(), ids
    raise RuntimeError("no clear Sentinel-2 pass over the bay in the last %d days" % max_days)


def wind_48h():
    lat, lon = 23.765, -15.833
    js = json.loads(radar.http_get(
        "https://api.open-meteo.com/v1/forecast?latitude=%.3f&longitude=%.3f&hourly=wind_speed_10m,wind_direction_10m"
        "&wind_speed_unit=kn&forecast_days=3&timezone=Africa%%2FCasablanca" % (lat, lon), timeout=60).decode())
    h = js["hourly"]
    now = dt.datetime.now().strftime("%Y-%m-%dT%H")
    i0 = next((i for i, t in enumerate(h["time"]) if t >= now), 0)
    sp = [v for v in h["wind_speed_10m"][i0:i0 + 48] if v is not None]
    di = [v for v in h["wind_direction_10m"][i0:i0 + 48] if v is not None]
    return {"t0": h["time"][i0], "speed": [round(v) for v in h["wind_speed_10m"][i0:i0 + 48]],
            "dir": [None if v is None else round(v) for v in h["wind_direction_10m"][i0:i0 + 48]],
            "now_kn": round(sp[0]) if sp else None, "max_kn": round(max(sp)) if sp else None,
            "dir_now": round(di[0]) if di else None}


# ------------------------------------------------------------------ demo data
def bay_mask(h, w):
    """Synthetic bay: an ellipse 37 x 13.5 km on a SW-NE axis, open to the ocean in the SW."""
    lat0, lat1, lon0, lon1 = BOX
    yy, xx = np.mgrid[0:h, 0:w]
    lat = lat1 - (yy + 0.5) * RES
    lon = lon0 + (xx + 0.5) * RES
    cy, cx, ang = 23.765, -15.833, math.radians(30)
    dy = (lat - cy) * 111.0
    dx = (lon - cx) * 111.0 * math.cos(math.radians(cy))
    u = dx * math.sin(ang) + dy * math.cos(ang)      # along the axis (NE +)
    v = dx * math.cos(ang) - dy * math.sin(ang)      # across
    bay = (u / 18.5) ** 2 + (v / 6.7) ** 2 < 1
    ocean = (v < -9.5) | ((u < -15) & (v < 2))       # ocean west of the dune ridge and at the mouth
    return bay | ocean, bay, u, v


def demo(day):
    rng = np.random.default_rng(day.toordinal())
    lat0, lat1, lon0, lon1 = BOX
    w, h = int(round((lon1 - lon0) / RES)), int(round((lat1 - lat0) / RES))
    water, bay, u, v = bay_mask(h, w)
    bloom = np.exp(-(((u - 6) / 4.0) ** 2 + ((v - 1) / 2.5) ** 2))       # a patch near Boutalha
    land = 0.28 + 0.04 * rng.standard_normal((h, w)).astype("float32")
    b = {}
    b["red"] = np.where(water, 0.018 + 0.02 * np.clip(-u / 18, 0, 1) - 0.006 * bloom, land * 1.15)
    b["green"] = np.where(water, 0.035 + 0.015 * bloom, land)
    b["blue"] = np.where(water, 0.045, land * 0.8)
    b["rededge1"] = np.where(water, b["red"] * (1.0 + 0.6 * bloom) + 0.002, land * 1.2)
    b["nir"] = np.where(water, 0.008, land * 1.3)
    b["scl"] = np.where(water, 6.0, 5.0).astype("float32")
    for k in b:
        b[k] = b[k].astype("float32")
    days = [(day - dt.timedelta(days=DAYS - 1 - k)).isoformat() for k in range(DAYS)]
    sst, chl = {}, {}
    for n, z in enumerate(ZONES):
        base = 19.0 + 0.4 * n
        s = base + 0.8 * np.sin(np.arange(DAYS) / 5.0 + n) + 0.15 * rng.standard_normal(DAYS)
        s[-3:] -= [0.8, 1.6, 2.3] if z["id"] in ("boutalha", "tiniguir") else [0, 0, 0]
        c = 1.6 + 0.4 * rng.standard_normal(DAYS).clip(-2, 2) * 0.5
        if z["id"] == "boutalha":
            c[-4:] = [2.5, 3.6, 4.8, 5.6]
        sst[z["id"]] = [round(float(x), 2) for x in s]
        chl[z["id"]] = [round(float(max(0.2, x)), 2) for x in c]
    hours = 48
    wind = {"t0": dt.datetime.now().strftime("%Y-%m-%dT%H:00"),
            "speed": [round(14 + 8 * math.sin(k / 7)) for k in range(hours)],
            "dir": [20 for _ in range(hours)], "now_kn": 14, "max_kn": 22, "dir_now": 20}
    return b, (day - dt.timedelta(days=2)).isoformat(), ["DEMO"], days, sst, days, chl, wind


# ------------------------------------------------------------------ analysis
def s2_products(b):
    water = b["scl"] == 6
    red, re1 = b["red"], b["rededge1"]
    ndci = (re1 - red) / (re1 + red + 1e-6)                 # Mishra & Mishra 2012
    rho = np.clip(red, 0, 0.15)
    turb = 228.1 * rho / (1 - rho / 0.1641)                  # Dogliotti et al. 2015, red band, FNU (approx.)
    ndci[~water] = np.nan
    turb[~water] = np.nan
    return water, ndci.astype("float32"), turb.astype("float32")


def zone_s2(ndci, turb):
    lat0, lat1, lon0, lon1 = BOX
    h, w = ndci.shape
    lat = lat1 - (np.arange(h) + 0.5) * RES
    lon = lon0 + (np.arange(w) + 0.5) * RES
    res = {}
    for z in ZONES:
        dy = (lat[:, None] - z["lat"]) * 111.0
        dx = (lon[None, :] - z["lon"]) * 111.0 * math.cos(math.radians(z["lat"]))
        m = (dy ** 2 + dx ** 2) <= ZONE_KM ** 2
        n = ndci[m]
        t = turb[m]
        n = n[np.isfinite(n)]
        t = t[np.isfinite(t)]
        res[z["id"]] = {"ndci": rnd(np.median(n), 3) if n.size > 30 else None,
                        "ndci_p90": rnd(np.percentile(n, 90), 3) if n.size > 30 else None,
                        "turb": rnd(np.median(t), 1) if t.size > 30 else None,
                        "px": int(n.size)}
    return res


def last(vals):
    for v in reversed(vals):
        if v is not None:
            return v
    return None


def judge(z, sst, chl, s2, s2_age, wind, clim=None):
    """Traffic light for one zone with the reasons (Arabic / French)."""
    R = []

    def add(level, key, ar, fr):
        R.append({"level": level, "key": key, "ar": ar, "fr": fr})

    s_now = last(sst)
    if s_now is not None:
        if s_now >= LIM["sst_hot"]:
            add(2, "hot", f"ماء حار جداً {s_now:.1f}°: إجهاد حراري ونفوق محتمل", f"Eau très chaude {s_now:.1f}° : stress thermique, risque de mortalité")
        elif s_now >= LIM["sst_warm"]:
            add(1, "warm", f"ماء دافئ {s_now:.1f}°: راقب النفوق والأمراض", f"Eau chaude {s_now:.1f}° : surveiller mortalités et maladies")
        prev = [v for v in sst[-6:-1] if v is not None]
        if prev and s_now - max(prev) <= -LIM["sst_drop"]:
            add(1, "cold", f"برودة مفاجئة ({s_now - max(prev):+.1f}° في أيام): مياه عميقة غنية دخلت الخليج، وقد يتبعها ازدهار طحالب",
                f"Refroidissement brutal ({s_now - max(prev):+.1f}°) : remontée d'eau profonde riche, une efflorescence peut suivre")
    c_now = last(chl)
    hist = [v for v in chl[:-3] if v is not None]
    ratio = None
    base_ar, base_fr = "", ""
    if c_now is not None and clim:
        ratio = c_now / clim if clim > 0 else None
        base_ar, base_fr = "عن المعتاد في هذا الموسم", "par rapport à la normale de saison"
    elif c_now is not None and len(hist) >= 5:
        med = float(np.median(hist))
        ratio = c_now / med if med > 0 else None
        base_ar, base_fr = "عن الأيام الأخيرة", "par rapport aux derniers jours"
    if ratio:
        if ratio >= LIM["chl_ratio_alert"] and c_now >= LIM["chl_alert_min"]:
            add(2, "bloom", f"الكلوروفيل ×{ratio:.1f} {base_ar} ({c_now:.1f} mg/m³): ازدهار طحالب قوي محتمل. زيدوا أخذ العينات وتابعوا نشرات INRH",
                f"Chlorophylle ×{ratio:.1f} {base_fr} ({c_now:.1f} mg/m³) : forte efflorescence possible. Renforcer les prélèvements, suivre les bulletins INRH")
        elif ratio >= LIM["chl_ratio_watch"]:
            add(1, "rise", f"الكلوروفيل أعلى ×{ratio:.1f} {base_ar} ({c_now:.1f} mg/m³)",
                f"Chlorophylle ×{ratio:.1f} {base_fr} ({c_now:.1f} mg/m³)")
    if s2 and s2_age is not None and s2_age <= LIM["s2_fresh_days"]:
        nd, tb = s2.get("ndci_p90"), s2.get("turb")
        if tb is not None and tb >= LIM["turb_mask"]:
            nd = None          # sediment or a visible sandy bottom also turns the index up: not algae
        if nd is not None and nd >= LIM["ndci_alert"]:
            add(2, "s2bloom", "صورة Sentinel-2 تُظهر بقعة طحالب كثيفة في المنطقة", "L'image Sentinel-2 montre une tache d'algues dense dans la zone")
        elif nd is not None and nd >= LIM["ndci_watch"]:
            add(1, "s2green", "صورة Sentinel-2 تُظهر ماء أخضر أكثر من العادة", "L'image Sentinel-2 montre une eau plus verte que d'habitude")
        if tb is not None and tb >= LIM["turb_watch"]:
            add(1, "turb", f"ماء عكر (~{tb:.0f} FNU): غذاء أقل جودة ورواسب", f"Eau trouble (~{tb:.0f} FNU) : sédiments, alimentation moins bonne")
    R.sort(key=lambda r: -r["level"])
    level = max([r["level"] for r in R], default=0)
    good = (c_now is not None and c_now >= 1.0 and s_now is not None and 16 <= s_now < LIM["sst_warm"])
    if level == 0:
        if good:
            add(0, "good", "ظروف نمو جيدة: غذاء كافٍ وحرارة مناسبة", "Bonnes conditions de croissance : nourriture suffisante, température adaptée")
        else:
            add(0, "calm", "لا إشارة غير عادية اليوم", "Aucun signal inhabituel aujourd'hui")
    return level, R, ratio


def save_png(arr, path, max_w=1400):
    from PIL import Image

    im = Image.fromarray(arr)
    if im.mode == "RGBA":
        max_w = 1000          # colour overlays: lighter for phones
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    if str(path).endswith(".jpg"):
        im.save(path, quality=84, optimize=True, progressive=True)
    elif im.mode == "RGBA":
        im.quantize(colors=64, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)
    else:
        im.save(path, optimize=True)
    return im.size


def images(b, water, ndci, turb):
    def stretch(a, lo, hi):
        return np.clip((a - lo) / (hi - lo), 0, 1)

    rgb = np.dstack([stretch(b["red"], 0.0, 0.30), stretch(b["green"], 0.0, 0.28), stretch(b["blue"], 0.0, 0.26)])
    rgb = np.power(np.nan_to_num(rgb), 1 / 1.6)
    true = (rgb * 255).astype("uint8")
    size = save_png(true, OUT / "s2_true.jpg")

    def overlay(val, stops):
        v = np.nan_to_num(val, nan=-9)
        xs = [s[0] for s in stops]
        out = np.zeros(val.shape + (4,), "uint8")
        for c in range(3):
            out[..., c] = np.interp(v, xs, [s[1][c] for s in stops]).astype("uint8")
        out[..., 3] = np.where(np.isfinite(val), 215, 0).astype("uint8")
        return out

    save_png(overlay(ndci, [(-0.15, (40, 70, 160)), (-0.05, (60, 150, 190)), (0.05, (120, 200, 150)),
                            (0.12, (240, 210, 80)), (0.22, (220, 90, 50)), (0.35, (150, 20, 40))]), OUT / "s2_chl.png")
    save_png(overlay(turb, [(0, (30, 60, 120)), (5, (70, 140, 180)), (12, (190, 190, 140)),
                            (25, (190, 130, 70)), (50, (120, 70, 40))]), OUT / "s2_turb.png")
    return size


# ------------------------------------------------------------------ outputs
def message(data):
    L = {0: "🟢", 1: "🟡", 2: "🔴"}
    ar = [f"🦪 مدار المحار · خليج الداخلة · {data['day']}", ""]
    fr = [f"🦪 Madar Huîtres · Baie de Dakhla · {data['day']}", ""]
    for z in data["zones"]:
        s = f"{z['sst']:.1f}°" if z["sst"] is not None else "–"
        c = f"{z['chl']:.1f}" if z["chl"] is not None else "–"
        ar.append(f"{L[z['level']]} {z['ar']}: {s} · كلوروفيل {c} — {z['reasons'][0]['ar']}")
        fr.append(f"{L[z['level']]} {z['fr']} : {s} · chl {c} — {z['reasons'][0]['fr']}")
    w = data.get("wind") or {}
    if w.get("max_kn") is not None:
        ar += ["", f"💨 الرياح 48 ساعة: حتى {w['max_kn']} عقدة"]
        fr += ["", f"💨 Vent 48 h : jusqu'à {w['max_kn']} nœuds"]
    ar += ["", "⚠️ الأقمار لا تقيس السموم. القرار الرسمي للجني والبيع: نشرات INRH وكتابة الدولة المكلفة بالصيد البحري."]
    fr += ["", "⚠️ Les satellites ne mesurent pas les toxines. Décision officielle de récolte et de vente : bulletins INRH / Pêche maritime."]
    return "\n".join(ar), "\n".join(fr)


def build_page(data):
    tpl = (ROOT / "oyster_console.html").read_text(encoding="utf-8")
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (OUT / "index.html").write_text(tpl.replace("/*__DATA__*/null", js), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--date")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
    OUT.mkdir(parents=True, exist_ok=True)
    src = {}
    if a.demo:
        b, s2_day, s2_ids, sdays, sst, cdays, chl, wind = demo(day)
        clim = {z["id"]: 1.6 for z in ZONES}
        src = {"sst": "DEMO", "chl": "DEMO", "s2": "DEMO", "wind": "DEMO"}
    else:
        b = s2_day = None
        s2_ids = []
        try:
            b, s2_day, s2_ids = sentinel2(day)
            src["s2"] = "Sentinel-2 L2A (ESA Copernicus, Earth Search)"
        except Exception as e:  # noqa: BLE001
            print("[warn] Sentinel-2:", e)
        try:
            t, la, lo, cube, name = sst_series(day)
            sdays, sst = t, zone_series(t, la, lo, cube)
            src["sst"] = name
        except Exception as e:  # noqa: BLE001
            print("[warn] SST:", e)
            sdays, sst = [], {z["id"]: [] for z in ZONES}
        try:
            t, la, lo, cube = cm_series(CHL_SERIES, day)
            cdays, chl = t, zone_series(t, la, lo, cube)
            src["chl"] = "Copernicus Marine ocean colour (L4, 1 km)"
        except Exception as e:  # noqa: BLE001
            print("[warn] CHL:", e)
            cdays, chl = [], {z["id"]: [] for z in ZONES}
        try:
            clim, clim_years, clim_src = chl_climatology(day)
            src["clim"] = f"{clim_src} ({clim_years} ans / سنوات)"
        except Exception as e:  # noqa: BLE001
            print("[warn] chlorophyll normal:", e)
            clim = {}
        try:
            wind = wind_48h()
            src["wind"] = "Open-Meteo"
        except Exception as e:  # noqa: BLE001
            print("[warn] wind:", e)
            wind = None

    s2z, size = {}, None
    s2_age = None
    if b is not None:
        water, ndci, turb = s2_products(b)
        s2z = zone_s2(ndci, turb)
        size = images(b, water, ndci, turb)
        s2_age = (day - dt.date.fromisoformat(s2_day)).days

    # keep the last 30 days
    sdays_k, cdays_k = sdays[-DAYS:], cdays[-DAYS:]
    zones = []
    for z in ZONES:
        zs = sst[z["id"]][-DAYS:]
        zc = chl[z["id"]][-DAYS:]
        level, reasons, ratio = judge(z, zs, zc, s2z.get(z["id"]), s2_age, wind, clim.get(z["id"]))
        zones.append({**z, "level": level, "reasons": reasons, "sst": last(zs), "chl": last(zc), "chl_normal": clim.get(z["id"]),
                      "chl_ratio": rnd(ratio), "s2": s2z.get(z["id"]), "sst_series": zs, "chl_series": zc})
    data = {
        "day": day.isoformat(), "made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "demo": a.demo,
        "box": BOX, "zone_km": ZONE_KM, "limits": LIM,
        "sst_days": sdays_k, "chl_days": cdays_k,
        "sst_day": sdays_k[-1] if sdays_k else None, "chl_day": cdays_k[-1] if cdays_k else None,
        "s2": {"day": s2_day, "age": s2_age, "ids": s2_ids, "size": size} if s2_day else None,
        "wind": wind, "sources": src, "zones": zones,
        "level": max([z["level"] for z in zones], default=0),
    }
    data["msg_ar"], data["msg_fr"] = message(data)
    (OUT / "oyster.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "message.txt").write_text(data["msg_ar"] + "\n\n" + data["msg_fr"], encoding="utf-8")
    build_page(data)
    print(data["msg_ar"])


if __name__ == "__main__":
    main()
