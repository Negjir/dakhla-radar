"""
Dakhla Fish Radar - daily potential fishing zones (PFZ) for pelagic fish.

Steps:
  1. Download yesterday's sea surface temperature (SST) and chlorophyll (CHL)
     for the Dakhla box from Copernicus Marine.
  2. Detect thermal fronts (where cold upwelled water meets warmer water).
  3. Score each pixel: front strength + chlorophyll suitability + distance
     from the coast + captains' recent catch reports.
  4. Pick the 5 best zones outside closed areas, with distance, bearing and
     fuel from the port.
  5. Write docs/map.png, docs/message.txt, docs/zones.json, docs/index.html.

Usage:
  python radar.py                 # today's map (uses data of the last available day)
  python radar.py --date 2026-09-15   # map for a past day (for back-testing)
  python radar.py --demo          # synthetic data, no account needed (to test the setup)
  python radar.py --check         # list the Copernicus datasets this script can use

Credentials: environment variables COPERNICUSMARINE_SERVICE_USERNAME and
COPERNICUSMARINE_SERVICE_PASSWORD (set as GitHub secrets).
"""
import argparse
import csv
import datetime as dt
import html
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs"
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

# Candidate datasets, tried in order. Copernicus renames datasets from time to
# time; if all fail, run `python radar.py --check` and put the new id in config.json.
SST_CANDIDATES = CFG.get("sst_datasets") or [
    ["METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2", "analysed_sst"],
]
CHL_CANDIDATES = CFG.get("chl_datasets") or [
    ["cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"],
    ["cmems_obs-oc_atl_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D", "CHL"],
]


# ----------------------------------------------------------------- data access
UA = {"User-Agent": "Mozilla/5.0 (dakhla-radar; +https://github.com/Negjir/dakhla-radar)"}


def http_get(url, timeout=240):
    """GET with a browser-like User-Agent (some servers refuse Python's default one)."""
    import urllib.request

    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as resp:
        return resp.read()


def _open(dataset_id, var, day, box):
    import copernicusmarine as cm

    kw = dict(
        dataset_id=dataset_id,
        variables=[var],
        minimum_longitude=box["lon_min"],
        maximum_longitude=box["lon_max"],
        minimum_latitude=box["lat_min"],
        maximum_latitude=box["lat_max"],
        # ocean colour arrives 3-5 days late: ask for a wide window and keep the newest day that has data
        start_datetime=f"{day - dt.timedelta(days=8)}T00:00:00",
        end_datetime=f"{day}T23:59:59",
    )
    try:
        ds = cm.open_dataset(coordinates_selection_method="inside", **kw)   # clip the window to what exists
    except TypeError:   # older toolbox without this option
        ds = cm.open_dataset(**kw)
    da = ds[var]
    if "depth" in da.dims:
        da = da.isel(depth=0)
    # keep the latest day that actually has data
    for t in range(da.sizes["time"] - 1, -1, -1):
        frame = da.isel(time=t)
        if np.isfinite(frame.values).mean() > 0.2:
            used = str(np.datetime_as_string(frame["time"].values, unit="D"))
            if (day - dt.date.fromisoformat(used)).days > MAX_AGE_DAYS:
                raise RuntimeError(f"newest data is {used}, too old")
            return frame, used
    raise RuntimeError("no valid data in the last 8 days")


def fetch(candidates, day, box, label):
    errors = []
    for dataset_id, var in candidates:
        try:
            frame, used = _open(dataset_id, var, day, box)
            print(f"[ok] {label}: {dataset_id} ({used})")
            return frame, used
        except Exception as e:  # noqa: BLE001 - we want to try the next id
            errors.append(f"{dataset_id}: {e}")
            print(f"[skip] {label}: {dataset_id} -> {e}")
    raise SystemExit(
        f"Could not download {label}. Tried:\n  " + "\n  ".join(errors)
        + "\nRun `python radar.py --check` and update config.json."
    )


# NASA / NOAA data through ERDDAP servers: no account needed.
SST_ERDDAP = CFG.get("sst_erddap") or [
    # NASA JPL MUR SST, 0.01 deg (~1 km), daily, multi-satellite (MODIS, VIIRS, AVHRR, AMSR2 microwave)
    {"name": "NASA MUR 1km", "base": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41",
     "mirrors": ["https://upwell.pfeg.noaa.gov/erddap/griddap/jplMURSST41"], "var": "analysed_sst", "alt": False, "stride": 3},
]
CHL_ERDDAP = CFG.get("chl_erddap") or [
    # NOAA VIIRS (S-NPP + NOAA-20) chlorophyll, gap-filled DINEOF, ~9 km, daily
    {"name": "NOAA VIIRS 9km", "base": "https://coastwatch.noaa.gov/erddap/griddap/noaacwNPPN20VIIRSDINEOFDaily", "var": "chlor_a", "alt": True, "stride": 1, "lat_desc": True},
]
MAX_AGE_DAYS = CFG.get("max_data_age_days", 5)
SOURCES = {}   # what was actually used today, shown in the message and page


def erddap(src, day, box):
    """Download one day (or the latest available) from an ERDDAP griddap dataset. Returns a 2-D DataArray (lat descending) and its date."""
    import tempfile

    import xarray as xr

    st = src.get("stride", 1)
    la0, la1 = (box["lat_max"], box["lat_min"]) if src.get("lat_desc") else (box["lat_min"], box["lat_max"])   # ERDDAP wants dataset order
    sp = f"[({la0}):{st}:({la1})][({box['lon_min']}):{st}:({box['lon_max']})]"
    alt = "[(0.0)]" if src.get("alt") else ""
    times = [f"({d}T12:00:00Z)" for d in (day, day - dt.timedelta(days=1), day - dt.timedelta(days=2))] + ["(last)"]
    last_err = None
    for tq, base in [(t, b) for b in [src["base"]] + src.get("mirrors", []) for t in times]:
        url = f"{base}.nc?{src['var']}[{tq}]{alt}{sp}"
        try:
            data = http_get(url)
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
                f.write(data)
                path = f.name
            ds = xr.open_dataset(path)
            da = ds[src["var"]]
            for d in [x for x in da.dims if x not in ("latitude", "longitude")]:
                da = da.isel({d: 0})
            used = str(np.datetime_as_string(ds["time"].values[0], unit="D"))
            age = (day - dt.date.fromisoformat(used)).days
            if age > MAX_AGE_DAYS:
                raise RuntimeError(f"latest data is {used}, too old")
            if np.isfinite(da.values).mean() < 0.2:
                raise RuntimeError("mostly empty")
            return da.sortby("latitude", ascending=False), used
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(last_err)


def fetch_any(erddap_list, cm_list, day, box, label, erddap_first):
    """Try NASA/NOAA (ERDDAP) and Copernicus sources in the given order; return the first that works."""
    tries = []
    for src in erddap_list:
        tries.append((src["name"], lambda s=src: erddap(s, day, box)))
    cm = [(f"Copernicus {d}", lambda d=d, v=v: _open(d, v, day, box)) for d, v in cm_list]
    tries = tries + cm if erddap_first else cm + tries
    errors = []
    for name, fn in tries:
        try:
            frame, used = fn()
            print(f"[ok] {label}: {name} ({used})")
            SOURCES[label] = f"{name.split(' cmems')[0].split(' METOFFICE')[0]} {used}".strip()
            return frame, used
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            print(f"[skip] {label}: {name} -> {e}")
    raise SystemExit(f"Could not download {label}. Tried:\n  " + "\n  ".join(errors))


def check():
    import copernicusmarine as cm

    for words in (["SST_GLO_SST_L4_NRT"], ["OCEANCOLOUR_GLO_BGC_L4_NRT"], ["OCEANCOLOUR_ATL_BGC_L4_NRT"]):
        cat = cm.describe(contains=words)
        for p in getattr(cat, "products", []):
            for d in p.datasets:
                print(p.product_id, "->", d.dataset_id)


def load_real(day, box):
    # SST: NASA MUR first (1 km), Copernicus OSTIA as backup.
    sst_da, sst_day = fetch_any(SST_ERDDAP, SST_CANDIDATES, day, box, "SST", erddap_first=True)
    # Chlorophyll: Copernicus first, NOAA VIIRS as backup.
    try:
        chl_da, chl_day = fetch_any(CHL_ERDDAP, CHL_CANDIDATES, day, box, "CHL", erddap_first=False)
    except SystemExit as e:
        print("[warn] no chlorophyll from any source today:", e)
        chl_da = None
    lat_name = "latitude" if "latitude" in sst_da.coords else "lat"
    lon_name = "longitude" if "longitude" in sst_da.coords else "lon"
    sst_da = sst_da.sortby(lat_name, ascending=False)
    lats = sst_da[lat_name].values
    lons = sst_da[lon_name].values
    sst = sst_da.values.astype("float32")
    if np.nanmean(sst) > 200:  # Kelvin -> Celsius
        sst = sst - 273.15
    if chl_da is None:
        return lats, lons, sst, np.full_like(sst, np.nan), sst_day
    clat = "latitude" if "latitude" in chl_da.coords else "lat"
    clon = "longitude" if "longitude" in chl_da.coords else "lon"
    chl = chl_da.sortby(clat).interp({clat: np.sort(lats), clon: lons}, method="linear").sortby(clat, ascending=False).values.astype("float32")
    return lats, lons, sst, chl, sst_day


def load_demo(day, box):
    """Synthetic upwelling field so the whole pipeline can be tested without an account."""
    lats = np.arange(box["lat_max"], box["lat_min"], -0.05)
    lons = np.arange(box["lon_min"], box["lon_max"], 0.05)
    LON, LAT = np.meshgrid(lons, lats)
    coast = np.interp(LAT, [21.0, 22.3, 23.6, 24.4, 26.5], [-17.05, -16.7, -16.0, -15.3, -14.25])
    d = (coast - LON) * 102  # km, rough
    rng = np.random.default_rng(int(day.strftime("%Y%m%d")))
    from scipy.ndimage import gaussian_filter

    noise = gaussian_filter(rng.standard_normal(LAT.shape), 8) * 6
    up = np.exp(-np.clip(d, 0, None) / (70 + 40 * noise))
    sst = 19 + (26.5 - LAT) * 0.4 + 2.5 - 5.5 * up + noise * 0.5
    chl = 10 ** (-0.7 + 1.7 * np.exp(-np.clip(d, 0, None) / 110) + 0.3 * noise)
    sst[d < 0] = np.nan
    chl[d < 0] = np.nan
    return lats, lons, sst.astype("float32"), chl.astype("float32"), str(day)


# ----------------------------------------------------------------- model
def nm_between(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 60
    dx = (lon2 - lon1) * 60 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def bearing(lat1, lon1, lat2, lon2):
    dy = lat2 - lat1
    dx = (lon2 - lon1) * math.cos(math.radians(lat2))
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def in_closed(lat, lon, zones):
    return any(z["lat_min"] < lat < z["lat_max"] and z["lon_min"] < lon < z["lon_max"] for z in zones)


def read_reports(day):
    path = ROOT / "reports.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                d = dt.date.fromisoformat(r["date"].strip())
                age = (day - d).days
                if 0 <= age <= CFG.get("report_memory_days", 10):
                    rows.append({**r, "lat": float(r["lat"]), "lon": float(r["lon"]),
                                 "tonnes": float(r.get("tonnes") or 0), "age": age})
            except (KeyError, ValueError):
                continue
    return rows


def score_field(lats, lons, sst, chl, reports):
    from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel

    ocean = np.isfinite(sst)
    step_km = abs(lats[1] - lats[0]) * 111
    filled = np.where(ocean, sst, np.nanmean(sst))
    sm = gaussian_filter(filled, 1.2)
    coslat = math.cos(math.radians(float(np.mean(lats))))
    gx = sobel(sm, axis=1) / (8 * step_km * coslat)
    gy = sobel(sm, axis=0) / (8 * step_km)
    grad = np.hypot(gx, gy)  # deg C per km
    grad[~ocean] = 0
    grad_s = gaussian_filter(grad, 1.5)
    ref = np.nanpercentile(grad_s[ocean], 97) or 1e-6
    front = np.clip(grad_s / ref, 0, 1)

    lc = np.log10(np.where(np.isfinite(chl) & (chl > 0), chl, 0.1))
    chl_ok = np.exp(-((lc - CFG.get("chl_best_log10", 0.25)) ** 2) / 0.35)

    dist_km = distance_transform_edt(ocean) * step_km
    dist_w = np.where(dist_km < 4, 0.2, np.exp(-np.clip(dist_km - CFG.get("max_offshore_km", 150), 0, None) / 60))

    s = (front * 0.62 + chl_ok * 0.38) * dist_w
    LON, LAT = np.meshgrid(lons, lats)
    for r in reports:
        w = min(1.0, r["tonnes"] / 40) * math.exp(-r["age"] / 5)
        d2 = (LAT - r["lat"]) ** 2 + ((LON - r["lon"]) * coslat) ** 2
        s = s + 0.18 * w * np.exp(-d2 / 0.02)
    s[~ocean] = 0
    mx = s.max() or 1
    return np.power(s / mx, 2.4), grad


def pick_zones(lats, lons, score, n=5):
    from scipy.ndimage import maximum_filter

    port = CFG["port"]
    peaks = (score == maximum_filter(score, size=7)) & (score > 0.2)
    cand = []
    for j, i in zip(*np.nonzero(peaks)):
        la, lo = float(lats[j]), float(lons[i])
        if in_closed(la, lo, CFG.get("closed_zones", [])):
            continue
        dnm = nm_between(port["lat"], port["lon"], la, lo)
        if dnm > CFG.get("max_range_nm", 150):
            continue
        cand.append((float(score[j, i]), la, lo, dnm))
    cand.sort(reverse=True)
    out = []
    for v, la, lo, dnm in cand:
        if all(math.hypot(la - z["lat"], (lo - z["lon"]) * 0.92) > 0.45 for z in out):
            b = bearing(port["lat"], port["lon"], la, lo)
            out.append({"id": "ABCDE"[len(out)], "lat": round(la, 4), "lon": round(lo, 4),
                        "confidence": round(v * 100), "distance_nm": round(dnm),
                        "bearing_deg": round(b),
                        "fuel_l": round(dnm * 2 * CFG.get("fuel_l_per_nm", 15))})
        if len(out) == n:
            break
    return out


# ----------------------------------------------------------------- outputs
def dm(v, width, hemi):
    a = abs(v)
    d = int(a)
    m = (a - d) * 60
    return f"{d:0{width}d}°{m:04.1f}'{hemi[0] if v >= 0 else hemi[1]}"


def pos(la, lo):
    return f"{dm(la, 2, 'NS')} {dm(lo, 3, 'EW')}"


COMPASS_AR = ["شمال", "شمال شرق", "شرق", "جنوب شرق", "جنوب", "جنوب غرب", "غرب", "شمال غرب"]
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def message(zones, data_day):
    if not zones:
        return f"رادار الصيد · {data_day}\nلا توجد جبهات قوية اليوم في مدى {CFG.get('max_range_nm',150)} ميل."
    z = zones[0]
    k = round(z["bearing_deg"] / 45) % 8
    lines = [
        f"🐟 رادار الصيد · {data_day}",
        f"أفضل منطقة: {z['id']} (ثقة {z['confidence']}%)",
        f"الموقع: {pos(z['lat'], z['lon'])}",
        f"من الميناء: {z['distance_nm']} nm · {z['bearing_deg']}° {COMPASS[k]} ({COMPASS_AR[k]})",
        f"الوقود ذهاباً وإياباً ≈ {z['fuel_l']:,} L",
        "",
        "مناطق أخرى:",
    ]
    for o in zones[1:]:
        lines.append(f"{o['id']}: {pos(o['lat'], o['lon'])} · {o['distance_nm']} nm · {o['confidence']}%")
    if SOURCES:
        lines += ["", "المصادر: " + " · ".join(f"{'حرارة' if k == 'SST' else 'كلوروفيل'} {v}" for k, v in SOURCES.items())]
    lines += ["", "بعد الصيد أرسل: الموقع، الكمية، النوع، ساعات البحث."]
    return "\n".join(lines)


def draw_map(lats, lons, sst, chl, score, grad, zones, data_day, path, only_index=False, demo=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ocean = np.isfinite(sst)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    cm_score = LinearSegmentedColormap.from_list(
        "s", ["#071a2c", "#0f3a55", "#136b6f", "#d39a3a", "#ffb547", "#ff6b4a"])
    cm_sst = LinearSegmentedColormap.from_list(
        "t", ["#1b2a6b", "#2563a8", "#2fa3b7", "#8fd3a5", "#f2d36b", "#f08a4b"])
    cm_chl = LinearSegmentedColormap.from_list(
        "c", ["#0e1f45", "#1d4f7a", "#1f8a70", "#7cc64a", "#e2e65a"])

    if only_index:
        fig, ax0 = plt.subplots(figsize=(7.2, 9), dpi=120, facecolor="#06121d")
        axes = [ax0]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 7.2), dpi=110, facecolor="#06121d",
                                 gridspec_kw={"width_ratios": [1.6, 1, 1]})
    panels = [
        (np.ma.masked_where(~ocean, score), cm_score, 0, 1, "Fishing index"),
        (np.ma.masked_where(~ocean, sst), cm_sst, np.nanpercentile(sst, 2), np.nanpercentile(sst, 98), "SST °C"),
        (np.ma.masked_where(~ocean, np.log10(np.clip(chl, 0.05, 50))), cm_chl, -1, 1, "Chl log10 mg/m³"),
    ]
    port = CFG["port"]
    gref = np.nanpercentile(grad[ocean], 93)
    for ax, (arr, cmap, lo, hi, title) in zip(axes, panels[:len(axes)]):
        ax.set_facecolor("#2a2418")
        im = ax.imshow(arr, extent=ext, origin="upper", cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="bilinear", aspect=1 / math.cos(math.radians(23.7)))
        ax.contour(lons, lats, ocean.astype(float), levels=[0.5], colors="#e6d2aa", linewidths=0.8)
        ax.set_title(title, color="#e6eef3", fontsize=11)
        ax.tick_params(colors="#8aa3b4", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#1c3448")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(colors="#8aa3b4", labelsize=7)
        ax.plot(port["lon"], port["lat"], "o", color="#4fe3d6", ms=6, mec="#06121d")
    ax = axes[0]
    fy, fx = np.nonzero((grad > gref) & ocean)
    ax.scatter(lons[fx], lats[fy], s=1.2, c="white", alpha=0.6, linewidths=0)
    for z in CFG.get("closed_zones", []):
        ax.add_patch(plt.Rectangle((z["lon_min"], z["lat_min"]), z["lon_max"] - z["lon_min"],
                                   z["lat_max"] - z["lat_min"], fill=False, ec="#ff6b4a", lw=1.4, hatch="///"))
    if zones:
        z = zones[0]
        ax.plot([port["lon"], z["lon"]], [port["lat"], z["lat"]], "--", color="white", lw=1.2)
    for i, z in enumerate(zones):
        ax.plot(z["lon"], z["lat"], "o", ms=17 if i == 0 else 13, mfc="#06121d", mec="white", mew=2)
        ax.text(z["lon"], z["lat"], z["id"], color="white", ha="center", va="center", fontsize=9, weight="bold")
        if only_index:
            ax.text(z["lon"] - 0.12, z["lat"], f"{z['distance_nm']} nm", color="white", ha="right", va="center",
                    fontsize=8, bbox=dict(fc="#06121d", ec="none", alpha=0.7, pad=1.5))
    src = "DEMO DATA" if demo else "Copernicus Marine"
    fig.suptitle(f"Dakhla Fish Radar · {data_day} · {src}", color="#e6eef3", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_page(zones, msg, data_day, demo):
    rows = "".join(
        f"<tr><td>{z['id']}</td><td>{z['confidence']}%</td><td dir='ltr'>{pos(z['lat'], z['lon'])}</td>"
        f"<td>{z['distance_nm']} nm · {z['bearing_deg']}°</td><td>{z['fuel_l']:,} L</td></tr>" for z in zones)
    warn = "<p class='w'>بيانات تجريبية (demo) وليست قراءات حقيقية.</p>" if demo else ""
    page = f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>رادار الصيد · الداخلة</title>
<style>body{{margin:0;background:#06121d;color:#e6eef3;font-family:system-ui,sans-serif;padding:16px}}
main{{max-width:1100px;margin:auto}}img{{width:100%;border-radius:12px;border:1px solid #1c3448}}
table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #1c3448;text-align:start}}
pre{{white-space:pre-wrap;background:#0a2a22;border:1px solid #1f5b49;border-radius:12px;padding:12px;font:15px/1.7 system-ui}}
button{{background:#4fe3d6;border:0;border-radius:999px;padding:12px 18px;font-weight:700;font-size:15px}}
.w{{color:#ffb547}}.s{{color:#8aa3b4;font-size:13px}}.t{{overflow-x:auto}}</style></head><body><main>
<h1>رادار الصيد · الداخلة</h1><p><a style="color:#4fe3d6;font-weight:700" href="forecast.html">← توقعات الأيام الثلاثة القادمة</a></p><p><a style="color:#4fe3d6;font-weight:700" href="coastal/">← رادار الساحل للقوارب التقليدية</a></p><p class="s">بيانات يوم {data_day} · تحديث {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC</p>{warn}
<img src="captain.png?v={data_day}" alt="خريطة مؤشر الصيد للربابنة"><p class="s">الصورة أعلاه للإرسال إلى الربابنة. التفاصيل الكاملة: <a style="color:#4fe3d6" href="map.png?v={data_day}">الحرارة والكلوروفيل</a></p>
<h2>رسالة الربابنة</h2><pre id="m">{html.escape(msg)}</pre>
<button onclick="navigator.clipboard.writeText(document.getElementById('m').innerText).then(()=>this.textContent='تم النسخ')">نسخ الرسالة</button>
<h2>المناطق</h2><div class="t"><table><tr><th></th><th>الثقة</th><th>الموقع</th><th>المسافة</th><th>وقود ذهاب وإياب</th></tr>{rows}</table></div>
<p class="s">المؤشر يجمع الجبهات الحرارية والكلوروفيل وتقارير الربابنة. مساعدة في القرار وليس ضماناً للصيد. احترم مناطق المنع والحصص الرسمية.</p>
</main></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
        return
    day = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
    box = CFG["box"]
    lats, lons, sst, chl, data_day = (load_demo if a.demo else load_real)(day, box)
    reports = read_reports(day)
    score, grad = score_field(lats, lons, sst, chl, reports)
    zones = pick_zones(lats, lons, score)
    msg = message(zones, data_day)

    OUT.mkdir(exist_ok=True)
    (OUT / "archive").mkdir(exist_ok=True)
    draw_map(lats, lons, sst, chl, score, grad, zones, data_day, OUT / "map.png", demo=a.demo)
    draw_map(lats, lons, sst, chl, score, grad, zones, data_day, OUT / "captain.png", only_index=True, demo=a.demo)
    draw_map(lats, lons, sst, chl, score, grad, zones, data_day, OUT / "archive" / f"{data_day}.png", demo=a.demo)
    (OUT / "message.txt").write_text(msg, encoding="utf-8")
    (OUT / "zones.json").write_text(json.dumps({"data_day": data_day, "demo": a.demo, "reports_used": len(reports),
                                                "zones": zones}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "archive" / "zones_history.csv").open("a", encoding="utf-8") as f:
        for z in zones:
            f.write(f"{data_day},{z['id']},{z['lat']},{z['lon']},{z['confidence']},{z['distance_nm']}\n")
    write_page(zones, msg, data_day, a.demo)
    print(msg)


if __name__ == "__main__":
    sys.exit(main())
