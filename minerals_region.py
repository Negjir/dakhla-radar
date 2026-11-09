"""
MADAR Minerals - regional scan of the Sahara (Dakhla-Oued Ed-Dahab and Laayoune-Sakia El Hamra).

Reconnaissance at ~100 m over ~266,000 km2, then 10 m picture chips of the best targets.

  1. For every Sentinel-2 tile (110 x 110 km) over the area, pick the clearest recent pass and read it at
     ~100 m from the image overviews (fast: the whole Sahara in one run).
  2. Band ratios as in minerals.py: clay (B11/B12), ferric iron (B4/B2), ferrous iron (B12/B8A + B3/B4),
     experimental rare-earth absorption (red-edge).
  3. ROCK-OUTCROP MASK. In the Sahara most false targets are sand sheets, dunes and sabkhas (salt flats):
     they are smooth at 100 m. Rock is rough. We keep pixels whose local texture (5 x 5 standard deviation
     of the SWIR band) is above the regional median, and drop very bright flats.
  4. Three models: iron (BIF / iron oxides), alteration (gold-copper style: clay + iron oxide),
     carbonatite (iron + experimental rare-earth index, the Twihinate / Lafhouda type).
  5. Targets = the strongest clusters of the top 0.15 % pixels of each model, ranked.
  6. Blind check on the four ONHYM carbonatite prospects (never used in the analysis).
  7. 10 m chips (true colour and SWIR geology composite) for the best targets.

  python minerals_region.py           real data (GitHub Actions: network to Earth Search / AWS)
  python minerals_region.py --demo    synthetic Sahara, coarse grid, to test the page
"""
import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "minerals" / "sahara"
STAC = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

BOX = [20.76, 27.70, -17.20, -8.65]        # lat0, lat1, lon0, lon1
RES = 0.0009                                # ~100 m
# Approximate outline of the area scanned (lon, lat). The western side runs at sea; the sea is removed
# by the water mask. Borders are drawn for orientation only, from the usual published coordinates.
OUTLINE = [(-17.06, 20.77), (-17.05, 21.33), (-13.0, 21.33), (-13.0, 23.45), (-12.0, 23.45),
           (-12.0, 26.0), (-8.67, 26.0), (-8.67, 27.667), (-13.17, 27.667), (-14.6, 27.667),
           (-16.2, 25.4), (-17.35, 23.4), (-17.45, 21.4), (-17.3, 20.77)]
REGION_SPLIT_LAT = 24.35   # approximate limit between Dakhla-Oued Ed-Dahab (south) and Laayoune-Sakia El Hamra
TOWNS = [  # approximate, for orientation and for "40 km SE of Aousserd" descriptions
    {"ar": "الداخلة", "fr": "Dakhla", "lat": 23.705, "lon": -15.935},
    {"ar": "أوسرد", "fr": "Aousserd", "lat": 22.553, "lon": -14.330},
    {"ar": "بئر كندوز", "fr": "Bir Gandouz", "lat": 21.610, "lon": -16.460},
    {"ar": "تيشلا", "fr": "Tichla", "lat": 21.590, "lon": -14.970},
    {"ar": "بئر أنزران", "fr": "Bir Anzarane", "lat": 23.890, "lon": -14.530},
    {"ar": "الكويرة", "fr": "Lagouira", "lat": 20.840, "lon": -17.090},
    {"ar": "بوجدور", "fr": "Boujdour", "lat": 26.126, "lon": -14.485},
    {"ar": "العيون", "fr": "Laâyoune", "lat": 27.153, "lon": -13.203},
    {"ar": "السمارة", "fr": "Smara", "lat": 26.739, "lon": -11.672},
    {"ar": "كلتة زمور", "fr": "Guelta Zemmur", "lat": 25.140, "lon": -12.370},
]
# Known carbonatite prospects (ONHYM). Used ONLY after the analysis, for the blind check. Approximate.
KNOWN = [
    {"ar": "تويهينات", "fr": "Twihinate", "lat": 22.44, "lon": -16.04, "model": "carbonatite"},
    {"ar": "كليبات لفحودة", "fr": "Gleibat Lafhouda", "lat": 22.15, "lon": -14.85, "model": "carbonatite"},
    {"ar": "كليبات الفرنان", "fr": "Gleibat El Fernan", "lat": 22.195, "lon": -14.85, "model": "carbonatite"},
    {"ar": "كليبات الدراك", "fr": "Gleibat Drag", "lat": 22.23, "lon": -14.93, "model": "carbonatite"},
]
MODELS = {
    "iron": {"ar": "الحديد", "fr": "Fer", "w": {"ferric": 0.5, "ferrous": 0.5}},
    "alteration": {"ar": "التغيّر الحراري (ذهب ونحاس)", "fr": "Altération (or, cuivre)", "w": {"clay": 0.5, "ferric": 0.5}},
    "carbonatite": {"ar": "الكربوناتيت والعناصر النادرة", "fr": "Carbonatite et terres rares",
                    "w": {"ferric": 0.35, "ferrous": 0.35, "ree": 0.3}},
}
TOP_PERCENT = 0.15
MIN_PIXELS = 4           # 4 ha at 100 m
N_TARGETS = 30           # per model
N_CHIPS = 12             # per model, 10 m pictures
CHECK_KM = 3.0
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "rededge1", "rededge2", "rededge3", "scl"]


# ----------------------------------------------------------------------------- grid helpers
def shape(res=None):
    res = res or RES
    lat0, lat1, lon0, lon1 = BOX
    return int(round((lat1 - lat0) / res)), int(round((lon1 - lon0) / res))


def km(a_lat, a_lon, b_lat, b_lon):
    p = math.pi / 180
    d = (math.sin((b_lat - a_lat) * p / 2) ** 2
         + math.cos(a_lat * p) * math.cos(b_lat * p) * math.sin((b_lon - a_lon) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(d))


def bearing(a_lat, a_lon, b_lat, b_lon):
    p = math.pi / 180
    y = math.sin((b_lon - a_lon) * p) * math.cos(b_lat * p)
    x = math.cos(a_lat * p) * math.sin(b_lat * p) - math.sin(a_lat * p) * math.cos(b_lat * p) * math.cos((b_lon - a_lon) * p)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


DIR_AR = ["شمال", "شمال شرق", "شرق", "جنوب شرق", "جنوب", "جنوب غرب", "غرب", "شمال غرب"]
DIR_FR = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]


def where(lat, lon):
    t = min(TOWNS, key=lambda t: km(lat, lon, t["lat"], t["lon"]))
    d = km(t["lat"], t["lon"], lat, lon)
    k = int(((bearing(t["lat"], t["lon"], lat, lon) + 22.5) % 360) // 45)
    return {"ar": f"{d:.0f} كم {DIR_AR[k]} {t['ar']}", "fr": f"{d:.0f} km {DIR_FR[k]} de {t['fr']}", "km": round(d)}


def outline_mask(h, w, res):
    from matplotlib.path import Path as MPath

    lat0, lat1, lon0, lon1 = BOX
    path = MPath(OUTLINE)
    m = np.zeros((h, w), bool)
    lons = lon0 + (np.arange(w) + 0.5) * res
    for r0 in range(0, h, 400):
        r1 = min(h, r0 + 400)
        lats = lat1 - (np.arange(r0, r1) + 0.5) * res
        xx, yy = np.meshgrid(lons, lats)
        m[r0:r1] = path.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(r1 - r0, w)
    return m


# ----------------------------------------------------------------------------- data
def pick_items(max_cloud=3, months=14):
    """Best pass per Sentinel-2 tile (lowest cloud, fullest coverage, then most recent)."""
    from pystac_client import Client

    lat0, lat1, lon0, lon1 = BOX
    end = date.today()
    start = end - timedelta(days=30 * months)
    items = list(Client.open(STAC).search(
        collections=[COLLECTION], bbox=[lon0, lat0, lon1, lat1], datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=6000).items())
    from matplotlib.path import Path as MPath

    path = MPath(OUTLINE)
    tiles = {}
    for it in items:
        p = it.properties
        tile = p.get("grid:code") or "%s%s%s" % (p.get("mgrs:utm_zone"), p.get("mgrs:latitude_band"), p.get("mgrs:grid_square"))
        bx0, by0, bx1, by1 = it.bbox
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        corners = [(bx0, by0), (bx0, by1), (bx1, by0), (bx1, by1), (cx, cy)]
        if not any(path.contains_point(c) for c in corners):
            continue
        tiles.setdefault(tile, []).append(it)
    ranked = {}
    for t, its in tiles.items():
        its.sort(key=lambda i: (round(i.properties.get("s2:nodata_pixel_percentage", 0) / 10),
                                round(i.properties.get("eo:cloud_cover", 100)),
                                -i.datetime.toordinal()))
        best = []
        for i in its:                       # different orbits first: their swath edges are elsewhere
            if all(i.properties.get("sat:relative_orbit") != b.properties.get("sat:relative_orbit") for b in best):
                best.append(i)
        best += [i for i in its if i not in best]
        ranked[t] = best[:5]
    print(f"[stac] {len(items)} passes, {len(ranked)} tiles over the area")
    return ranked


def read_item(it, res):
    """Bands of one item, resampled to `res` degrees, as (window, dict). Reads image overviews."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    lat0, lat1, lon0, lon1 = BOX
    bx0, by0, bx1, by1 = it.bbox
    c0 = max(0, int(math.floor((bx0 - lon0) / res)))
    c1 = min(shape(res)[1], int(math.ceil((bx1 - lon0) / res)))
    r0 = max(0, int(math.floor((lat1 - by1) / res)))
    r1 = min(shape(res)[0], int(math.ceil((lat1 - by0) / res)))
    if c1 <= c0 or r1 <= r0:
        return None, None
    dst_t = from_origin(lon0 + c0 * res, lat1 - r0 * res, res, res)
    target_m = res * 111320
    off = 0.0
    if str(it.properties.get("s2:processing_baseline", "00.00")) >= "04.00" and \
            not it.properties.get("earthsearch:boa_offset_applied", False):
        off = -1000.0
    out = {}
    for b in BANDS:
        with rasterio.open(it.assets[b].href) as src:
            f = max(1, int(round(target_m / src.res[0])))
            oh, ow = max(1, src.height // f), max(1, src.width // f)
            rs = Resampling.mode if b == "scl" else Resampling.average
            a = src.read(1, out_shape=(oh, ow), resampling=rs).astype("float32")
            st = src.transform * src.transform.scale(src.width / ow, src.height / oh)
            crs = src.crs
        a[a == 0] = np.nan
        dst = np.full((r1 - r0, c1 - c0), np.nan, "float32")
        reproject(a, dst, src_transform=st, src_crs=crs, dst_transform=dst_t, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest if b == "scl" else Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        out[b] = dst if b == "scl" else (dst + off) / 10000.0
    return (r0, r1, c0, c1), out


def ratios(b):
    eps = 1e-4
    ndvi = (b["nir08"] - b["red"]) / (b["nir08"] + b["red"] + eps)
    bright = (b["blue"] + b["green"] + b["red"]) / 3
    scl = b["scl"]
    ok = np.isfinite(b["red"]) & np.isfinite(b["swir22"]) & np.isin(scl, [2, 4, 5, 7, 11])
    ok &= (ndvi < 0.25) & (b["swir22"] > 0.03) & (bright < 0.45)
    r = {
        "clay": b["swir16"] / (b["swir22"] + eps),
        "ferric": b["red"] / (b["blue"] + eps),
        "ferrous": b["swir22"] / (b["nir08"] + eps) + b["green"] / (b["red"] + eps),
        "ree": (b["rededge1"] + b["nir08"]) / (b["rededge2"] + b["rededge3"] + eps),
        "swir22": b["swir22"], "bright": bright,
    }
    # Damp ground: sabkha floors and wet flats are dark in the SWIR and mimic iron / clay anomalies.
    ndmi = (b["nir08"] - b["swir16"]) / (b["nir08"] + b["swir16"] + eps)
    r["wet"] = (ndmi > -0.06) & np.isfinite(ndmi)
    rgb = np.dstack([b["red"], b["green"], b["blue"]])
    return ok, r, rgb


def scan(demo):
    h, w = shape()
    keys = ["clay", "ferric", "ferrous", "ree", "swir22", "bright"]
    wet = np.zeros((h, w), bool)
    G = {k: np.full((h, w), np.nan, "float16") for k in keys}
    valid = np.zeros((h, w), bool)
    filled = np.zeros((h, w), bool)
    land = np.zeros((h, w), bool)
    rgb = np.zeros((h, w, 3), "uint8")
    used = []
    inside = outline_mask(h, w, RES)
    if demo:
        b = demo_scene(h, w)
        ok, r, col = ratios(b)
        for k in keys:
            G[k][:] = r[k]
        wet[:] = r["wet"]
        G["wet"] = wet
        valid[:] = ok & inside
        land[:] = (b["scl"] != 6) & inside
        rgb[:] = to8(col)
        return G, valid, rgb, inside, land, [{"id": "DEMO", "date": date.today().isoformat(), "cloud": 0, "tile": "-"}], {}
    ranked = pick_items()
    by_id = {}
    for rnd_i in range(5):                       # best pass first, then fill the gaps with the next ones
        for tile, its in sorted(ranked.items()):
            if rnd_i >= len(its):
                continue
            it = its[rnd_i]
            bx0, by0, bx1, by1 = it.bbox
            r0 = int((BOX[1] - by1) / RES); r1 = int((BOX[1] - by0) / RES)
            c0 = int((bx0 - BOX[2]) / RES); c1 = int((bx1 - BOX[2]) / RES)
            r0, c0 = max(r0, 0), max(c0, 0)
            sub_in = inside[r0:r1, c0:c1]
            if rnd_i > 0:
                gap = sub_in & ~filled[r0:r1, c0:c1]
                if sub_in.sum() == 0 or gap.sum() < 0.003 * sub_in.size:
                    continue
            try:
                win, b = read_item(it, RES)
            except Exception as e:  # noqa: BLE001
                print("[skip]", it.id, e)
                continue
            if win is None:
                continue
            r0, r1, c0, c1 = win
            ok, r, col = ratios(b)
            have = np.isfinite(b["red"]) & np.isin(b["scl"], [2, 4, 5, 6, 7, 11])
            put = have & ~filled[r0:r1, c0:c1]
            if not put.any():
                continue
            for k in keys:
                sub = G[k][r0:r1, c0:c1]
                sub[put] = r[k][put]
            valid[r0:r1, c0:c1][put] = ok[put]
            wet[r0:r1, c0:c1][put] = r["wet"][put]
            rgb[r0:r1, c0:c1][put] = to8(col)[put]
            filled[r0:r1, c0:c1] |= put
            land[r0:r1, c0:c1] |= put & (b["scl"] != 6)
            rec = {"id": it.id, "date": it.datetime.date().isoformat(),
                   "cloud": round(float(it.properties.get("eo:cloud_cover", 0)), 2), "tile": tile}
            used.append(rec)
            by_id[it.id] = it
            print("[ok] %-28s %s cloud %.1f%%  +%.2f Mpx  coverage %.1f%%" % (
                it.id, rec["date"], rec["cloud"], put.sum() / 1e6, 100 * (filled & inside).sum() / inside.sum()))
    valid &= inside
    land &= inside
    G["wet"] = wet
    return G, valid, rgb, inside, land, used, by_id


def to8(col, lo=0.0, hi=0.42):
    x = np.clip((np.nan_to_num(col) - lo) / (hi - lo), 0, 1)
    return (np.power(x, 1 / 1.5) * 255).astype("uint8")


def demo_scene(h, w, seed=7):
    """Synthetic Sahara: sand seas (smooth), rock (rough), sabkhas, and buried 'deposits'."""
    rng = np.random.default_rng(seed)
    lat0, lat1, lon0, lon1 = BOX
    res = (lat1 - lat0) / h
    yy, xx = np.mgrid[0:h, 0:w]
    lat = lat1 - (yy + 0.5) * res
    lon = lon0 + (xx + 0.5) * res
    rock = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 12) > 0.02
    rough = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 1) * 0.04
    base = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 30)
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    b = {"blue": 0.16 + 0.05 * base, "green": 0.22 + 0.06 * base, "red": 0.30 + 0.08 * base,
         "nir08": 0.36 + 0.06 * base, "swir16": 0.46 + 0.06 * base, "swir22": 0.38 + 0.05 * base,
         "rededge1": 0.32 + 0.07 * base, "rededge2": 0.34 + 0.07 * base, "rededge3": 0.35 + 0.07 * base}
    for k in b:
        b[k] = b[k] + np.where(rock, rough - 0.05, 0.0)
    for (la, lo, kind) in [(22.44, -16.04, "c"), (22.15, -14.85, "c"), (22.9, -13.6, "i"), (24.9, -13.1, "a"),
                           (26.2, -11.0, "i"), (21.9, -15.6, "a")]:
        g = np.exp(-(((lat - la) * 111) ** 2 + ((lon - lo) * 102) ** 2) / (2 * 2.5 ** 2))
        if kind in "ic":
            b["red"] = b["red"] + 0.07 * g
            b["blue"] = b["blue"] - 0.02 * g
            b["swir22"] = b["swir22"] + 0.03 * g
        if kind == "a":
            b["swir22"] = b["swir22"] - 0.05 * g
            b["red"] = b["red"] + 0.05 * g
        if kind == "c":
            b["rededge2"] = b["rededge2"] - 0.02 * g
        for k in b:
            b[k] = np.where(rock | (g > 0.3), b[k], b[k])
    sea = lon < (-16.3 + (lat - 21) * 0.55)
    for k in b:
        b[k] = np.where(sea, 0.02, b[k]).astype("float32")
    b["scl"] = np.where(sea, 6, 5).astype("float32")
    return b


# ----------------------------------------------------------------------------- analysis
def robust_stats(a, mask, n=4_000_000, seed=0):
    idx = np.flatnonzero(mask)
    if idx.size > n:
        idx = np.random.default_rng(seed).choice(idx, n, replace=False)
    v = a.ravel()[idx].astype("float32")
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) * 1.4826 + 1e-9)
    return med, mad


def outcrop(G, valid):
    s = np.nan_to_num(G["swir22"].astype("float32"))
    m = ndimage.uniform_filter(s, 5)
    m2 = ndimage.uniform_filter(s * s, 5)
    tex = np.sqrt(np.clip(m2 - m * m, 0, None))
    tex[~valid] = np.nan
    thr = float(np.nanpercentile(tex[valid][::7], 50))
    bright = G["bright"].astype("float32")
    damp = ndimage.binary_dilation(G["wet"], iterations=4)          # sabkhas + 400 m margin
    keep = valid & (tex >= thr) & (bright < 0.38) & ~damp
    # towns, ports and roads around them light up too: leave 6 km around each town out
    lat0, lat1, lon0, lon1 = BOX
    r = int(round(6.0 / (RES * 111)))
    for tw in TOWNS:
        j, i = int((lat1 - tw["lat"]) / RES), int((tw["lon"] - lon0) / RES)
        y0, x0, y1, x1 = max(j - r, 0), max(i - r, 0), min(j + r + 1, keep.shape[0]), min(i + r + 1, keep.shape[1])
        if y1 > y0 and x1 > x0:
            yy, xx = np.ogrid[y0 - j:y1 - j, x0 - i:x1 - i]
            keep[y0:y1, x0:x1] &= (yy * yy + xx * xx) > r * r
    print(f"[mask] damp ground (sabkha) removed: {damp[valid].mean() * 100:.1f}% of valid")
    print(f"[mask] valid {valid.mean() * 100:.1f}% of box, rock outcrop kept {keep.sum() / max(valid.sum(), 1) * 100:.1f}% of valid")
    return keep, tex


BG_KM = 30   # anomalies are measured against the rocks around them (~30 km), not against the whole Sahara


def local_background(a, keep, block=10, size_km=BG_KM):
    """Smooth regional background of an index (median of 1 km block means over ~30 km), full resolution."""
    h, w = a.shape
    H, W = -(-h // block), -(-w // block)
    pa = np.zeros((H * block, W * block), "float32")
    pk = np.zeros((H * block, W * block), "float32")
    pa[:h, :w] = np.where(keep, np.nan_to_num(a), 0)
    pk[:h, :w] = keep
    s_ = pa.reshape(H, block, W, block).sum((1, 3))
    n_ = pk.reshape(H, block, W, block).sum((1, 3))
    small = np.where(n_ > 5, s_ / np.maximum(n_, 1), np.nan)
    holes = ~np.isfinite(small)
    if holes.all():
        return np.zeros_like(a)
    idx = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
    small = small[tuple(idx)]
    k = max(3, int(round(size_km / (RES * 111 * block))) | 1)
    bg = ndimage.median_filter(small, size=k)
    bg = ndimage.uniform_filter(bg, size=3)
    full = ndimage.zoom(bg, block, order=1)[:h, :w]
    return full.astype("float32")


def model_score(G, keep, wts):
    score = np.zeros(keep.shape, "float32")
    for k, wt in wts.items():
        a = G[k].astype("float32")
        a -= local_background(a, keep)
        med, mad = robust_stats(a, keep)
        z = (a - med) / mad
        np.nan_to_num(z, copy=False)
        np.clip(z, 0, 8, out=z)
        score += wt * z
        del z, a
    score = ndimage.gaussian_filter(score, 1.0)
    score[~keep] = 0
    return score


def find_targets(score, keep, model):
    """Top targets of each region separately, so the Dakhla region is not crowded out by the north."""
    lat0, lat1, lon0, lon1 = BOX
    split = int(round((lat1 - REGION_SPLIT_LAT) / RES))
    out, thr = [], None
    for reg, sl in (("dakhla", slice(split, None)), ("laayoune", slice(0, split))):
        sub_keep = np.zeros_like(keep)
        sub_keep[sl] = keep[sl]
        if sub_keep.sum() < 100:
            continue
        tg, thr = _targets(score, sub_keep, model)
        out += tg
    return out, thr, None


def _targets(score, keep, model):
    lat0, lat1, lon0, lon1 = BOX
    v = score[keep]
    thr = float(np.percentile(v[::5], 100 - TOP_PERCENT))
    anom = (score >= thr) & keep
    lab, n = ndimage.label(anom, structure=np.ones((3, 3)))
    if n == 0:
        return [], thr
    idx = np.arange(1, n + 1)
    cnt = ndimage.sum(anom, lab, idx)
    strength = ndimage.sum(score, lab, idx)
    peak = ndimage.maximum(score, lab, idx)
    com = ndimage.center_of_mass(score, lab, idx)
    order = np.argsort(-strength)
    out = []
    for k in order:
        if cnt[k] < MIN_PIXELS:
            continue
        cy, cx = com[k]
        la, lo = lat1 - (cy + 0.5) * RES, lon0 + (cx + 0.5) * RES
        if any(km(la, lo, t["lat"], t["lon"]) < 4 for t in out):     # one target per 4 km
            continue
        a_ha = float(cnt[k]) * (RES * 111.32) * (RES * 111.32 * math.cos(math.radians(la))) * 100
        out.append({"lat": round(la, 5), "lon": round(lo, 5), "area_ha": round(a_ha), "peak": round(float(peak[k]), 2),
                    "strength": float(strength[k]), "model": model, "_comp": int(idx[k])})
        if len(out) >= N_TARGETS:
            break
    top = out[0]["strength"] if out else 1
    for r, t in enumerate(out):
        t["rank"] = r + 1
        t["rel"] = round(100 * t["strength"] / top)
        t["region"] = "dakhla" if t["lat"] < REGION_SPLIT_LAT else "laayoune"
        t["where"] = where(t["lat"], t["lon"])
        del t["strength"]
    return out, thr


def blind(score, keep, targets, model):
    lat0, lat1, lon0, lon1 = BOX
    v = score[keep][::5]
    res = []
    for k in KNOWN:
        if k["model"] != model:
            continue
        j, i = int((lat1 - k["lat"]) / RES), int((k["lon"] - lon0) / RES)
        r = int(round(2.0 / (RES * 111)))
        win = score[max(j - r, 0):j + r + 1, max(i - r, 0):i + r + 1]
        pct = float((v < win.max()).mean() * 100) if win.size else None
        near = sorted(((km(k["lat"], k["lon"], t["lat"], t["lon"]), t) for t in targets), key=lambda x: x[0])
        d, t = near[0] if near else (None, None)
        res.append({**k, "percentile": round(pct, 2) if pct is not None else None,
                    "nearest_km": round(d, 1) if d is not None else None,
                    "nearest_rank": t["rank"] if t else None, "hit": bool(d is not None and d <= CHECK_KM)})
    # chance: share of the scanned rock area that lies within CHECK_KM of one of the targets
    cover = np.zeros(keep.shape, bool)
    r = int(round(CHECK_KM / (RES * 111)))
    for t in targets:
        j, i = int((lat1 - t["lat"]) / RES), int((t["lon"] - lon0) / RES)
        y0, x0 = max(j - r, 0), max(i - r, 0)
        y1, x1 = min(j + r + 1, cover.shape[0]), min(i + r + 1, cover.shape[1])
        yy, xx = np.ogrid[y0 - j:y1 - j, x0 - i:x1 - i]
        cover[y0:y1, x0:x1] |= (yy * yy + xx * xx) <= r * r
    p = float((cover & keep).sum() / max(keep.sum(), 1))
    return res, round(p, 5)


# ----------------------------------------------------------------------------- chips (10 m)
def chips(targets, by_id, used, demo):
    """True colour (10 m) and SWIR geology composite (B12-B11-B2, 20 m) around the best targets."""
    from PIL import Image

    folder = OUT / "chips"
    folder.mkdir(parents=True, exist_ok=True)
    if demo:
        for t in targets:
            g = (np.random.default_rng(t["rank"]).random((200, 200, 3)) * 80 + 90).astype("uint8")
            for kind in ("tc", "geo"):
                name = f"{t['model']}_{t['region']}_{t['rank']}_{kind}.jpg"
                Image.fromarray(g).save(folder / name, quality=80)
                t[f"chip_{kind}"] = f"chips/{name}"
        return
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    half = 0.018      # ~4 km box
    res = 0.0001      # ~10 m
    n = int(2 * half / res)
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                       GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")
    items = list(by_id.values())
    with env:
        for t in targets:
            cands = [it for it in items if it.bbox[0] < t["lon"] - half and it.bbox[2] > t["lon"] + half
                     and it.bbox[1] < t["lat"] - half and it.bbox[3] > t["lat"] + half]
            if not cands:
                continue
            it = cands[0]
            dst_t = from_origin(t["lon"] - half, t["lat"] + half, res, res)
            off = -1000.0 if (str(it.properties.get("s2:processing_baseline", "0")) >= "04.00" and
                              not it.properties.get("earthsearch:boa_offset_applied", False)) else 0.0
            bands = {}
            try:
                for b in ("red", "green", "blue", "swir22", "swir16"):
                    with rasterio.open(it.assets[b].href) as src:
                        dst = np.full((n, n), np.nan, "float32")
                        reproject(rasterio.band(src, 1), dst, dst_transform=dst_t, dst_crs="EPSG:4326",
                                  resampling=Resampling.bilinear, src_nodata=0, dst_nodata=np.nan)
                    bands[b] = (dst + off) / 10000.0
            except Exception as e:  # noqa: BLE001
                print("[skip chip]", t["model"], t["rank"], e)
                continue

            def st(a):
                v = a[np.isfinite(a)]
                if v.size == 0:
                    return np.zeros_like(a)
                p0, p1 = np.percentile(v, [1, 99.5])
                return np.clip((a - p0) / (p1 - p0 + 1e-9), 0, 1)

            tc = np.dstack([st(bands["red"]), st(bands["green"]), st(bands["blue"])])
            geo = np.dstack([st(bands["swir22"]), st(bands["swir16"]), st(bands["blue"])])
            for kind, img in (("tc", tc), ("geo", geo)):
                im = Image.fromarray((np.power(np.nan_to_num(img), 0.9) * 255).astype("uint8")).resize((360, 360), Image.LANCZOS)
                name = f"{t['model']}_{t['region']}_{t['rank']}_{kind}.jpg"
                im.save(folder / name, quality=80, optimize=True)
                t[f"chip_{kind}"] = f"chips/{name}"
            t["chip_date"] = it.datetime.date().isoformat()
            print("[chip]", t["model"], t["rank"], it.id)


# ----------------------------------------------------------------------------- images for the viewer
def pyramid(rgb, scores, inside, keep):
    """Two levels: overview (2000 px wide) and detail tiles (6000 px wide, 500 px tiles)."""
    from PIL import Image

    h, w = rgb.shape[:2]
    rgb = rgb.copy()
    rgb[~inside] = (rgb[~inside].astype("float32") * 0.35).astype("uint8")       # dim outside the area
    base = Image.fromarray(rgb)
    levels = {"ov": 2000, "dt": 6000}
    meta = {}
    for name, W in levels.items():
        W = min(W, w)
        H = round(h * W / w)
        im = base.resize((W, H), Image.LANCZOS)
        meta[name] = {"w": W, "h": H, "tile": 500, "have": {}}
        d = OUT / "tiles" / name
        d.mkdir(parents=True, exist_ok=True)
        if name == "ov":
            im.save(OUT / "tiles" / "ov.jpg", quality=82, optimize=True, progressive=True)
        else:
            for ty in range(0, H, 500):
                for tx in range(0, W, 500):
                    im.crop((tx, ty, min(tx + 500, W), min(ty + 500, H))).save(d / f"base_{ty // 500}_{tx // 500}.jpg", quality=80, optimize=True)
        for m, s in scores.items():
            v = s[keep][::5]
            lo, hi = np.percentile(v, 98.0), np.percentile(v, 99.97)
            f = max(1, int(round(w / W)))
            sm = ndimage.maximum_filter(s, size=f) if f > 1 else s
            x = np.clip((sm - lo) / (hi - lo + 1e-9), 0, 1)
            x[sm < lo] = -1
            rgba = colour(x)
            ov = Image.fromarray(rgba).resize((W, H), Image.NEAREST)
            if name == "ov":
                ov.save(OUT / "tiles" / f"ov_{m}.png", optimize=True)
            else:
                for ty in range(0, H, 500):
                    for tx in range(0, W, 500):
                        c = ov.crop((tx, ty, min(tx + 500, W), min(ty + 500, H)))
                        if c.getextrema()[3][1] == 0:
                            continue
                        c.save(d / f"{m}_{ty // 500}_{tx // 500}.png", optimize=True)
                        meta[name]["have"].setdefault(m, []).append(f"{ty // 500}_{tx // 500}")
    return meta


def colour(x):
    stops = np.array([[150, 40, 30, 150], [215, 100, 40, 200], [245, 175, 60, 235], [255, 240, 170, 255]], "float32")
    out = np.zeros(x.shape + (4,), "uint8")
    m = x >= 0
    t = np.clip(x[m], 0, 1) * (len(stops) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(stops) - 2)
    f = (t - i)[:, None]
    out[m] = (stops[i] * (1 - f) + stops[i + 1] * f).astype("uint8")
    return out


def geojson(all_t):
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [t["lon"], t["lat"]]},
              "properties": {"model": t["model"], "rank": t["rank"], "area_ha": t["area_ha"], "rel": t["rel"],
                             "region": t["region"], "where": t["where"]["fr"]}} for t in all_t]
    (OUT / "targets.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False))
    lines = ["model,rank,lat,lon,area_ha,strength_rel,region,where"]
    lines += [f"{t['model']},{t['rank']},{t['lat']},{t['lon']},{t['area_ha']},{t['rel']},{t['region']},\"{t['where']['fr']}\"" for t in all_t]
    (OUT / "targets.csv").write_text("\n".join(lines), encoding="utf-8")


def build_page(data):
    tpl = (ROOT / "minerals_sahara.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--page-only", action="store_true", help="rebuild index.html from sahara.json")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.page_only:
        build_page(json.loads((OUT / "sahara.json").read_text(encoding="utf-8")))
        return
    if a.demo:
        RES = 0.004
    import shutil
    for old in ("chips", "tiles"):            # remove pictures of the previous run
        shutil.rmtree(OUT / old, ignore_errors=True)
    G, valid, rgb, inside, land, used, by_id = scan(a.demo)
    keep, _ = outcrop(G, valid)
    scores, result, all_t = {}, {}, []
    for m, spec in MODELS.items():
        s = model_score(G, keep, spec["w"])
        tg, thr, _ = find_targets(s, keep, m)
        check, p_chance = blind(s, keep, tg, m)
        scores[m] = s
        result[m] = {"ar": spec["ar"], "fr": spec["fr"], "targets": tg, "check": check, "p_chance": p_chance}
        all_t += tg
        for c in check:
            print(f"[check] {m}: {c['fr']} percentile {c['percentile']} nearest target #{c['nearest_rank']} at {c['nearest_km']} km")
    chip_list = [t for m in MODELS for reg in ("dakhla", "laayoune")
                 for t in [q for q in result[m]["targets"] if q["region"] == reg][:N_CHIPS]]
    chips(chip_list, by_id, used, a.demo)
    meta = pyramid(rgb, scores, inside, keep)
    geojson(all_t)
    area_km2 = float(land.sum()) * (RES * 111.32) ** 2 * math.cos(math.radians(24))
    data = {
        "made": date.today().isoformat(), "demo": a.demo, "box": BOX, "res_m": round(RES * 111320),
        "outline": OUTLINE, "split_lat": REGION_SPLIT_LAT, "towns": TOWNS,
        "area_km2": round(area_km2, -2), "rock_pct": round(100 * float(keep.sum()) / max(float(valid.sum()), 1), 1),
        "valid_pct": round(100 * float(valid.sum()) / max(float(land.sum()), 1), 1),
        "models": result, "tiles": meta, "scenes": used, "check_km": CHECK_KM,
    }
    (OUT / "sahara.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    build_page(data)
    print(f"[done] {len(all_t)} targets, {len(used)} Sentinel-2 passes")


if __name__ == "__main__":
    main()
