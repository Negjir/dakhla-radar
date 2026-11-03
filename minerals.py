"""
MADAR Minerals: proof of concept on known mines.

For each site, reads cloud-free Sentinel-2 L2A images (ESA Copernicus, free, via the public
AWS Earth Search catalogue), computes classic mineral ratio indices and ranks "target" areas.
Then a BLIND TEST: the known mine coordinates are NOT used in the analysis; afterwards we
measure how far the ranked targets are from the known mine.

Indices (Sentinel-2 band ratios, standard in remote-sensing geology):
  clay / hydroxyl (argillic alteration)  B11 / B12   SWIR 1.6 um / 2.2 um
  ferric iron oxide (gossan, hematite)   B04 / B02   red / blue
  ferrous iron                           B12 / B8A + B03 / B04

Honest limits (also printed on the page):
  - This narrows WHERE to look first; it does not prove a deposit. Field sampling and drilling decide.
  - Active mines light up partly because of their own pits and tailings, so a hit on a working mine
    is a weaker proof than a hit on a known occurrence that was never mined. Both are reported.
  - Sand dunes, vegetation, clouds and water are masked or unreliable.

Usage:  python minerals.py            (real data, needs network: run in GitHub Actions)
        python minerals.py --demo     (synthetic scene, to test the pipeline and the page)
"""
import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "minerals"
STAC = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
RES = 0.0002  # degrees, ~20 m (native resolution of the SWIR bands)
BANDS = ["blue", "green", "red", "nir", "nir08", "swir16", "swir22", "scl"]

# Known points are used ONLY after the analysis, to score the blind test.
# kind: "mine" = worked deposit (pits/tailings visible), "occurrence" = known mineralisation.
SITES = [
    {
        "id": "imiter", "name": "Imiter", "name_ar": "إميضر", "country": "Morocco",
        "metal": "Silver (Ag)", "metal_ar": "الفضة",
        "box": [31.24, 31.46, -5.98, -5.62],  # lat0, lat1, lon0, lon1
        "profile": "alteration",
        "known": [
            {"name": "Imiter silver mine (plant and ponds)", "lat": 31.3495, "lon": -5.7234, "kind": "mine",
             "src": "Imider village, Wikipedia 31.350N 5.725W; mine ponds visible in Sentinel-2"},
        ],
    },
    {
        "id": "bouazzer", "name": "Bou Azzer", "name_ar": "بوعزر", "country": "Morocco",
        "metal": "Cobalt (Co), nickel, gold", "metal_ar": "الكوبالت والنيكل والذهب",
        "box": [30.38, 30.62, -6.98, -6.50],
        "profile": "alteration",
        "known": [
            {"name": "Vein No. 61, Ait Ahmane (Bou Azzer district)", "lat": 30.4806, "lon": -6.5878,
             "kind": "occurrence", "src": "mindat.org loc-212416"},
        ],
    },
    {
        "id": "kediet", "name": "Kediet ej Jill (Zouerat)", "name_ar": "كدية الجل (الزويرات)",
        "country": "Mauritania", "metal": "Iron ore (Fe)", "metal_ar": "الحديد",
        "box": [22.52, 22.78, -12.75, -12.40],
        "profile": "iron",
        "known": [
            {"name": "Rouessa iron mine (SNIM)", "lat": 22.6667, "lon": -12.6667, "kind": "mine",
             "src": "USGS major deposits via GEM.wiki (rounded, may be ~1-2 km off)"},
        ],
    },
]

PROFILES = {
    # weights of robust z-scores in the prospectivity score
    "alteration": {"clay": 0.5, "ferric": 0.5, "ferrous": 0.0},
    "iron": {"clay": 0.0, "ferric": 0.5, "ferrous": 0.5},
}
TOP_PERCENT = 1.0      # pixels kept as anomalous
MIN_PIXELS = 6         # smallest target (~0.25 ha... ~ 6 x 400 m2)
N_TARGETS = 10
HIT_KM = 2.0


# ----------------------------------------------------------------------------- data
def grid(box):
    lat0, lat1, lon0, lon1 = box
    w = int(round((lon1 - lon0) / RES))
    h = int(round((lat1 - lat0) / RES))
    return w, h


def fetch(site, max_cloud=5, years=2, max_items=6):
    """Cloud-free mosaic on a lat/lon grid. Returns dict band -> float32 (h, w), reflectance 0..1."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
    from pystac_client import Client

    lat0, lat1, lon0, lon1 = site["box"]
    w, h = grid(site["box"])
    dst_t = from_bounds(lon0, lat0, lon1, lat1, w, h)
    end = date.today()
    start = end - timedelta(days=365 * years)
    cat = Client.open(STAC)
    items = list(cat.search(collections=[COLLECTION], bbox=[lon0, lat0, lon1, lat1],
                            datetime=f"{start}/{end}", query={"eo:cloud_cover": {"lt": max_cloud}},
                            max_items=200).items())
    if not items:
        raise RuntimeError("no Sentinel-2 scene under %d%% cloud" % max_cloud)
    # Prefer dry-season, low-cloud, recent scenes; several tiles may be needed to cover the box.
    # Use whole acquisition dates (all tiles of one satellite pass) so the mosaic has no seams
    # between different days; best dates = cover the box, no cloud, dry season light.
    def frac(it):
        bx0, by0, bx1, by1 = it.bbox
        ix = max(0.0, min(bx1, lon1) - max(bx0, lon0))
        iy = max(0.0, min(by1, lat1) - max(by0, lat0))
        return ix * iy / ((lon1 - lon0) * (lat1 - lat0))
    by_day = {}
    for it in items:
        by_day.setdefault(it.datetime.date(), []).append(it)
    days = sorted(by_day, key=lambda d: (-min(1.0, sum(frac(i) for i in by_day[d])),
                                         max(i.properties.get("eo:cloud_cover", 100) for i in by_day[d]),
                                         -d.toordinal()))
    items = [it for d in days for it in sorted(by_day[d], key=lambda i: -frac(i))]
    out = {b: np.full((h, w), np.nan, "float32") for b in BANDS}
    used = []
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                       GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")
    with env:
        for it in items:
            if not np.isnan(out["red"]).any():
                break
            if len(used) >= max_items:
                break
            offset = 0.0
            pb = str(it.properties.get("s2:processing_baseline", "00.00"))
            if pb >= "04.00" and not it.properties.get("earthsearch:boa_offset_applied", False):
                offset = -1000.0
            try:
                tile = {}
                for b in BANDS:
                    href = it.assets[b].href
                    with rasterio.open(href) as src:
                        rs = Resampling.nearest if b == "scl" else Resampling.bilinear
                        with WarpedVRT(src, crs="EPSG:4326", transform=dst_t, width=w, height=h,
                                       resampling=rs, nodata=0) as vrt:
                            a = vrt.read(1).astype("float32")
                    a[a == 0] = np.nan
                    if b != "scl":
                        a = (a + offset) / 10000.0
                    tile[b] = a
            except Exception as e:  # noqa: BLE001
                print("[skip]", it.id, e)
                continue
            scl = tile["scl"]
            good = np.isin(scl, [2, 4, 5, 7, 11])  # dark area, vegetation, bare soil, unclassified, snow? (11 kept: bright desert often flagged)
            good &= ~np.isnan(tile["red"])
            fill = good & np.isnan(out["red"])
            if fill.sum() == 0:
                continue
            for b in BANDS:
                out[b][fill] = tile[b][fill]
            used.append({"id": it.id, "date": it.datetime.date().isoformat(),
                         "cloud": round(float(it.properties.get("eo:cloud_cover", 0)), 2)})
            print("[ok] %s  %s  cloud %.1f%%  filled %.1f%%" % (it.id, it.datetime.date(),
                  it.properties.get("eo:cloud_cover", 0), 100 * fill.mean()))
    return out, used


def demo_scene(site, seed=1):
    """Synthetic desert with a few alteration halos; one of them on the known point."""
    rng = np.random.default_rng(seed)
    w, h = grid(site["box"])
    lat0, lat1, lon0, lon1 = site["box"]
    yy, xx = np.mgrid[0:h, 0:w]
    base = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 25)
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    b = {
        "blue": 0.12 + 0.05 * base, "green": 0.17 + 0.06 * base, "red": 0.22 + 0.08 * base,
        "nir": 0.27 + 0.08 * base, "nir08": 0.28 + 0.08 * base,
        "swir16": 0.36 + 0.08 * base, "swir22": 0.30 + 0.07 * base,
    }
    k = site["known"][0]
    centres = [((lat1 - k["lat"]) / RES + 8, (k["lon"] - lon0) / RES - 5)]
    for _ in range(5):
        centres.append((rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w))
    for n, (cy, cx) in enumerate(centres):
        r = rng.uniform(15, 35)
        g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
        s = 1.0 if n == 0 else rng.uniform(0.4, 0.9)
        b["swir22"] = b["swir22"] - 0.05 * s * g        # clay absorption at 2.2 um
        b["red"] = b["red"] + 0.06 * s * g              # hematite
        b["blue"] = b["blue"] - 0.02 * s * g
    for v in b.values():
        v += rng.normal(0, 0.004, v.shape)
    b = {kk: v.astype("float32") for kk, v in b.items()}
    b["scl"] = np.full((h, w), 5, "float32")
    return b, [{"id": "DEMO", "date": date.today().isoformat(), "cloud": 0}]


# ----------------------------------------------------------------------------- analysis
def robust_z(a, mask):
    v = a[mask]
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826 + 1e-9
    z = (a - med) / mad
    z[~mask] = np.nan
    return z


def analyse(b, profile):
    eps = 1e-4
    ndvi = (b["nir"] - b["red"]) / (b["nir"] + b["red"] + eps)
    bright = (b["blue"] + b["green"] + b["red"]) / 3
    valid = ~np.isnan(b["red"]) & ~np.isnan(b["swir22"])
    valid &= ndvi < 0.25                      # vegetation hides the rock
    valid &= b["swir22"] > 0.03               # water / deep shadow
    valid &= bright < 0.45                    # saturated / cloud remnants
    idx = {
        "clay": b["swir16"] / (b["swir22"] + eps),
        "ferric": b["red"] / (b["blue"] + eps),
        "ferrous": b["swir22"] / (b["nir08"] + eps) + b["green"] / (b["red"] + eps),
    }
    z = {k: robust_z(v, valid) for k, v in idx.items()}
    wts = PROFILES[profile]
    score = sum(np.clip(np.nan_to_num(z[k]), 0, 8) * wt for k, wt in wts.items() if wt)
    score = ndimage.gaussian_filter(score, 1.5)
    score[~valid] = 0
    return idx, z, score, valid


def targets(score, valid, box):
    lat0, lat1, lon0, lon1 = box
    thr = np.percentile(score[valid], 100 - TOP_PERCENT)
    anom = (score >= thr) & valid
    lab, n = ndimage.label(anom, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        m = lab == i
        npx = int(m.sum())
        if npx < MIN_PIXELS:
            continue
        ys, xs = np.nonzero(m)
        wgt = score[m]
        cy = float((ys * wgt).sum() / wgt.sum())
        cx = float((xs * wgt).sum() / wgt.sum())
        out.append({
            "lat": round(lat1 - (cy + 0.5) * RES, 5), "lon": round(lon0 + (cx + 0.5) * RES, 5),
            "area_ha": round(npx * px_area_m2(box) / 1e4, 1),
            "strength": float(wgt.sum()), "peak": round(float(wgt.max()), 2),
        })
    out.sort(key=lambda t: -t["strength"])
    out = out[:N_TARGETS]
    if out:
        top = out[0]["strength"]
        for r, t in enumerate(out):
            t["rank"] = r + 1
            t["label"] = "T%d" % (r + 1)
            t["rel"] = round(100 * t["strength"] / top)
            del t["strength"]
    return out, float(anom.sum()) * px_area_m2(box) / 1e6, thr


def px_area_m2(box):
    lat = (box[0] + box[1]) / 2
    return (RES * 111320) * (RES * 111320 * math.cos(math.radians(lat)))


def km(a_lat, a_lon, b_lat, b_lon):
    p = math.pi / 180
    d = (math.sin((b_lat - a_lat) * p / 2) ** 2
         + math.cos(a_lat * p) * math.cos(b_lat * p) * math.sin((b_lon - a_lon) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(d))


def blind_test(site, tg, score, valid):
    lat0, lat1, lon0, lon1 = site["box"]
    res = []
    vals = score[valid]
    for k in site["known"]:
        dists = [(km(k["lat"], k["lon"], t["lat"], t["lon"]), t) for t in tg]
        dists.sort(key=lambda x: x[0])
        nearest = dists[0] if dists else (None, None)
        hit = [t for d, t in dists if d <= HIT_KM]
        j = int((lat1 - k["lat"]) / RES)
        i = int((k["lon"] - lon0) / RES)
        r = 12  # ~250 m window around the point
        win = score[max(j - r, 0):j + r + 1, max(i - r, 0):i + r + 1]
        pct = float((vals < win.max()).mean() * 100) if win.size else None
        res.append({
            "name": k["name"], "lat": k["lat"], "lon": k["lon"], "kind": k["kind"], "src": k["src"],
            "nearest_km": round(nearest[0], 2) if nearest[0] is not None else None,
            "nearest_target": nearest[1]["label"] if nearest[1] else None,
            "best_rank_within": min([t["rank"] for t in hit]) if hit else None,
            "percentile": round(pct, 1) if pct is not None else None,
            "hit": bool(hit),
        })
    return res


# ----------------------------------------------------------------------------- images
def stretch(a, valid, lo=2, hi=98):
    v = a[valid & np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a)
    p0, p1 = np.percentile(v, [lo, hi])
    return np.clip((a - p0) / (p1 - p0 + 1e-9), 0, 1)


def save_images(site, b, idx, score, valid, folder, scale=2):
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    ok = ~np.isnan(b["red"])

    def small(img):
        h, w = img.shape[:2]
        return img[::scale, ::scale] if scale > 1 else img

    rgb = np.dstack([stretch(np.nan_to_num(b[k]), ok, 1, 99.5) for k in ("red", "green", "blue")])
    rgb = (np.power(rgb, 0.85) * 255).astype("uint8")
    Image.fromarray(rgb).save(folder / "truecolor.jpg", quality=84)

    comp = np.dstack([stretch(np.nan_to_num(idx["clay"]), valid),
                      stretch(np.nan_to_num(idx["ferric"]), valid),
                      stretch(np.nan_to_num(idx["ferrous"]), valid)])
    comp[~valid] = 0.08
    Image.fromarray(small((comp * 255).astype("uint8"))).save(folder / "ratios.jpg", quality=88)

    for name, key in (("clay", "clay"), ("iron", "ferric")):
        s = stretch(np.nan_to_num(idx[key]), valid, 50, 99.7)
        Image.fromarray(small(heat_rgba(s, valid))).save(folder / f"{name}.png", optimize=True)

    s = score / (np.percentile(score[valid], 99.8) + 1e-9)
    s = np.clip((s - 0.35) / 0.65, 0, 1)
    Image.fromarray(heat_rgba(s, valid)).save(folder / "score.png", optimize=True)


def heat_rgba(s, valid):
    """0 -> transparent, rising through amber to hot white-orange."""
    stops = np.array([[0, 0, 0, 0], [180, 60, 20, 110], [255, 140, 0, 190], [255, 210, 80, 230],
                      [255, 255, 220, 255]], "float32")
    x = np.clip(s, 0, 1) * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    rgba = stops[i] * (1 - f) + stops[i + 1] * f
    rgba[~valid] = 0
    return rgba.astype("uint8")


# ----------------------------------------------------------------------------- main
def run_site(site, demo):
    print("==", site["name"])
    if demo:
        b, used = demo_scene(site)
    else:
        b, used = fetch(site)
    idx, z, score, valid = analyse(b, site["profile"])
    tg, anom_km2, thr = targets(score, valid, site["box"])
    bt = blind_test(site, tg, score, valid)
    lat0, lat1, lon0, lon1 = site["box"]
    w, h = grid(site["box"])
    total_km2 = w * h * px_area_m2(site["box"]) / 1e6
    folder = OUT / site["id"]
    save_images(site, b, idx, score, valid, folder)
    res = {
        "id": site["id"], "name": site["name"], "name_ar": site["name_ar"], "country": site["country"],
        "metal": site["metal"], "metal_ar": site["metal_ar"], "profile": site["profile"],
        "bounds": [[lat0, lon0], [lat1, lon1]],
        "area_km2": round(total_km2), "valid_pct": round(100 * float(valid.mean()), 1),
        "anomaly_km2": round(anom_km2, 2), "reduction": round(total_km2 / max(anom_km2, 1e-6)),
        "targets": tg, "blind_test": bt, "scenes": used,
        "images": {k: f"{site['id']}/{k}" for k in
                   ("truecolor.jpg", "ratios.jpg", "clay.png", "iron.png", "score.png")},
    }
    for k in bt:
        print("   blind test: %s -> nearest %s at %s km, within %g km: %s, percentile %s" % (
            k["name"], k["nearest_target"], k["nearest_km"], HIT_KM, k["hit"], k["percentile"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--site", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for s in SITES:
        if a.site and s["id"] != a.site:
            continue
        try:
            results.append(run_site(s, a.demo))
        except Exception as e:  # noqa: BLE001
            print("[fail]", s["id"], e)
    if not results:
        sys.exit("no site processed")
    data = {"generated": date.today().isoformat(), "demo": a.demo, "hit_km": HIT_KM,
            "top_percent": TOP_PERCENT, "source": "ESA Copernicus Sentinel-2 L2A (Earth Search, AWS)",
            "sites": results}
    (OUT / "minerals.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    page = (ROOT / "minerals_console.html").read_text(encoding="utf-8")
    page = page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print("[ok] docs/minerals/index.html")


if __name__ == "__main__":
    main()
