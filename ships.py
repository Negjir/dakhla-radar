"""Madar Offshore - vessels seen from space (Sentinel-1 radar), with or without AIS.

Radar sees through cloud and at night, and a steel hull shines on the dark sea whether its AIS transponder is on or
off. For each Sentinel-1 pass over the Dakhla fishing grounds this finds the bright points on the sea (a CFAR test
against the sea around them), estimates their size, drops what stays in place across passes (platforms, buoys,
rocks, oyster tables) and writes docs/ships/ships.json plus a small image of each vessel.

Images: Copernicus Sentinel-1 GRD, read from Microsoft Planetary Computer (free, anonymous signed URLs).
Land: Copernicus DEM GLO-30 (AWS open data).

    python ships.py --start 2026-10-15 --end 2026-11-10
    python ships.py --demo        # synthetic scene, to test the detector offline
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "ships"
BOX = (20.9, 26.6, -18.7, -14.3)          # lat0, lat1, lon0, lon1: Madar Offshore grounds
PORT = (23.705, -15.925)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd"
DEM = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
LAND_RES = 0.002                           # land mask cell (~200 m)
LAND_BUFFER = 3                            # cells added around land (~600 m): rocks and surf are not ships
DEC = 2                                    # read the 10 m GRD at 20 m
BLOCK = 2048
GUARD, BACK = 9, 51                        # CFAR windows in pixels (20 m): guard ~180 m, background ~1 km
K_SIGMA, K_RATIO = 7.0, 10.0               # a ship must beat the sea by both tests
MIN_PX = 2
PX_M = 10.0 * DEC
FIXED_SCENES = 3                           # seen at the same spot on this many passes = not a vessel
CHIP = 40                                  # half-size of each vessel image, pixels
MAX_CHIPS = 400


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- land mask
def tile_name(lat, lon):
    return (f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")


def land_mask():
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
    lat0, lat1, lon0, lon1 = BOX
    w, h = round((lon1 - lon0) / LAND_RES), round((lat1 - lat0) / LAND_RES)
    tr = from_bounds(lon0, lat0, lon1, lat1, w, h)
    land = np.zeros((h, w), bool)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        for la in range(math.floor(lat0), math.floor(lat1) + 1):
            for lo in range(math.floor(lon0), math.floor(lon1) + 1):
                try:
                    with rasterio.open(DEM.format(n=tile_name(la, lo))) as src, WarpedVRT(
                            src, crs="EPSG:4326", transform=tr, width=w, height=h,
                            resampling=Resampling.max, nodata=-9999) as v:
                        a = v.read(1)
                except Exception:  # noqa: BLE001 - no tile = open sea
                    continue
                land |= a > 0.5
                log("[land]", tile_name(la, lo))
    land = ndimage.binary_dilation(land, iterations=LAND_BUFFER)
    return land, (lat1, lon0, LAND_RES)      # row 0 = north


def on_land(land, geo, lat, lon):
    lat_top, lon0, r = geo
    i = np.clip(((lat_top - lat) / r).astype(int), 0, land.shape[0] - 1)
    j = np.clip(((lon - lon0) / r).astype(int), 0, land.shape[1] - 1)
    out = land[i, j]
    outside = (lat < BOX[0]) | (lat > BOX[1]) | (lon < BOX[2]) | (lon > BOX[3])
    return out | outside


# ---------------------------------------------------------------- scenes
def http_json(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    for k in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            log("[retry]", url, e)
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"no answer from {url}")


def scenes(start, end):
    lat0, lat1, lon0, lon1 = BOX
    body = {"collections": ["sentinel-1-grd"], "bbox": [lon0, lat0, lon1, lat1],
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
            "query": {"sar:instrument_mode": {"eq": "IW"}}}
    items, url = [], STAC
    while url:
        r = http_json(url, body)
        items += r.get("features", [])
        nxt = next((l for l in r.get("links", []) if l.get("rel") == "next"), None)
        url, body = (nxt["href"], nxt.get("body", body)) if nxt else (None, None)
    items.sort(key=lambda i: i["properties"]["datetime"])
    log(f"[stac] {len(items)} Sentinel-1 IW scenes {start} → {end}")
    return items


_token = {"t": None, "at": 0}


def signed(href):
    if not _token["t"] or time.time() - _token["at"] > 1800:
        _token["t"], _token["at"] = http_json(SAS)["token"], time.time()
    return href + ("&" if "?" in href else "?") + _token["t"]


# ---------------------------------------------------------------- detection
def cfar(I, sea):
    """I: intensity, sea: valid sea pixels. Returns mask of pixels much brighter than the sea around them."""
    v = sea.astype("float32")
    Iv = np.where(sea, I, 0).astype("float32")

    def box(a, n):
        return ndimage.uniform_filter(a, n, mode="constant") * n * n
    n_b, n_g = box(v, BACK) - box(v, GUARD), None
    s1 = box(Iv, BACK) - box(Iv, GUARD)
    s2 = box(Iv * Iv, BACK) - box(Iv * Iv, GUARD)
    n_b = np.maximum(n_b, 1)
    mu = s1 / n_b
    sd = np.sqrt(np.maximum(s2 / n_b - mu * mu, 0))
    enough = n_b > 0.5 * (BACK * BACK - GUARD * GUARD)
    return sea & enough & (I > mu + K_SIGMA * sd) & (I > K_RATIO * mu), mu


def objects(hit, I, mu):
    lab, n = ndimage.label(hit, structure=np.ones((3, 3)))
    out = []
    if not n:
        return out
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(I), lab, idx)
    for k, sl in enumerate(ndimage.find_objects(lab)):
        if size[k] < MIN_PX:
            continue
        m = lab[sl] == k + 1
        yy, xx = np.nonzero(m)
        w = I[sl][m]
        cy, cx = (yy * w).sum() / w.sum() + sl[0].start, (xx * w).sum() / w.sum() + sl[1].start
        if len(yy) > 2:          # long axis from the second moments of the bright pixels
            c = np.cov(np.vstack([yy, xx]))
            length = 4 * math.sqrt(max(np.linalg.eigvalsh(c)[-1], 0.25)) * PX_M
        else:
            length = len(yy) * PX_M
        peak = float(w.max())
        out.append({"y": float(cy), "x": float(cx), "px": int(size[k]), "len_m": round(length),
                    "snr_db": round(10 * math.log10(peak / max(float(mu[int(cy), int(cx)]), 1e-6)), 1)})
    return out


class Geo:
    """Pixel/line -> lat/lon from the scene's ground control points."""

    def __init__(self, gcps):
        from scipy.interpolate import LinearNDInterpolator
        p = np.array([[g.row, g.col] for g in gcps])
        self.lon = LinearNDInterpolator(p, [g.x for g in gcps])
        self.lat = LinearNDInterpolator(p, [g.y for g in gcps])
        self.nlon = __import__("scipy.interpolate", fromlist=["NearestNDInterpolator"]).NearestNDInterpolator(
            p, [g.x for g in gcps])
        self.nlat = __import__("scipy.interpolate", fromlist=["NearestNDInterpolator"]).NearestNDInterpolator(
            p, [g.y for g in gcps])

    def __call__(self, rows, cols):
        la, lo = self.lat(rows, cols), self.lon(rows, cols)
        bad = ~np.isfinite(la)
        if bad.any():
            la[bad], lo[bad] = self.nlat(rows[bad], cols[bad]), self.nlon(rows[bad], cols[bad])
        return la, lo


def scene_detect(item, land, geo_land, demo=None):
    import rasterio
    from rasterio.windows import Window
    a = item["assets"]
    vv = a.get("vv") or a.get("hh")
    vh = a.get("vh") or a.get("hv")
    if not vv:
        return [], None
    found, chips = [], []
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                       GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE")
    with env, rasterio.open(signed(vv["href"])) as src:
        gcps, _ = src.gcps
        geo = Geo(gcps)
        H, W = src.height, src.width
        src_vh = rasterio.open(signed(vh["href"])) if vh else None
        pad = BACK * DEC
        for r0 in range(0, H, BLOCK * DEC):
            for c0 in range(0, W, BLOCK * DEC):
                rr = np.array([r0, r0, min(H, r0 + BLOCK * DEC), min(H, r0 + BLOCK * DEC)], float)
                cc = np.array([c0, min(W, c0 + BLOCK * DEC), c0, min(W, c0 + BLOCK * DEC)], float)
                la, lo = geo(rr, cc)
                gy = np.linspace(rr.min(), rr.max(), 9)
                gx = np.linspace(cc.min(), cc.max(), 9)
                GY, GX = np.meshgrid(gy, gx, indexing="ij")
                gla, glo = geo(GY.ravel(), GX.ravel())
                if on_land(land, geo_land, gla, glo).all():
                    continue
                ra, ca = max(0, r0 - pad), max(0, c0 - pad)
                rb, cb = min(H, r0 + BLOCK * DEC + pad), min(W, c0 + BLOCK * DEC + pad)
                win = Window(ca, ra, cb - ca, rb - ra)
                shp = ((rb - ra) // DEC, (cb - ca) // DEC)
                try:
                    d = src.read(1, window=win, out_shape=shp).astype("float32")
                except Exception as e:  # noqa: BLE001
                    log("[read fail]", item["id"], e)
                    continue
                I = d * d
                # land, swath edge and no data
                sy = np.linspace(ra, rb, 33)
                sx = np.linspace(ca, cb, 33)
                SY, SX = np.meshgrid(sy, sx, indexing="ij")
                sla, slo = geo(SY.ravel(), SX.ravel())
                sla, slo = sla.reshape(33, 33), slo.reshape(33, 33)
                zy, zx = shp[0] / 33, shp[1] / 33
                fla = ndimage.zoom(sla, (zy, zx), order=1)[:shp[0], :shp[1]]
                flo = ndimage.zoom(slo, (zy, zx), order=1)[:shp[0], :shp[1]]
                fla = np.pad(fla, ((0, shp[0] - fla.shape[0]), (0, shp[1] - fla.shape[1])), mode="edge")
                flo = np.pad(flo, ((0, shp[0] - flo.shape[0]), (0, shp[1] - flo.shape[1])), mode="edge")
                sea = (d > 0) & ~on_land(land, geo_land, fla, flo)
                sea = ndimage.binary_erosion(sea, iterations=3)
                if sea.sum() < 1000:
                    continue
                hit, mu = cfar(I, sea)
                obs = objects(hit, I, mu)
                if obs and src_vh is not None:      # cross-pol check: a hull is bright in VH too, a wave crest rarely
                    try:
                        dvh = src_vh.read(1, window=win, out_shape=shp).astype("float32")
                        Ivh = dvh * dvh
                        mvh = ndimage.median_filter(Ivh[::4, ::4], 15)
                        for o in obs:
                            y, x = int(o["y"]), int(o["x"])
                            loc = Ivh[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()
                            o["vh_ratio"] = round(float(loc / max(mvh[min(y // 4, mvh.shape[0] - 1),
                                                                     min(x // 4, mvh.shape[1] - 1)], 1e-6)), 1)
                    except Exception as e:  # noqa: BLE001
                        log("[vh fail]", e)
                for o in obs:
                    fy, fx = ra + o["y"] * DEC, ca + o["x"] * DEC
                    if not (r0 <= fy < r0 + BLOCK * DEC and c0 <= fx < c0 + BLOCK * DEC):
                        continue                    # belongs to the next block
                    la1, lo1 = geo(np.array([fy]), np.array([fx]))
                    o["lat"], o["lon"] = round(float(la1[0]), 5), round(float(lo1[0]), 5)
                    if on_land(land, geo_land, la1, lo1)[0]:
                        continue
                    if o.get("vh_ratio") is not None and o["vh_ratio"] < 3:
                        continue
                    y, x = int(o["y"]), int(o["x"])
                    o["chip"] = d[max(0, y - CHIP):y + CHIP, max(0, x - CHIP):x + CHIP].copy()
                    for k in ("y", "x"):
                        o.pop(k)
                    found.append(o)
        if src_vh is not None:
            src_vh.close()
    return found, None


# ---------------------------------------------------------------- after all scenes
def mark_fixed(dets):
    """Anything seen within ~250 m on several different passes does not move: not a vessel."""
    cell = 0.0025
    seen = {}
    for d in dets:
        key = (round(d["lat"] / cell), round(d["lon"] / cell))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                seen.setdefault((key[0] + dy, key[1] + dx), set()).add(d["scene"])
    for d in dets:
        key = (round(d["lat"] / cell), round(d["lon"] / cell))
        d["fixed"] = len(seen[key]) >= FIXED_SCENES
        d["port"] = math.dist((d["lat"], d["lon"]), PORT) < 0.03


def save_chip(a, path):
    from PIL import Image
    a = np.asarray(a, "float32")
    a = 10 * np.log10(np.maximum(a * a, 1))
    lo, hi = np.percentile(a, 2), max(np.percentile(a, 99.8), np.percentile(a, 2) + 1)
    g = np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
    Image.fromarray(g).resize((g.shape[1] * 2, g.shape[0] * 2), Image.NEAREST).save(path, quality=85)


def land_png(land, path):
    from PIL import Image
    small = land[::4, ::4]
    img = np.zeros(small.shape + (4,), "uint8")
    img[small] = (38, 34, 28, 255)
    Image.fromarray(img, "RGBA").save(path, optimize=True)


def footprint(item):
    g = item.get("geometry") or {}
    c = g.get("coordinates") or []
    ring = c[0] if g.get("type") == "Polygon" else (c[0][0] if c else [])
    return [[round(p[1], 3), round(p[0], 3)] for p in ring[::max(1, len(ring) // 12)]]


def demo_items():
    """A fake scene object understood by scene_detect_demo."""
    return [{"id": f"DEMO_{k}", "properties": {"datetime": f"2026-10-{15 + 3 * k:02d}T19:00:00Z",
                                                 "platform": "sentinel-1a", "sat:orbit_state": "ascending"},
             "geometry": {"type": "Polygon", "coordinates": [[[-18.5, 22.0], [-15.0, 22.0], [-15.0, 25.0],
                                                               [-18.5, 25.0], [-18.5, 22.0]]]}} for k in range(4)]


def demo_detect(item, k):
    """Speckled sea with a few ships and one platform that never moves."""
    rnd = np.random.default_rng(k)
    n = 1200
    sea = rnd.gamma(4.0, 25 / 4.0, (n, n)).astype("float32")      # intensity, 4-look speckle
    truth = [(300 + 60 * k, 400, 3), (800, 900 - 40 * k, 2), (600, 600, 2)]    # last one = platform
    for y, x, r in truth:
        sea[y - r:y + r + 1, x - r:x + r + 1] *= 400
    hit, mu = cfar(sea, np.ones_like(sea, bool))
    obs = objects(hit, sea, mu)
    for o in obs:
        o["lat"] = round(25.0 - o["y"] / n * 3, 5)
        o["lon"] = round(-18.5 + o["x"] / n * 3.5, 5)
        y, x = int(o["y"]), int(o["x"])
        o["chip"] = np.sqrt(sea[max(0, y - CHIP):y + CHIP, max(0, x - CHIP):x + CHIP])
        o.pop("y"), o.pop("x")
    return obs


def main():
    ap = argparse.ArgumentParser()
    today = dt.date.today()
    ap.add_argument("--start", default=(today - dt.timedelta(days=10)).isoformat())
    ap.add_argument("--end", default=today.isoformat())
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chips").mkdir(exist_ok=True)
    old = json.loads((OUT / "ships.json").read_text()) if (OUT / "ships.json").exists() and not a.demo else {}
    done = {s["id"] for s in old.get("scenes", [])}
    dets, scs = list(old.get("detections", [])), list(old.get("scenes", []))
    for d in dets:
        d.pop("fixed", None)
    if a.demo:
        items = demo_items()
        land = None
        from PIL import Image
        m = np.zeros((570, 440, 4), "uint8")
        yy, xx = np.mgrid[0:570, 0:440]
        m[xx > 330 + (yy - 285) * 0.25] = (38, 34, 28, 255)
        Image.fromarray(m, "RGBA").save(OUT / "land.png")
    else:
        items = [i for i in scenes(a.start, a.end) if i["id"] not in done]
        land, geo_land = land_mask()
        land_png(land, OUT / "land.png")
    for k, it in enumerate(items):
        t0 = time.time()
        p = it["properties"]
        try:
            found = demo_detect(it, k) if a.demo else scene_detect(it, land, geo_land)[0]
        except Exception as e:  # noqa: BLE001
            log("[scene fail]", it["id"], e)
            continue
        sid = it["id"]
        for n, o in enumerate(found):
            o["scene"] = sid
            o["t"] = p["datetime"][:16].replace("T", " ")
            o["id"] = f"{sid[-6:]}_{n}"
            o["_chip"] = o.pop("chip", None)
        dets += found
        scs.append({"id": sid, "t": p["datetime"][:16].replace("T", " "), "sat": p.get("platform", ""),
                    "orbit": p.get("sat:orbit_state", ""), "fp": footprint(it), "n": len(found)})
        log(f"[scene] {sid} {p['datetime'][:16]}: {len(found)} bright objects on the sea · {time.time() - t0:.0f} s")
    mark_fixed(dets)
    moving = sorted((d for d in dets if not d["fixed"] and "_chip" in d and d["_chip"] is not None),
                    key=lambda d: -d["len_m"])[:MAX_CHIPS]
    for d in moving:
        save_chip(d["_chip"], OUT / "chips" / f"{d['id']}.jpg")
        d["img"] = True
    for d in dets:
        d.pop("_chip", None)
    lat0, lat1, lon0, lon1 = BOX
    data = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "demo": a.demo,
            "box": BOX, "port": PORT, "scenes": sorted(scs, key=lambda s: s["t"]),
            "detections": sorted(dets, key=lambda d: d["t"]),
            "method": {"px_m": PX_M, "k_sigma": K_SIGMA, "k_ratio": K_RATIO, "fixed_scenes": FIXED_SCENES,
                       "land_buffer_m": LAND_BUFFER * 220},
            "sources": {"images": "Copernicus Sentinel-1 GRD (ESA), via Microsoft Planetary Computer",
                        "land": "Copernicus DEM GLO-30 (ESA), AWS open data"}}
    (OUT / "index.html").write_text((ROOT / "ships_console.html").read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "ships.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    mv = [d for d in dets if not d["fixed"]]
    log(f"[done] {len(scs)} passes · {len(dets)} objects · {len(mv)} moving (vessels) · "
        f"{sum(1 for d in dets if d['fixed'])} fixed")


if __name__ == "__main__":
    sys.exit(main())
