"""MADAR Minerals - gold model for the whole Sahara survey, from every source that sees part of the story.

    EMIT (NASA, 285 bands)       alteration minerals left by hot fluids: sericite, alunite, chlorite, carbonate, jarosite
    Sentinel-2 (ESA, 13 bands)   clays and iron oxides, a second and independent look at the same alteration
    ASTER (NASA/METI, thermal)   silica: quartz veins and silicified rock, which hold the gold in most deposits
    Copernicus DEM (90 m)        faults and fractures, the paths the fluids took

Each source becomes an anomaly against the rocks around it (~30 km). A place becomes a gold target only where at
least two independent sources agree, and each target says which ones do.

    python minerals_gold.py

Reads: docs/minerals/sahara/sahara.json, data/cache/sahara_emit.npz, data/cache/sahara_s2.npz (if present).
Downloads: ASTER Global Emissivity Dataset AG100 v003 (Earthdata login), Copernicus GLO-90 (AWS open data).
"""
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import minerals_region as mr
import minerals_region_emit as mre

ROOT = Path(__file__).resolve().parent
OUT = mr.OUT
CACHE = ROOT / "data" / "cache"
RES = mr.RES
DEM90 = "https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif"
W = {"emit": 0.35, "s2": 0.20, "silica": 0.20, "fault": 0.25}
AGREE_Z = 2.0             # a source "sees something" above this anomaly
FAULT_KM = 0.5            # on a fault if closer than this
TOP_PERCENT = 0.12
MIN_CELLS = 4
N_TARGETS = 30


def log(*a):
    print("[gold]", *a, flush=True)


def anomaly(a, keep):
    """Robust z of a layer against the rocks around it (~30 km)."""
    a = np.where(keep, a, np.nan).astype(np.float32)
    a -= mr.local_background(a, keep)
    med, mad = mr.robust_stats(a, keep)
    z = (a - med) / mad
    z[~keep] = np.nan
    return z


# ---------------------------------------------------------------- terrain: faults and fractures
def dem_grid(h, w):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
    f = CACHE / "sahara_dem90.npy"
    if f.exists():
        return np.load(f)
    lat0, lat1, lon0, lon1 = mr.BOX
    tr = from_bounds(lon0, lat0, lon1, lat1, w, h)
    z = np.full((h, w), np.nan, np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        for la in range(math.floor(lat0), math.floor(lat1) + 1):
            for lo in range(math.floor(lon0), math.floor(lon1) + 1):
                n = (f"Copernicus_DSM_COG_30_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
                     f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")
                try:
                    with rasterio.open(DEM90.format(n=n)) as src, WarpedVRT(
                            src, crs="EPSG:4326", transform=tr, width=w, height=h,
                            resampling=Resampling.bilinear, nodata=-9999) as v:
                        a = v.read(1).astype(np.float32)
                except Exception:  # noqa: BLE001 - sea or no tile
                    continue
                m = a > -9000
                z[m] = a[m]
    z = np.nan_to_num(z, nan=0.0)
    np.save(f, z)
    log("terrain model assembled")
    return z


def faults(z):
    """Lineaments at 200 m (the 90 m terrain model, halved to fit in memory), brought back to the 100 m grid."""
    import minerals_struct as ms
    h, w = z.shape
    z2 = z[::2, ::2].astype(np.float32)
    dy = 2 * RES * 111320
    dx = dy * math.cos(math.radians(24))
    line, ridge, edges, az, strength, coh = ms.lineaments(z2, dx, dy)
    del ridge, edges, az, strength, coh
    dist = ndimage.distance_transform_edt(~line, sampling=(dy / 1000, dx / 1000)).astype(np.float32)
    dens = ndimage.uniform_filter(line.astype(np.float32), 11)          # share of fault pixels within ~2 km

    def up(a):
        return np.repeat(np.repeat(a, 2, 0), 2, 1)[:h, :w]
    return up(line), up(dist), up(dens)


# ---------------------------------------------------------------- ASTER thermal: silica
def silica(h, w):
    """ASTER Global Emissivity Dataset (100 m, 2000-2008 mean). Quartz has a deep emissivity low at 8.6-9.1 um
    (bands 10-12) and not at 11 um: quartz index QI = e11^2 / (e10 * e12) (Ninomiya 2003)."""
    f = CACHE / "sahara_aster_qi.npy"
    if f.exists():
        return np.load(f)
    if not (os.environ.get("EARTHDATA_TOKEN") or os.environ.get("EARTHDATA_PASSWORD")):
        log("no Earthdata login: silica layer skipped")
        return None
    import earthaccess
    import h5py
    earthaccess.login(strategy="environment")
    lat0, lat1, lon0, lon1 = mr.BOX
    res = earthaccess.search_data(short_name="AG100", version="003", bounding_box=(lon0, lat0, lon1, lat1), count=200)
    log(f"ASTER emissivity tiles: {len(res)}")
    qi = np.full((h, w), np.nan, np.float32)
    for g in res:
        tmp = Path(tempfile.mkdtemp())
        try:
            p = earthaccess.download([g], str(tmp))
            for q in p:
                if not str(q).endswith(".h5"):
                    continue
                with h5py.File(q, "r") as hf:
                    e = hf["Emissivity"]["Mean"][:].astype(np.float32) * 0.001        # (5, ny, nx): bands 10-14
                    la = hf["Geolocation"]["Latitude"][:].astype(np.float64)
                    lo = hf["Geolocation"]["Longitude"][:].astype(np.float64)
                ok = (e[0] > 0.5) & (e[1] > 0.5) & (e[2] > 0.5)
                v = np.where(ok, e[1] * e[1] / np.maximum(e[0] * e[2], 1e-4), np.nan)
                inside = ok & (la >= lat0) & (la < lat1) & (lo >= lon0) & (lo < lon1)
                ci = ((lat1 - la[inside]) / RES).astype(int)
                cj = ((lo[inside] - lon0) / RES).astype(int)
                m = (ci >= 0) & (ci < h) & (cj >= 0) & (cj < w)
                qi[ci[m], cj[m]] = v[inside][m]
        except Exception as ex:  # noqa: BLE001
            log("ASTER tile skipped:", ex)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    # ASTER cells (100 m) and our grid (100 m) do not line up exactly: fill one-cell holes
    holes = ~np.isfinite(qi)
    if (~holes).any():
        idx = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
        filled = qi[tuple(idx)]
        near = ndimage.distance_transform_edt(holes) <= 1.5
        qi = np.where(holes & near, filled, qi)
    np.save(f, qi)
    log(f"silica layer: {100 * np.isfinite(qi).mean():.1f}% of the box")
    return qi


# ---------------------------------------------------------------- fusion
def main():
    t0 = time.time()
    for k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):
        if os.environ.get(k, "").strip():
            os.environ[k] = os.environ[k].strip()
        else:
            os.environ.pop(k, None)
    data = json.loads((OUT / "sahara.json").read_text())
    h, w = mr.shape(RES)
    inside = mr.outline_mask(h, w, RES)
    layers, used = {}, []

    ec = CACHE / "sahara_emit.npz"
    if ec.exists():
        z = np.load(ec)
        A = {"fam": z["fam"], "depth": z["depth"].astype(np.float32), "seen": z["seen"]}
        sc, div, keep_e = mre.gold_score(A, inside)
        layers["emit"] = sc
        used.append("EMIT")
        kinds_src = A
        log("EMIT alteration ready")
    else:
        kinds_src = None
        log("no EMIT grid in the cache")

    sc2 = CACHE / "sahara_s2.npz"
    keep = None
    if sc2.exists():
        z = np.load(sc2)
        keep = z["keep"]
        zc = z["clay"].astype(np.float32) / 25 - 2
        zf = z["ferric"].astype(np.float32) / 25 - 2
        both = np.where(keep, 0.5 * np.clip(zc, 0, None) + 0.5 * np.clip(zf, 0, None) + 0.5 * np.minimum(zc, zf).clip(0), np.nan)
        layers["s2"] = both.astype(np.float32)
        used.append("Sentinel-2")
        log("Sentinel-2 alteration ready")
    if keep is None:                                  # bare rock: where EMIT saw minerals, else everything inside
        keep = inside & (kinds_src["seen"] if kinds_src else True)
    keep = keep & inside

    qi = silica(h, w)
    if qi is not None and np.isfinite(qi).any():
        layers["silica"] = anomaly(qi, keep & np.isfinite(qi))
        used.append("ASTER")
        log("silica anomaly ready")

    dem = dem_grid(h, w)
    line, dist_km, dens = faults(dem)
    del dem
    near = np.exp(-dist_km / 1.0)                     # 1 on a fault, 0.37 at 1 km, ~0 beyond 3 km
    dz = anomaly(dens, keep)
    fault_sc = np.clip(4 * near + 0.5 * np.nan_to_num(dz).clip(0, 6), 0, 8)
    layers["fault"] = np.where(keep, fault_sc, np.nan).astype(np.float32)
    used.append("Copernicus DEM")
    log("faults ready")

    tot = sum(W[k] for k in layers)
    score = np.zeros((h, w), np.float32)
    agree = np.zeros((h, w), np.uint8)
    for k, a in layers.items():
        a = np.nan_to_num(a, nan=0.0)
        score += W[k] / tot * np.clip(a, 0, 8)
        if k == "fault":
            agree += (dist_km < FAULT_KM).astype(np.uint8)
        else:
            agree += (a >= AGREE_Z).astype(np.uint8)
    score = ndimage.uniform_filter(score, 3)          # a target is a place, not a single 100 m cell
    score[~keep] = np.nan
    ok = keep & (agree >= 2)

    # targets
    thr = np.nanpercentile(score[keep], 100 - TOP_PERCENT) if keep.any() else np.inf
    hot = ok & (score >= thr)
    log(f"threshold {thr:.2f} · {int(hot.sum())} cells above it where 2+ sources agree")
    lab, n = ndimage.label(hot, structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    size = ndimage.sum(hot, lab, idx) if n else []
    peak = ndimage.maximum(np.nan_to_num(score), lab, idx) if n else []
    com = ndimage.center_of_mass(np.nan_to_num(score).clip(0), lab, idx) if n else []
    lat0, lat1, lon0, lon1 = mr.BOX
    cand = []
    for k in (np.argsort(-np.asarray(peak)) if n else []):
        if size[k] < MIN_CELLS:
            continue
        cy, cx = com[k]
        iy, ix = int(cy), int(cx)
        lat, lon = lat1 - (cy + 0.5) * RES, lon0 + (cx + 0.5) * RES
        sl = (slice(max(0, iy - 3), iy + 4), slice(max(0, ix - 3), ix + 4))
        ev = {}
        for name, a in layers.items():
            if name == "fault":
                continue
            ev[name] = round(float(np.nanmax(a[sl])) if np.isfinite(a[sl]).any() else 0, 1)
        ev["fault_km"] = round(float(dist_km[sl].min()), 2)
        kinds = {}
        if kinds_src is not None:
            win = kinds_src["fam"][max(0, iy - 8):iy + 9, max(0, ix - 8):ix + 9]
            sv = kinds_src["seen"][max(0, iy - 8):iy + 9, max(0, ix - 8):ix + 9]
            for kn, ids in mre.KINDS.items():
                c = float((np.isin(win, ids) & sv).sum()) / max(int(sv.sum()), 1)
                if c >= 0.03:
                    kinds[kn] = round(100 * c)
        srcs = [s for s, ok_ in (("EMIT", ev.get("emit", 0) >= AGREE_Z), ("Sentinel-2", ev.get("s2", 0) >= AGREE_Z),
                                 ("ASTER", ev.get("silica", 0) >= AGREE_Z), ("faults", ev["fault_km"] < FAULT_KM)) if ok_]
        cand.append({"lat": round(lat, 5), "lon": round(lon, 5), "cells": int(size[k]), "peak": round(float(peak[k]), 2),
                     "model": "gold", "region": "dakhla" if lat < mr.REGION_SPLIT_LAT else "laayoune",
                     "evidence": ev, "agree": srcs, "kinds": kinds, "where0": mr.where(lat, lon)})
    tg = []
    for reg in ("dakhla", "laayoune"):
        sel = [t for t in cand if t["region"] == reg][:N_TARGETS]
        top = sel[0]["peak"] if sel else 1
        for i, t in enumerate(sel, 1):
            t["rank"], t["rel"] = i, int(round(100 * t["peak"] / top))
            t["area_ha"] = int(t.pop("cells") * (RES * 111.32) ** 2 * 100)
            wh = t.pop("where0")
            ks = sorted(t["kinds"], key=lambda q: -t["kinds"][q])
            t["where"] = {"ar": wh["ar"] + (" · " + "، ".join(mre.KIND_AR[q] for q in ks) if ks else ""),
                          "fr": wh["fr"] + (" · " + ", ".join(mre.KIND_FR[q] for q in ks) if ks else ""), "km": wh["km"]}
        tg += sel

    # map layer
    from PIL import Image
    vv = score[keep & np.isfinite(score)]
    lo, hi = np.nanpercentile(vv, 98.5), np.nanpercentile(vv, 99.97)
    x = np.clip((np.nan_to_num(score, nan=-9) - lo) / (hi - lo + 1e-9), 0, 1)
    x[~((np.nan_to_num(score, nan=-9) >= lo) & (agree >= 2))] = -1
    x = ndimage.maximum_filter(x, size=3)
    meta = data["tiles"]
    mre.save_tiles(Image.fromarray(mr.colour(x)), "gold", "png", meta)
    data["tiles"] = meta
    data["models"]["gold"] = {"ar": "الذهب", "fr": "Or", "targets": tg, "check": [],
                              "sources": used, "weights": {k: W[k] for k in layers}, "agree_z": AGREE_Z,
                              "fault_km": FAULT_KM}
    (OUT / "sahara.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    mr.build_page(data)
    lines = ["region,rank,lat,lon,area_ha,strength_rel,sources,emit_z,s2_z,silica_z,fault_km,alteration,where"]
    for t in tg:
        e = t["evidence"]
        lines.append(f"{t['region']},{t['rank']},{t['lat']},{t['lon']},{t['area_ha']},{t['rel']},{'+'.join(t['agree'])},"
                     f"{e.get('emit', '')},{e.get('s2', '')},{e.get('silica', '')},{e['fault_km']},"
                     f"{'+'.join(t['kinds'])},\"{t['where']['fr']}\"")
    (OUT / "gold_targets.csv").write_text("\n".join(lines), encoding="utf-8")
    log(f"sources: {used} · {len(tg)} targets · {time.time() - t0:.0f} s")
    for t in tg[:6] + [t for t in tg if t["region"] == "laayoune"][:4]:
        log(f"{t['region']} #{t['rank']} {t['lat']},{t['lon']} {t['agree']} {t['evidence']} {t['kinds']} · {t['where']['fr']}")


if __name__ == "__main__":
    main()
