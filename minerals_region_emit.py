"""MADAR Minerals - NASA EMIT over the whole Sahara survey area, and a gold model built on it.

Gold itself cannot be seen from space. What can be seen is the rock that hot, gold-bearing fluids changed on their
way up: sericite (fine muscovite), carbonate, chlorite and epidote around orogenic gold in shear zones; alunite,
pyrophyllite and kaolinite around epithermal gold; jarosite and goethite where sulphides weathered at the surface.
The gold model scores, at 100 m, how much of that alteration a place shows within 1 km, how many different kinds
sit together (zoned alteration is a stronger sign than one mineral), and how far it stands out from the rocks
around it (~30 km), then lists the strongest places.

    python minerals_region_emit.py            # download what is missing and rebuild the layers
    python minerals_region_emit.py --max 600  # cap on EMIT scenes read

Grid and page are the ones of the Sahara survey (minerals_region.py, docs/minerals/sahara/).
"""
import argparse
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import minerals_emit as me
import minerals_region as mr

ROOT = Path(__file__).resolve().parent
OUT = mr.OUT
CACHE = ROOT / "data" / "cache" / "sahara_emit.npz"     # kept between runs by the workflow cache, not in git
OLD_CACHE = ROOT / "data" / "sahara_emit.npz"
RES = mr.RES
F = me.FAM_ID
# how much each mineral family says "hot fluids passed here" for gold (0 = says nothing)
GOLD_W = {F["muscovite"]: 1.0, F["alunite"]: 1.0, F["pyrophyllite"]: 1.0, F["jarosite"]: 1.0, F["copper"]: 1.0,
          F["chlorite"]: 0.6, F["epidote"]: 0.6, F["carbonate"]: 0.5, F["kaolinite"]: 0.3, F["montmorillonite"]: 0.2}
KINDS = {"sericite": [F["muscovite"]], "argillic": [F["alunite"], F["pyrophyllite"], F["kaolinite"]],
         "propylitic": [F["chlorite"], F["epidote"]], "carbonate": [F["carbonate"]],
         "sulphide_weathering": [F["jarosite"]]}
WIN = 11            # 1.1 km window at 100 m
TOP_PERCENT = 0.12
MIN_CELLS = 4
N_TARGETS = 30
KIND_AR = {"sericite": "سيريسيت", "argillic": "تغيّر طيني متقدم", "propylitic": "كلوريت وإبيدوت",
           "carbonate": "كربونات", "sulphide_weathering": "جاروسيت (كبريتيدات متأكسدة)"}
KIND_FR = {"sericite": "séricite", "argillic": "argilique avancée", "propylitic": "chlorite-épidote",
           "carbonate": "carbonates", "sulphide_weathering": "jarosite (sulfures oxydés)"}


def grid():
    h, w = mr.shape(RES)
    return h, w


def accumulate(path, A, table):
    """Add one EMIT scene to the Sahara grid: per cell keep the strongest clay/carbonate/sulphate answer."""
    import netCDF4
    fam, depth, seen = A["fam"], A["depth"], A["seen"]
    h, w = fam.shape
    lat0, lat1, lon0, lon1 = mr.BOX
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        gt = [float(x) for x in ds.getncattr("geotransform")]
        loc = ds.groups["location"]
        gx, gy = loc["glt_x"][:].astype(np.int32), loc["glt_y"][:].astype(np.int32)
        ok = (gx > 0) & (gy > 0)
        oi, oj = np.nonzero(ok)
        lon = gt[0] + (oj + 0.5) * gt[1]
        lat = gt[3] + (oi + 0.5) * gt[5]
        inside = (lat >= lat0) & (lat < lat1) & (lon >= lon0) & (lon < lon1)
        if not inside.any():
            return 0
        oi, oj, lat, lon = oi[inside], oj[inside], lat[inside], lon[inside]
        ry, rx = gy[oi, oj] - 1, gx[oi, oj] - 1
        ci = np.clip(((lat1 - lat) / RES).astype(np.int32), 0, h - 1)
        cj = np.clip(((lon - lon0) / RES).astype(np.int32), 0, w - 1)
        seen[ci, cj] = True
        ftab = me.file_table(ds, table)
        lut = np.zeros(max(max(ftab.keys(), default=0), 4096) + 2, np.uint8)
        for k, v in ftab.items():
            if 0 <= k < len(lut):
                lut[k] = v
        for g in (1, 2):
            vm, vb = ds.variables.get(f"group_{g}_mineral_id"), ds.variables.get(f"group_{g}_band_depth")
            if vm is None or vb is None:
                continue
            mid = np.asarray(vm[:])[ry, rx].astype(np.int64)
            bd = np.asarray(vb[:], dtype=np.float32)[ry, rx]
            f = lut[np.clip(mid, 0, len(lut) - 1)]
            good = (mid > 0) & (bd > 0) & np.isfinite(bd) & (f > 0) & ~np.isin(f, list(me.IRON))
            if not good.any():
                continue
            order = np.argsort(bd[good])
            ii, jj, ff, dd = ci[good][order], cj[good][order], f[good][order], bd[good][order]
            st = dd > depth[ii, jj]
            fam[ii[st], jj[st]] = ff[st]
            depth[ii[st], jj[st]] = dd[st]
    return int(inside.sum())


def footprint(g):
    """Bounding box of a granule from its CMR record."""
    try:
        sp = g["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        pts = [(p["Latitude"], p["Longitude"]) for poly in sp.get("GPolygons", []) for p in poly["Boundary"]["Points"]]
        if pts:
            la, lo = zip(*pts)
            return min(la), max(la), min(lo), max(lo)
        r = sp["BoundingRectangles"][0]
        return r["SouthBoundingCoordinate"], r["NorthBoundingCoordinate"], r["WestBoundingCoordinate"], r["EastBoundingCoordinate"]
    except (KeyError, IndexError, TypeError):
        return None


def seen_share(A, fp):
    lat0, lat1, lon0, lon1 = mr.BOX
    a0, a1, o0, o1 = fp
    h, w = A["seen"].shape
    i0, i1 = int(max(0, (lat1 - a1) / RES)), int(min(h, (lat1 - a0) / RES))
    j0, j1 = int(max(0, (o0 - lon0) / RES)), int(min(w, (o1 - lon0) / RES))
    if i1 <= i0 or j1 <= j0:
        return 1.0
    return float(A["seen"][i0:i1:4, j0:j1:4].mean())


def collect(A, max_scenes, table):
    import earthaccess
    lat0, lat1, lon0, lon1 = mr.BOX
    res = earthaccess.search_data(short_name="EMITL2BMIN", bounding_box=(lon0, lat0, lon1, lat1), count=4000)
    print(f"[sahara-emit] {len(res)} EMIT records over the area")
    best = {}
    for g in res:                                   # one record per acquisition, newest product version
        name = g["umm"].get("GranuleUR", "")
        m = re.search(r"_(\d{3})_(\d{8}T\d{6})", name)
        key = m.group(2) if m else name
        v = m.group(1) if m else ""
        if key not in best or v > best[key][0]:
            best[key] = (v, g)

    def cloud(g):
        try:
            return float(g["umm"].get("CloudCover") or 0)
        except (TypeError, ValueError):
            return 0.0
    todo = sorted((g for _, g in best.values()), key=cloud)
    done = set(A.get("done", []))
    print(f"[sahara-emit] {len(todo)} acquisitions, {len(done)} already read")
    n, t0 = 0, time.time()
    for g in todo:
        if n >= max_scenes:
            break
        name = g["umm"].get("GranuleUR", "")
        if name in done:
            continue
        fp = footprint(g)
        if fp and seen_share(A, fp) > 0.95:          # this place is already covered by clearer scenes
            continue
        links = [u for u in g.data_links() if u.endswith(".nc") and "MINUNCERT" not in u]
        if not links:
            continue
        tmp = Path(tempfile.mkdtemp())
        try:
            p = earthaccess.download(links[:1], str(tmp))
            if p:
                px = accumulate(p[0], A, table)
                n += 1
                done.add(name)
                if n % 10 == 0:
                    print(f"[sahara-emit] {n} scenes read · {100 * A['seen'].mean():.1f}% of the box seen · "
                          f"{time.time() - t0:.0f} s")
        except Exception as e:  # noqa: BLE001
            print("[sahara-emit] skip", name, e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    A["done"] = sorted(done)
    return n


def gold_score(A, inside):
    fam, seen = A["fam"], A["seen"]
    w = np.zeros(fam.shape, np.float32)
    for k, v in GOLD_W.items():
        w[fam == k] = v
    s = seen & inside
    sf = s.astype(np.float32)
    n = ndimage.uniform_filter(sf, WIN)
    dens = np.where(n > 0.2, ndimage.uniform_filter(w * sf, WIN) / np.maximum(n, 1e-3), 0).astype(np.float32)
    div = np.zeros(fam.shape, np.float32)                # how many kinds of alteration sit together
    for ids in KINDS.values():
        share = ndimage.uniform_filter((np.isin(fam, ids) & s).astype(np.float32), WIN + 4) / np.maximum(
            ndimage.uniform_filter(sf, WIN + 4), 1e-3)
        div += share >= 0.03
    keep = s & (n > 0.2)
    anom = dens - mr.local_background(dens, keep)
    med, mad = mr.robust_stats(anom, keep)
    z = (anom - med) / mad
    score = np.where(keep, z * (1 + 0.25 * div), np.nan).astype(np.float32)
    return score, div, keep


def targets(score, keep, A):
    v = score[keep]
    thr = np.nanpercentile(v, 100 - TOP_PERCENT)
    hot = keep & (score >= thr)
    lab, n = ndimage.label(hot, structure=np.ones((3, 3)))
    if not n:
        return []
    idx = np.arange(1, n + 1)
    size = ndimage.sum(hot, lab, idx)
    peak = ndimage.maximum(np.nan_to_num(score), lab, idx)
    com = ndimage.center_of_mass(np.nan_to_num(score).clip(0), lab, idx)
    lat0, lat1, lon0, lon1 = mr.BOX
    fam = A["fam"]
    out = []
    for k in np.argsort(-peak):
        if size[k] < MIN_CELLS:
            continue
        cy, cx = com[k]
        lat, lon = lat1 - (cy + 0.5) * RES, lon0 + (cx + 0.5) * RES
        r = 8                                          # assemblage within ~800 m
        y0, y1, x0, x1 = int(max(0, cy - r)), int(cy + r + 1), int(max(0, cx - r)), int(cx + r + 1)
        win = fam[y0:y1, x0:x1]
        sv = A["seen"][y0:y1, x0:x1]
        kinds = {}
        for kn, ids in KINDS.items():
            c = float((np.isin(win, ids) & sv).sum()) / max(int(sv.sum()), 1)
            if c >= 0.03:
                kinds[kn] = round(100 * c)
        region = "dakhla" if lat < mr.REGION_SPLIT_LAT else "laayoune"
        out.append({"lat": round(lat, 5), "lon": round(lon, 5), "area_ha": int(size[k]) * 1, "peak": round(float(peak[k]), 1),
                    "model": "gold", "region": region, "kinds": kinds,
                    "where": mr.where(lat, lon)})
    res = []
    for reg in ("dakhla", "laayoune"):
        sel = [t for t in out if t["region"] == reg][:N_TARGETS]
        top = sel[0]["peak"] if sel else 1
        for i, t in enumerate(sel, 1):
            t["rank"] = i
            t["rel"] = int(round(100 * t["peak"] / top))
            t["area_ha"] = int(t["area_ha"] * (RES * 111.32) ** 2 * 100)     # cells -> hectares
            ks = sorted(t["kinds"], key=lambda q: -t["kinds"][q])
            t["where"] = dict(t["where"])
            t["where"]["ar"] += " · " + "، ".join(KIND_AR[q] for q in ks) if ks else ""
            t["where"]["fr"] += " · " + ", ".join(KIND_FR[q] for q in ks) if ks else ""
        res += sel
    return res


def save_tiles(img, key, fmt, meta):
    from PIL import Image
    for name in ("ov", "dt"):
        W, H = meta[name]["w"], meta[name]["h"]
        im = img.resize((W, H), Image.NEAREST)
        if name == "ov":
            im.save(OUT / "tiles" / f"ov_{key}.{fmt}", optimize=True)
            continue
        have = []
        for ty in range(0, H, 500):
            for tx in range(0, W, 500):
                c = im.crop((tx, ty, min(tx + 500, W), min(ty + 500, H)))
                if c.getextrema()[3][1] == 0:
                    continue
                c = c.quantize(colors=48, method=Image.Quantize.FASTOCTREE)
                c.save(OUT / "tiles" / "dt" / f"{key}_{ty // 500}_{tx // 500}.{fmt}", optimize=True)
                have.append(f"{ty // 500}_{tx // 500}")
        meta[name]["have"][key] = have


def mineral_image(A, inside):
    from PIL import Image
    fam, depth, seen = A["fam"], A["depth"].astype(np.float32), A["seen"]
    rgba = np.zeros(fam.shape + (4,), np.uint8)
    rgba[inside & ~seen] = (20, 20, 20, 120)
    for k, (_, c) in enumerate(me.FAMILIES, 1):
        m = fam == k
        if m.any():
            rgba[m, :3] = c
            rgba[m, 3] = np.clip(70 + depth[m] / 0.12 * 185, 70, 255).astype(np.uint8)
    rgba[~inside] = 0
    return Image.fromarray(rgba, "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=700)
    a = ap.parse_args()
    data = json.loads((OUT / "sahara.json").read_text())
    h, w = grid()
    if not CACHE.exists() and OLD_CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(OLD_CACHE), str(CACHE))
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        A = {"fam": z["fam"].copy(), "depth": z["depth"].astype(np.float32), "seen": z["seen"].copy(),
             "done": [str(x) for x in z["done"]]}
    else:
        A = {"fam": np.zeros((h, w), np.uint8), "depth": np.zeros((h, w), np.float32),
             "seen": np.zeros((h, w), bool), "done": []}
    for k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):
        if os.environ.get(k, "").strip():
            os.environ[k] = os.environ[k].strip()
        else:
            os.environ.pop(k, None)
    table = me.lookup()
    if os.environ.get("EARTHDATA_TOKEN") or os.environ.get("EARTHDATA_PASSWORD"):
        import earthaccess
        earthaccess.login(strategy="environment")
        n = collect(A, a.max, table)
        print(f"[sahara-emit] read {n} new scenes")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, fam=A["fam"], depth=A["depth"].astype(np.float16), seen=A["seen"],
                            done=np.array(A["done"]))
    inside = mr.outline_mask(h, w, RES)
    cov = float((A["seen"] & inside).sum()) / max(int(inside.sum()), 1)
    print(f"[sahara-emit] EMIT coverage of the Sahara area: {100 * cov:.1f}%")
    meta = data["tiles"]
    save_tiles(mineral_image(A, inside), "emit", "png", meta)
    score, div, keep = gold_score(A, inside)
    v = score[keep]
    lo, hi = np.nanpercentile(v, 98.0), np.nanpercentile(v, 99.97)
    x = np.clip((np.nan_to_num(score, nan=-9) - lo) / (hi - lo + 1e-9), 0, 1)
    x[~(np.nan_to_num(score, nan=-9) >= lo)] = -1
    big = ndimage.maximum_filter(x, size=3)             # keep small hot spots visible on the overview
    from PIL import Image
    save_tiles(Image.fromarray(mr.colour(big)), "gold", "png", meta)
    tg = targets(score, keep, A)
    fams = {}
    n_seen = max(int((A["seen"] & inside).sum()), 1)
    for k, (f, _) in enumerate(me.FAMILIES, 1):
        c = int(((A["fam"] == k) & inside).sum())
        if c:
            fams[f] = round(100 * c / n_seen, 2)
    data["models"]["gold"] = {"ar": "الذهب", "fr": "Or", "targets": tg, "check": []}
    data["emit"] = {"coverage_pct": round(100 * cov, 1), "scenes": len(A["done"]), "families": fams,
                    "src": "NASA EMIT L2B Mineral Identification (EMITL2BMIN), LP DAAC"}
    data["tiles"] = meta
    (OUT / "sahara.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    mr.build_page(data)
    for t in tg[:5] + [t for t in tg if t["region"] == "laayoune"][:3]:
        print(f"[gold] {t['region']} #{t['rank']} {t['lat']},{t['lon']} peak {t['peak']} kinds {t['kinds']} · {t['where']['fr']}")
    print(f"[sahara-emit] gold targets: {len(tg)} · families {sorted(fams.items(), key=lambda q: -q[1])[:6]}")


if __name__ == "__main__":
    main()
