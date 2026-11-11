"""MADAR Minerals - NASA EMIT mineral layer for each study site.

EMIT (imaging spectrometer on the ISS, 285 bands, 60 m) publishes, for every scene, the mineral that best explains
the shape of each pixel's spectrum: one answer near 2.2 um (clays, micas, carbonates, sulfates) and one near 1 um
(iron oxides). This script takes those answers (product EMITL2BMIN), groups the ~290 reference spectra into the
mineral families a geologist reads alteration with, mosaics the clearest scenes over each site, and checks which
family sits under each Madar target.

    python minerals_emit.py               # sites without an EMIT layer yet; re-attach the others from disk
    python minerals_emit.py --refresh     # download again for every site

Needs a free NASA Earthdata login in EARTHDATA_USERNAME / EARTHDATA_PASSWORD (GitHub secrets EARTHDATA_USER/PASS).
Reference table: emit-sds/emit-sds-l2b data/mineral_grouping_matrix_20230503.csv (Apache-2.0), copied in data/.
"""
import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "minerals"
TABLE = ROOT / "data" / "emit_mineral_groups.csv"
RES = 0.0006                  # site grid, degrees (~60 m, EMIT's own pixel)
MAX_SCENES = 8
TARGET_R_M = 300

# families in the order a geologist reads them, with the colour of the layer
FAMILIES = [
    ("copper", (0, 200, 180)), ("ree", (157, 78, 221)), ("serpentine", (120, 160, 120)),
    ("alunite", (255, 45, 85)), ("kaolinite", (255, 110, 199)), ("pyrophyllite", (199, 125, 255)),
    ("muscovite", (255, 214, 10)), ("montmorillonite", (244, 162, 97)), ("chlorite", (82, 183, 136)),
    ("epidote", (45, 106, 79)), ("carbonate", (76, 201, 240)), ("gypsum", (230, 230, 230)),
    ("jarosite", (192, 138, 0)), ("hematite", (214, 40, 40)), ("goethite", (247, 127, 0)),
    ("iron_oxide", (231, 111, 81)), ("clay", (181, 131, 141)), ("mafic", (90, 98, 110)),
]
FAM_ID = {f: k + 1 for k, (f, _) in enumerate(FAMILIES)}
NAME_RULES = [("azurite", "copper"), ("malachite", "copper"), ("chrysocolla", "copper"), ("cuprite", "copper"),
              ("chalcopyrite", "copper"), ("neodymium", "ree"), ("samarium", "ree"), ("praseodym", "ree"),
              ("chrysotile", "serpentine"), ("serpentin", "serpentine"), ("antigorite", "serpentine"), ("talc", "serpentine"),
              ("montmor", "montmorillonite"), ("beidellite", "montmorillonite"), ("nontronite", "montmorillonite"),
              ("actinolite", "chlorite"), ("tremolite", "chlorite"), ("clinochlore", "chlorite"), ("thuringite", "chlorite"),
              ("chlorite", "chlorite"), ("siderite", "carbonate"), ("rhodochrosite", "carbonate"),
              ("szomolnokite", "jarosite"), ("copiapite", "jarosite"), ("coquimbite", "jarosite"), ("butlerite", "jarosite"),
              ("schwertmannite", "jarosite"), ("ferrihydrite", "iron_oxide"), ("magnetite", "iron_oxide"), ("pyrite", "iron_oxide"),
              ("phlogopite", "muscovite"), ("biotite", "muscovite"), ("paragonite", "muscovite"),
              ("olivine", "mafic"), ("hypersthene", "mafic"), ("diopside", "mafic"), ("augite", "mafic"), ("bronzite", "mafic"),
              ("enstatite", "mafic"), ("pigeonite", "mafic"), ("hedenbergite", "mafic"), ("basalt", "mafic"),
              ("alunite", "alunite"), ("jaros", "jarosite"), ("pyrophyl", "pyrophyllite"), ("epidote", "epidote"),
              ("kaol", "kaolinite"), ("dickite", "kaolinite"), ("halloysite", "kaolinite")]
COL_RULES = [("Kaolinite", "kaolinite"), ("Illite+Muscovite", "muscovite"), ("Montmorillonite", "montmorillonite"),
             ("Calcite", "carbonate"), ("Dolomite", "carbonate"), ("Carbonate", "carbonate"), ("Chlorite", "chlorite"),
             ("Gypsum", "gypsum"), ("Sulfate", "gypsum"), ("Hematite", "hematite"), ("Goethite", "goethite"),
             ("Iron Oxide", "iron_oxide"), ("Clay", "clay"), ("Vermiculite", "clay")]
SKIP_COLS = ("Vegetation", "Artifical Materials", "Coating")


def lookup():
    """EMIT mineral index -> family number (0 = not a mineral we map: vegetation, plastics, coatings)."""
    out = {}
    for r in csv.DictReader(TABLE.open(encoding="utf-8")):
        name, fam = r["Name"].lower(), None
        if any((r.get(c) or "").strip() for c in SKIP_COLS) or any(
                k in name for k in ("plastic", "cyanide", "acid", "wtc", "varnish", "pitch", "tarp")):
            out[int(r["Index"])] = 0
            continue
        for key, f in NAME_RULES:
            if key in name:
                fam = f
                break
        if fam is None:
            best = 0.0
            for col, f in COL_RULES:
                try:
                    v = float(r.get(col) or 0)
                except ValueError:
                    v = 0.0
                if v > best:
                    best, fam = v, f
        out[int(r["Index"])] = FAM_ID.get(fam, 0)
    return out


def site_grid(s):
    (a0, o0), (a1, o1) = s["bounds"]
    w, h = int(math.ceil((o1 - o0) / RES)), int(math.ceil((a1 - a0) / RES))
    return a0, o0, a1, o1, w, h


def granules(bounds):
    import earthaccess
    (a0, o0), (a1, o1) = bounds
    res = earthaccess.search_data(short_name="EMITL2BMIN", bounding_box=(o0, a0, o1, a1), count=200)

    def cloud(r):
        try:
            return float(r["umm"].get("CloudCover") or 0)
        except (TypeError, ValueError):
            return 0.0
    res = sorted(res, key=cloud)[:MAX_SCENES]
    print(f"[emit] {len(res)} scenes kept (clearest first)")
    return res


def mosaic(paths, s, table):
    """Per site cell: family of the strongest absorption seen in any scene, its depth, and whether EMIT saw it."""
    import netCDF4
    a0, o0, a1, o1, w, h = site_grid(s)
    fam = np.zeros((h, w), np.uint8)
    depth = np.zeros((h, w), np.float32)
    seen = np.zeros((h, w), bool)
    used = []
    for p in paths:
        try:
            ds = netCDF4.Dataset(p)
        except OSError as e:
            print("[emit] unreadable", p, e)
            continue
        with ds:
            ds.set_auto_mask(False)
            if not used:
                print("[emit] file layout:", Path(p).name, "vars", list(ds.variables)[:12], "groups",
                      {g: list(ds.groups[g].variables) for g in ds.groups}, "attrs", ds.ncattrs()[:20])
            gt = [float(x) for x in ds.getncattr("geotransform")]
            loc = ds.groups["location"]
            gx, gy = loc["glt_x"][:].astype(np.int64), loc["glt_y"][:].astype(np.int64)
            ok = (gx > 0) & (gy > 0)
            oi, oj = np.nonzero(ok)
            lon = gt[0] + (oj + 0.5) * gt[1]
            lat = gt[3] + (oi + 0.5) * gt[5]
            inside = (lat >= a0) & (lat < a1) & (lon >= o0) & (lon < o1)
            if not inside.any():
                continue
            oi, oj, lat, lon = oi[inside], oj[inside], lat[inside], lon[inside]
            ry, rx = gy[oi, oj] - 1, gx[oi, oj] - 1
            ci = np.clip(((a1 - lat) / RES).astype(int), 0, h - 1)
            cj = np.clip(((lon - o0) / RES).astype(int), 0, w - 1)
            seen[ci, cj] = True
            for g in (1, 2):
                mid = np.asarray(ds[f"group_{g}_mineral_id"][:])[ry, rx].astype(np.int64)
                bd = np.asarray(ds[f"group_{g}_band_depth"][:], dtype=np.float32)[ry, rx]
                f = np.array([table.get(int(m), 0) for m in mid], np.uint8) if mid.size < 2_000_000 else \
                    np.vectorize(lambda m: table.get(int(m), 0), otypes=[np.uint8])(mid)
                good = (mid > 0) & (bd > 0) & np.isfinite(bd) & (f > 0)
                if not good.any():
                    continue
                order = np.argsort(bd[good])            # strongest written last, so it wins
                ii, jj, ff, dd = ci[good][order], cj[good][order], f[good][order], bd[good][order]
                stronger = dd > depth[ii, jj]
                fam[ii[stronger], jj[stronger]] = ff[stronger]
                depth[ii[stronger], jj[stronger]] = dd[stronger]
            used.append(Path(p).name)
    return fam, depth, seen, used


def picture(fam, depth, seen, path):
    from PIL import Image
    h, w = fam.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[~seen] = (20, 20, 20, 110)                       # not imaged by EMIT: shaded, so no one reads it as "nothing"
    for k, (_, c) in enumerate(FAMILIES, 1):
        m = fam == k
        if m.any():
            a = np.clip(55 + depth[m] / 0.12 * 200, 55, 255).astype(np.uint8)
            rgba[m, :3] = c
            rgba[m, 3] = a
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def summarise(s, fam, depth, seen):
    a0, o0, a1, o1, w, h = site_grid(s)
    n_seen = int(seen.sum())
    fams = {}
    for k, (f, _) in enumerate(FAMILIES, 1):
        c = int((fam == k).sum())
        if c:
            fams[f] = round(100 * c / max(n_seen, 1), 2)
    for t in s["targets"]:
        ci, cj = int((a1 - t["lat"]) / RES), int((t["lon"] - o0) / RES)
        r = max(1, int(TARGET_R_M / (RES * 111320)))
        sl = (slice(max(0, ci - r), ci + r + 1), slice(max(0, cj - r), cj + r + 1))
        sv, sf = seen[sl], fam[sl]
        if not sv.any():
            t["emit"] = {"cov": False}
            continue
        vals, cnt = np.unique(sf[sf > 0], return_counts=True)
        if not len(vals):
            t["emit"] = {"cov": True, "fam": None}
            continue
        k = int(vals[np.argmax(cnt)])
        t["emit"] = {"cov": True, "fam": FAMILIES[k - 1][0], "pct": round(100 * cnt.max() / sv.sum()),
                     "depth": round(float(depth[sl][sf == k].mean()), 3)}
    return {"coverage_pct": round(100 * n_seen / (w * h), 1), "families": fams}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    data = json.loads((OUT / "minerals.json").read_text())
    table = lookup()
    have_login = bool(os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"))
    if have_login:
        import earthaccess
        earthaccess.login(strategy="environment")
    else:
        print("[emit] no Earthdata login: re-attaching saved layers only")
    for s in data["sites"]:
        d = OUT / s["id"]
        npz = d / "emit.npz"
        if npz.exists() and not a.refresh:
            z = np.load(npz)
            fam, depth, seen, used = z["fam"], z["depth"], z["seen"], list(z["used"])
        elif have_login:
            res = granules(s["bounds"])
            if not res:
                print(f"[emit] {s['id']}: no EMIT scene over the site")
                s.pop("emit", None)
                continue
            tmp = Path(tempfile.mkdtemp())
            try:
                import earthaccess
                paths = earthaccess.download(res, str(tmp))
                fam, depth, seen, used = mosaic(paths, s, table)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            if not used:
                print(f"[emit] {s['id']}: scenes found but none covers the site")
                continue
            np.savez_compressed(npz, fam=fam, depth=depth.astype(np.float16), seen=seen, used=np.array(used))
        else:
            continue
        picture(fam, depth.astype(np.float32), seen, d / "emit.png")
        info = summarise(s, fam, depth.astype(np.float32), seen)
        info["scenes"] = [u.replace("EMIT_L2B_MIN_001_", "")[:15] for u in used]
        info["src"] = "NASA EMIT L2B Mineral Identification (EMITL2BMIN v001), LP DAAC"
        s["emit"] = info
        s["images"]["emit.png"] = f"{s['id']}/emit.png"
        top = sorted(info["families"].items(), key=lambda x: -x[1])[:4]
        on = [t["label"] for t in s["targets"] if (t.get("emit") or {}).get("fam")]
        print(f"[emit] {s['id']}: {info['coverage_pct']}% imaged · {len(used)} scenes · top {top} · "
              f"targets with an EMIT mineral: {len(on)}/{len(s['targets'])}")
    (OUT / "minerals.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    page = (ROOT / "minerals_console.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


if __name__ == "__main__":
    main()
