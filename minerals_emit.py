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
import re
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
# iron oxides coat almost every desert surface; they are read apart, so they do not hide the clays and carbonates
IRON = {FAM_ID[f] for f in ("hematite", "goethite", "iron_oxide", "mafic")}
MIN_LOCAL, MIN_ENRICH = 0.05, 1.5     # a target "has" a mineral if it covers 5 % around it and 1.5x its share of the site
# copper: only malachite and azurite have clear SWIR features. Chrysocolla is confused with desert opal crusts,
# cuprite and chalcopyrite have no SWIR signature, and "lazurite" (a blue silicate) must not match "azurite".
NO_MATCH = ("lazurite", "chrysocolla", "cuprite", "chalcopyrite")
NAME_RULES = [("neodymium", "ree"), ("samarium", "ree"), ("praseodym", "ree"),
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


NAME_EXTRA = [("hematite", "hematite"), ("goethite", "goethite"), ("lepidocrocite", "iron_oxide"), ("calcite", "carbonate"),
              ("dolomite", "carbonate"), ("ankerite", "carbonate"), ("gypsum", "gypsum"), ("anhydrite", "gypsum"),
              ("muscovite", "muscovite"), ("illite", "muscovite"), ("sericite", "muscovite"), ("vermiculite", "clay"),
              ("halloysite", "kaolinite"), ("smect", "montmorillonite"), ("iron oxide", "iron_oxide"), ("fe3+", "iron_oxide"),
              ("fe2+", "mafic")]
CSV_FAM = {}


def family_of_name(name):
    n = name.lower()
    if n.startswith(("azurite", "malachite")):
        return FAM_ID["copper"]
    if n.startswith(NO_MATCH):
        return 0
    if any(k in n for k in ("plastic", "cyanide", "acid", "wtc", "varnish", "pitch", "tarp", "grass", "plywood",
                            "fiberglass", "polystyrene", "painted", "vegetation", "lichen")):
        return 0
    for key, f in NAME_RULES + NAME_EXTRA:
        if key in n:
            return FAM_ID[f]
    return CSV_FAM.get(n, 0)


def file_table(ds, fallback):
    """v002 files carry their own index -> name table; use it so the ids are read with the right names."""
    g = ds.groups.get("mineral_metadata")
    if g is None or "index" not in g.variables or "name" not in g.variables:
        return fallback
    idx = np.asarray(g["index"][:]).astype(int).ravel()
    names = g["name"][:]
    names = [n if isinstance(n, str) else bytes(n).decode("utf-8", "ignore") for n in np.asarray(names).ravel()]
    return {int(i): family_of_name(str(n)) for i, n in zip(idx, names)}


def lookup():
    """EMIT mineral index -> family number (0 = not a mineral we map: vegetation, plastics, coatings)."""
    out = {}
    for r in csv.DictReader(TABLE.open(encoding="utf-8")):
        name, fam = r["Name"].lower(), None
        if any((r.get(c) or "").strip() for c in SKIP_COLS) or any(
                k in name for k in ("plastic", "cyanide", "acid", "wtc", "varnish", "pitch", "tarp")):
            out[int(r["Index"])] = 0
            continue
        if name.startswith(("azurite", "malachite")):
            fam = "copper"
        elif name.startswith(NO_MATCH):
            out[int(r["Index"])] = 0
            continue
        for key, f in ([] if fam else NAME_RULES):
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
        CSV_FAM[r["Name"].lower()] = FAM_ID.get(fam, 0)
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


def mosaic(paths, s, table, kind="sw"):
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
            ftab = file_table(ds, table)
            if not used:
                print(f"[emit] mineral table: {len(ftab)} entries, "
                      f"{sum(1 for v in ftab.values() if v)} mapped to a family")
            for g in (1, 2):
                vm, vb = ds.variables.get(f"group_{g}_mineral_id"), ds.variables.get(f"group_{g}_band_depth")
                if vm is None or vb is None:
                    print("[emit] missing group", g, "in", Path(p).name)
                    continue
                mid = np.asarray(vm[:])[ry, rx].astype(np.int64)
                bd = np.asarray(vb[:], dtype=np.float32)[ry, rx]
                table_ = ftab
                lut = np.zeros(max(max(table_.keys(), default=0), int(mid.max(initial=0))) + 2, np.uint8)
                for k_, v_ in table_.items():
                    if k_ >= 0:
                        lut[k_] = v_
                f = lut[np.clip(mid, 0, len(lut) - 1)]
                is_fe = np.isin(f, list(IRON))
                f = np.where(is_fe if kind == "fe" else ~is_fe, f, 0).astype(np.uint8)
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


def shares(fam, mask):
    n = max(int(mask.sum()), 1)
    out = {}
    for k, (f, _) in enumerate(FAMILIES, 1):
        c = int(((fam == k) & mask).sum())
        if c:
            out[f] = c / n
    return out


def summarise(s, fam, depth, seen, fam_fe):
    a0, o0, a1, o1, w, h = site_grid(s)
    n_seen = int(seen.sum())
    site = shares(fam, seen)
    site_fe = shares(fam_fe, seen)
    for t in s["targets"]:
        ci, cj = int((a1 - t["lat"]) / RES), int((t["lon"] - o0) / RES)
        r = max(1, int(TARGET_R_M / (RES * 111320)))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (yy * yy + xx * xx) <= r * r
        sl = (slice(ci - r, ci + r + 1), slice(cj - r, cj + r + 1))
        if ci - r < 0 or cj - r < 0 or ci + r >= h or cj + r >= w:
            t["emit"] = {"cov": False}
            continue
        sv = seen[sl] & disk
        if sv.sum() < 5:
            t["emit"] = {"cov": False}
            continue
        loc = shares(fam[sl], sv)
        best = None
        for f, v in loc.items():                   # the mineral that is concentrated here, not just present
            x = v / max(site.get(f, 0), 1e-3)
            if v >= MIN_LOCAL and x >= MIN_ENRICH and (best is None or x > best[2]):
                best = (f, v, x)
        t["emit"] = {"cov": True, "fam": best[0], "pct": round(100 * best[1]), "x": round(best[2], 1)} if best else \
            {"cov": True, "fam": None}
    rnd = lambda d: {k: round(100 * v, 2) for k, v in sorted(d.items(), key=lambda kv: -kv[1])}
    return {"coverage_pct": round(100 * n_seen / (w * h), 1), "families": rnd(site), "iron": rnd(site_fe)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    data = json.loads((OUT / "minerals.json").read_text())
    table = lookup()
    # trim stray spaces or line breaks pasted into the secrets; say what is wrong without printing them
    for k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):
        v = os.environ.get(k)
        if v is not None:
            if v != v.strip():
                print(f"[emit] {k} had spaces or a line break at its ends: trimmed")
            os.environ[k] = v.strip()
            if not os.environ[k]:
                del os.environ[k]
    u = os.environ.get("EARTHDATA_USERNAME", "")
    print(f"[emit] login check: username {len(u)} chars{' (looks like an email: use the username)' if '@' in u else ''}, "
          f"password {'set' if os.environ.get('EARTHDATA_PASSWORD') else 'missing'}, "
          f"token {'set' if os.environ.get('EARTHDATA_TOKEN') else 'not set'}")
    have_login = bool(os.environ.get("EARTHDATA_TOKEN") or (u and os.environ.get("EARTHDATA_PASSWORD")))
    if have_login:
        import earthaccess
        try:
            earthaccess.login(strategy="environment")
        except Exception as e:  # noqa: BLE001
            print("[emit] NASA refused the login:", e)
            have_login = False
    if not have_login:
        print("[emit] no Earthdata login: re-attaching saved layers only")
    for s in data["sites"]:
        d = OUT / s["id"]
        npz = d / "emit.npz"
        z = np.load(npz) if npz.exists() and not a.refresh else None
        if z is not None and "fam_fe" in z.files:
            fam, depth, seen, used = z["fam"], z["depth"], z["seen"], list(z["used"])
            fam_fe = z["fam_fe"]
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
                paths = [q for q in paths if "MINUNCERT" not in Path(q).name]
                best = {}                                  # one file per acquisition, newest product version
                for q in paths:
                    m_ = re.search(r"_(\d{3})_(\d{8}T\d{6})", Path(q).name)
                    key = m_.group(2) if m_ else Path(q).name
                    if key not in best or (m_ and m_.group(1) > best[key][0]):
                        best[key] = (m_.group(1) if m_ else "", q)
                paths = [v[1] for v in best.values()]
                fam, depth, seen, used = mosaic(paths, s, table, "sw")
                fam_fe, _, _, _ = mosaic(paths, s, table, "fe")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            if not used:
                print(f"[emit] {s['id']}: scenes found but none covers the site")
                continue
            np.savez_compressed(npz, fam=fam, depth=depth.astype(np.float16), seen=seen, used=np.array(used),
                                fam_fe=fam_fe)
        else:
            continue
        picture(fam, depth.astype(np.float32), seen, d / "emit.png")
        info = summarise(s, fam, depth.astype(np.float32), seen, fam_fe)
        info["scenes"] = sorted({(re.search(r"\d{8}T\d{6}", str(u)) or re.search(r".*", str(u))).group(0) for u in used})
        info["src"] = "NASA EMIT L2B Mineral Identification (EMITL2BMIN), LP DAAC"
        s["emit"] = info
        s["images"]["emit.png"] = f"{s['id']}/emit.png"
        top = sorted(info["families"].items(), key=lambda x: -x[1])[:4]
        on = [f'{t["label"]}:{t["emit"]["fam"]}x{t["emit"]["x"]}' for t in s["targets"] if (t.get("emit") or {}).get("fam")]
        print(f"[emit] {s['id']}: {info['coverage_pct']}% imaged · {len(used)} scenes · top {top} · "
              f"targets with a concentrated EMIT mineral: {len(on)}/{len(s['targets'])} {on}")
    (OUT / "minerals.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    page = (ROOT / "minerals_console.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


if __name__ == "__main__":
    main()
