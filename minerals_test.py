"""
MADAR Minerals: large blind test on known deposits, compared with chance.

For each known deposit (exact coordinates from Mindat / Wikipedia), a ~22 x 22 km box is placed with a RANDOM
offset (so the deposit is not at the centre). MADAR ranks its top targets without knowing the deposit.
  hit      = a top-10 target within 2 km of the deposit
  p_chance = probability that a random point in the box would be within 2 km of a top-10 target
The sum of p_chance is the number of hits expected by pure luck; the Poisson-binomial tail gives a p-value.

Usage:  python minerals_test.py          (real Sentinel-2 data, run in GitHub Actions)
        python minerals_test.py --demo   (synthetic scenes, to test the pipeline and the page)
"""
import argparse
import json
import math
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np

import minerals as M

OUT = M.OUT / "test"
BOX_DEG = 0.20
MAX_OFFSET = 0.06
SEED = 2026

# kind: mine = worked (pits/tailings visible), occurrence = known mineralisation / small workings
DEPOSITS = [
    ("Vein No. 61, Ait Ahmane", "Bou Azzer", "Morocco", "Co", 30.4806, -6.5878, "occurrence", "mindat loc-212416"),
    ("Inguijem", "Bou Azzer", "Morocco", "Co-Ni", 30.5025, -6.6578, "occurrence", "mindat loc-123410"),
    ("Mechoui deposit", "Bou Azzer", "Morocco", "Co-As-Cu", 30.5553, -6.9828, "occurrence", "mindat loc-123388"),
    ("Bleida Far West mine", "Bleida", "Morocco", "Au-Pd-Cu", 30.3894, -6.5142, "mine", "mindat loc-187573"),
    ("Iouriren mine", "Tafraout", "Morocco", "Au", 29.4867, -8.7192, "mine", "mindat locdana-11603"),
    ("Patte d'Oie deposit", "Bou Skour", "Morocco", "Cu-Pb", 30.9006, -6.2786, "mine", "mindat loc-425009"),
    ("Tizi Moudou deposit", "Saghro", "Morocco", "Cu-Ag", 31.1297, -5.5300, "occurrence", "mindat loc-214452"),
    ("Tikhermine", "Ouarzazate", "Morocco", "Cu", 30.9667, -6.9831, "occurrence", "mindat loc-71512"),
    ("Mefis mine", "Tafilalet", "Morocco", "Pb-Cu-Zn", 31.0339, -3.9506, "mine", "mindat loc-16037"),
    ("Irhoud mine", "Youssoufia", "Morocco", "Ba-Cu", 31.8797, -8.8717, "mine", "mindat loc-132981"),
    ("Imiter silver mine", "Imiter", "Morocco", "Ag", 31.3495, -5.7234, "mine", "Wikipedia Imider + image"),
    ("Old copper works Mibladen-Aouli", "Midelt", "Morocco", "Cu", 32.7764, -4.6439, "occurrence", "mindat loc-250673"),
    ("Lhamda mine (Touroug)", "Tafilalet", "Morocco", "F-Ba", 31.5114, -4.6058, "mine", "mindat loc-134746"),
    ("Tasiast gold mine", "Inchiri", "Mauritania", "Au", 20.5694, -15.5044, "mine", "mindat locdetailed-133715"),
    ("Guilabet El Louebda", "Akjoujt", "Mauritania", "Cu-Fe", 19.8997, -14.3500, "occurrence", "mindat loc-308681"),
    ("Legleitat El Khader", "Akjoujt", "Mauritania", "Cu", 19.6442, -14.2453, "occurrence", "mindat loc-308668"),
]


def boxes():
    rng = random.Random(SEED)
    out = []
    for i, (name, area, country, metal, lat, lon, kind, src) in enumerate(DEPOSITS):
        clat = lat + rng.uniform(-MAX_OFFSET, MAX_OFFSET)
        clon = lon + rng.uniform(-MAX_OFFSET, MAX_OFFSET)
        h = BOX_DEG / 2
        out.append({
            "id": "t%02d" % (i + 1), "name": name, "name_ar": name, "area": area, "country": country,
            "metal": metal, "metal_ar": metal, "profile": "alteration",
            "box": [round(clat - h, 4), round(clat + h, 4), round(clon - h, 4), round(clon + h, 4)],
            "known": [{"name": name, "lat": lat, "lon": lon, "kind": kind, "src": src}],
        })
    return out


def chance(tg, valid, box):
    """Probability that a random valid pixel lies within HIT_KM of any target."""
    lat0, lat1, lon0, lon1 = box
    h, w = valid.shape
    lats = lat1 - (np.arange(h) + 0.5) * M.RES
    lons = lon0 + (np.arange(w) + 0.5) * M.RES
    LON, LAT = np.meshgrid(lons, lats)
    near = np.zeros_like(valid)
    kx = 111.32 * math.cos(math.radians((lat0 + lat1) / 2))
    for t in tg:
        d = np.hypot((LAT - t["lat"]) * 111.32, (LON - t["lon"]) * kx)
        near |= d <= M.HIT_KM
    return float((near & valid).sum() / max(valid.sum(), 1))


def poisson_binomial_tail(ps, k):
    """P(X >= k) for X = sum of independent Bernoulli(p_i)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())


def thumb(site, b, score, valid, tg, known, path, size=420):
    from PIL import Image, ImageDraw

    ok = ~np.isnan(b["red"])
    rgb = np.dstack([M.stretch(np.nan_to_num(b[k]), ok, 1, 99.5) for k in ("red", "green", "blue")])
    base = (np.power(rgb, 0.85) * 255 * 0.62).astype("float32")
    s = score / (np.percentile(score[valid], 99.8) + 1e-9)
    heat = M.heat_rgba(np.clip((s - 0.35) / 0.65, 0, 1), valid).astype("float32")
    a = heat[..., 3:4] / 255.0
    img = (base * (1 - a) + heat[..., :3] * a).astype("uint8")
    im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    lat0, lat1, lon0, lon1 = site["box"]

    def xy(lat, lon):
        return ((lon - lon0) / (lon1 - lon0) * size, (lat1 - lat) / (lat1 - lat0) * size)

    r_hit = M.HIT_KM / ((lat1 - lat0) * 111.32) * size
    kx, ky = xy(known["lat"], known["lon"])
    d.ellipse([kx - r_hit, ky - r_hit, kx + r_hit, ky + r_hit], outline=(98, 227, 255), width=2)
    for t in tg:
        x, y = xy(t["lat"], t["lon"])
        r = 9 if t["rank"] <= 3 else 7
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 243, 214) if t["rank"] <= 3 else (255, 184, 92), width=2)
        d.text((x + r + 2, y - 7), t["label"], fill=(255, 220, 160))
    d.polygon([(kx, ky - 8), (kx + 8, ky), (kx, ky + 8), (kx - 8, ky)], outline=(98, 227, 255), fill=(98, 227, 255))
    im.save(path, quality=84)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in boxes():
        print("==", site["id"], site["name"])
        try:
            if a.demo:
                b, used = M.demo_scene(site, seed=int(site["id"][1:]))
            else:
                b, used = M.fetch(site)
            idx, z, score, valid = M.analyse(b, site["profile"])
            tg, anom_km2, thr = M.targets(score, valid, site["box"])
            bt = M.blind_test(site, tg, score, valid)[0]
            for t in tg:
                t.pop("_comp", None)
            p = chance(tg, valid, site["box"])
            thumb(site, b, score, valid, tg, site["known"][0], OUT / (site["id"] + ".jpg"))
        except Exception as e:  # noqa: BLE001
            print("[fail]", site["id"], e)
            continue
        k = site["known"][0]
        rows.append({
            "id": site["id"], "name": site["name"], "area": site["area"], "country": site["country"],
            "metal": site["metal"], "kind": k["kind"], "src": k["src"], "lat": k["lat"], "lon": k["lon"],
            "box": site["box"], "hit": bt["hit"], "nearest_km": bt["nearest_km"],
            "nearest_target": bt["nearest_target"], "rank": bt["best_rank_within"],
            "percentile": bt["percentile"], "p_chance": round(p, 4), "valid_pct": round(100 * float(valid.mean()), 1),
            "scenes": [s["date"] for s in used], "img": "test/" + site["id"] + ".jpg",
        })
        print("   hit=%s nearest=%s km  percentile=%s  chance=%.1f%%" % (bt["hit"], bt["nearest_km"],
                                                                         bt["percentile"], 100 * p))
    if not rows:
        sys.exit("no site processed")
    hits = sum(r["hit"] for r in rows)
    expected = sum(r["p_chance"] for r in rows)
    pval = poisson_binomial_tail([r["p_chance"] for r in rows], hits)
    pct = [r["percentile"] for r in rows if r["percentile"] is not None]
    summary = {
        "n": len(rows), "hits": hits, "expected_by_chance": round(expected, 2),
        "lift": round(hits / expected, 2) if expected else None, "p_value": round(pval, 4),
        "median_percentile": round(float(np.median(pct)), 1) if pct else None,
        "hits_mines": sum(r["hit"] for r in rows if r["kind"] == "mine"),
        "n_mines": sum(r["kind"] == "mine" for r in rows),
        "hits_occ": sum(r["hit"] for r in rows if r["kind"] == "occurrence"),
        "n_occ": sum(r["kind"] == "occurrence" for r in rows),
    }
    data = {"generated": date.today().isoformat(), "demo": a.demo, "hit_km": M.HIT_KM, "top_n": M.N_TARGETS,
            "box_km": round(BOX_DEG * 111), "summary": summary, "sites": rows}
    (OUT / "test.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    page = (M.ROOT / "minerals_test.html").read_text(encoding="utf-8")
    (M.OUT / "test.html").write_text(page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)),
                                     encoding="utf-8")
    print("[ok] %d/%d hits, expected by chance %.2f, p=%.4f" % (hits, len(rows), expected, pval))


if __name__ == "__main__":
    main()
