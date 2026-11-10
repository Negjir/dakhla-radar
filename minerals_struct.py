"""
MADAR Minerals - structural layer from the terrain (Copernicus DEM 30 m, see minerals_dem.py).

Ore follows structure: faults and fractures carry the hot fluids (gold, copper, cobalt), and carbonatites
come up as ring-shaped intrusions. Sentinel-2 sees colour; the terrain shows the structure. For every site:

  1. LINEAMENTS: valleys and scarps that are straight over a distance. Found with a Laplacian-of-Gaussian
     of the DEM (valleys = positive curvature) kept only where the structure tensor says the feature is
     linear (coherence). Coloured by direction.
  2. ROSE DIAGRAM: the dominant directions of the lineaments (10 degree bins), i.e. the fault families.
  3. RINGS: circular structures 1-6 km across, found with a circular Hough-type vote on the edge map and
     kept only when the ring is closed on most of its circumference.
  4. STRUCTURAL SUPPORT of each target: distance to the nearest strong lineament and to the nearest ring.

Writes docs/minerals/<site>/struct.png, adds "structure" to each site and "struct_km"/"ring_km" to each
target in minerals.json, then rebuilds the console page.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "minerals"


def load_dem(site_id):
    meta = json.loads((OUT / site_id / "dem.json").read_text())
    h = np.frombuffer((OUT / site_id / "dem.bin").read_bytes(), dtype="<u2").reshape(meta["h"], meta["w"]).astype("float32")
    return h + meta["min"], meta


def pixel_m(meta):
    (a0, o0), (a1, o1) = meta["bounds"]
    dx = (o1 - o0) * 111320 * math.cos(math.radians((a0 + a1) / 2)) / (meta["w"] - 1)
    dy = (a1 - a0) * 111320 / (meta["h"] - 1)
    return dx, dy


def hillshade(z, dx, dy, az, alt=40):
    gy, gx = np.gradient(z, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, e = math.radians(az), math.radians(alt)
    return np.clip(np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def lineaments(z, dx, dy):
    px = (dx + dy) / 2
    zs = ndimage.gaussian_filter(z, 1.0)
    log = ndimage.gaussian_laplace(z, sigma=1.6)            # >0 valleys, <0 ridges
    gy, gx = np.gradient(zs, dy, dx)
    # structure tensor -> orientation of linear features and how linear they are (coherence)
    s = 2.5
    jxx, jyy, jxy = (ndimage.gaussian_filter(v, s) for v in (gx * gx, gy * gy, gx * gy))
    tmp = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    l1, l2 = (jxx + jyy + tmp) / 2, (jxx + jyy - tmp) / 2
    coh = np.where(l1 + l2 > 0, (l1 - l2) / (l1 + l2 + 1e-12), 0)
    # gradient direction (degrees from north, image rows go south): feature direction is perpendicular
    grad_dir = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))       # angle of gradient vs x (east)
    # rows run south, so the angle is measured from east towards south; a valley runs perpendicular to the
    # gradient, which makes its azimuth (from north, clockwise) equal to that angle modulo 180.
    feat_az = grad_dir % 180
    strength = np.abs(log)
    valley = log > 0
    thr = np.percentile(strength[valley], 88) if valley.any() else 1
    line = valley & (strength >= thr) & (coh > 0.45)
    ridge_thr = np.percentile(strength[~valley], 92) if (~valley).any() else 1
    ridge = (~valley) & (strength >= ridge_thr) & (coh > 0.45)
    edges = line | ridge
    return line, ridge, edges, feat_az, strength, coh


def rose(line, az, strength, bins=18):
    h, _ = np.histogram(az[line], bins=bins, range=(0, 180), weights=strength[line])
    h = h / (h.max() + 1e-9)
    top = int(np.argmax(h))
    return [round(float(v), 3) for v in h], top * (180 // bins) + (180 // bins) / 2


def rings(edges, az, px_m, rmin_km=1.2, rmax_km=5.0, n_best=4):
    """Ring structures 2.4-10 km across. A candidate counts only if (a) edges are denser on the ring than
    inside and outside it, (b) the edges on the ring run tangentially (as around an intrusion, not across
    it), and (c) the ring is closed on most of its circumference. Random terrain fails (b) and (c)."""
    from numpy.fft import irfft2, rfft2

    h, w = edges.shape
    E = edges.astype("float32")
    fE = rfft2(E, s=(h * 2, w * 2))

    def conv(kern, R):
        k = np.zeros((h * 2, w * 2), "float32")
        k[:2 * R + 1, :2 * R + 1] = kern
        return irfft2(fE * rfft2(k), s=(h * 2, w * 2))[R:R + h, R:R + w]

    cands = []
    for r_km in np.arange(rmin_km, rmax_km + 0.01, 0.3):
        r = r_km * 1000 / px_m
        if r > min(h, w) / 2.4:
            break
        R = int(math.ceil(r * 1.45 + 2))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        d = np.hypot(yy, xx)
        ring = (np.abs(d - r) <= 1.5).astype("float32")
        inner = (d <= 0.6 * r).astype("float32")
        outer = ((d >= 1.25 * r) & (d <= 1.45 * r)).astype("float32")
        on = conv(ring, R) / ring.sum()
        ins = conv(inner, R) / inner.sum()
        out = conv(outer, R) / outer.sum()
        contrast = on / (np.maximum(ins, out) + 0.01)
        m = int(r * 1.45) + 1
        contrast[:m], contrast[-m:], contrast[:, :m], contrast[:, -m:] = 0, 0, 0, 0
        mx = ndimage.maximum_filter(contrast, size=int(r))
        for (j, i) in np.argwhere((contrast == mx) & (contrast > 1.6) & (on > 0.12)):
            ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            closed, tang, n = 0, 0, 0
            for a in ang:
                found = False
                for dr in (-1, 0, 1):
                    y, x = int(round(j - (r + dr) * math.cos(a))), int(round(i + (r + dr) * math.sin(a)))
                    if 0 <= y < h and 0 <= x < w and edges[y, x]:
                        found = True
                        tangent_az = (math.degrees(a) + 90) % 180          # direction of the circle at this point
                        diff = abs(((az[y, x] - tangent_az) + 90) % 180 - 90)
                        n += 1
                        tang += diff <= 30
                        break
                closed += found
            closure = closed / len(ang)
            tfrac = tang / n if n else 0
            if closure >= 0.6 and tfrac >= 0.5:
                cands.append((float(contrast[j, i]) * closure * tfrac, int(j), int(i), float(r), closure, tfrac))
    cands.sort(reverse=True)
    out = []
    for c in cands:
        if any(math.hypot(c[1] - o[1], c[2] - o[2]) < max(c[3], o[3]) for o in out):
            continue
        out.append(c)
        if len(out) >= n_best:
            break
    return out


def to_latlon(j, i, meta):
    (a0, o0), (a1, o1) = meta["bounds"]
    return a1 - (a1 - a0) * j / (meta["h"] - 1), o0 + (o1 - o0) * i / (meta["w"] - 1)


def az_colour(az):
    """Cyclic colour for directions (0 and 180 the same)."""
    t = np.radians(az * 2)
    r = 0.5 + 0.5 * np.cos(t)
    g = 0.5 + 0.5 * np.cos(t - 2.094)
    b = 0.5 + 0.5 * np.cos(t + 2.094)
    return np.stack([r, g, b], -1)


def picture(z, dx, dy, line, ridge, az, rg, meta, path, width=1600):
    from PIL import Image, ImageDraw

    ex = 3.0 if np.ptp(z) < 400 else 1.5                      # flat desert: exaggerate the relief
    hs = 0.6 * hillshade(z * ex, dx, dy, 315) + 0.4 * hillshade(z * ex, dx, dy, 15)
    shade = (0.35 + 0.95 * hs)[..., None]
    tc = OUT / meta["site"] / "truecolor.jpg"
    if tc.exists():                                            # the real colours of the ground, shaded by relief
        from PIL import Image as _I
        col0 = np.asarray(_I.open(tc).convert("RGB").resize((meta["w"], meta["h"]), _I.BILINEAR), "float32") / 255
        base = np.clip(col0 * 0.7 * shade, 0, 1)
    else:
        base = np.clip(shade * np.array([0.55, 0.58, 0.55]), 0, 1)
    col = az_colour(az)
    img = base.copy()
    img[ridge] = img[ridge] * 0.55 + np.array([0.95, 0.92, 0.85]) * 0.45
    img[line] = col[line] * 0.9 + 0.1
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).resize(
        (width, round(width * meta["h"] / meta["w"])), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    sx, sy = im.width / meta["w"], im.height / meta["h"]
    for k, (_, j, i, r, c, tf) in enumerate(rg):
        for wdt, colr in ((9, (12, 31, 27)), (4, (228, 200, 142))):
            d.ellipse([(i - r) * sx, (j - r) * sy, (i + r) * sx, (j + r) * sy], outline=colr, width=wdt)
        d.text(((i + r * 0.72) * sx, (j - r * 0.72) * sy), f"R{k + 1}", fill=(228, 200, 142))
    im.save(path, quality=86, optimize=True)


def main():
    data = json.loads((OUT / "minerals.json").read_text(encoding="utf-8"))
    for s in data["sites"]:
        if not (OUT / s["id"] / "dem.bin").exists():
            print("[skip] no DEM", s["id"])
            continue
        z, meta = load_dem(s["id"])
        meta["site"] = s["id"]
        dx, dy = pixel_m(meta)
        line, ridge, edges, az, strength, coh = lineaments(z, dx, dy)
        rs, dom = rose(line, az, strength)
        rg = rings(edges, az, (dx + dy) / 2)
        picture(z, dx, dy, line, ridge, az, rg, meta, OUT / s["id"] / "struct.jpg")
        rings_out = []
        for k, (score, j, i, r, c, tf) in enumerate(rg):
            la, lo = to_latlon(j, i, meta)
            rings_out.append({"id": f"R{k + 1}", "lat": round(la, 5), "lon": round(lo, 5),
                              "radius_km": round(r * (dx + dy) / 2 / 1000, 2), "closure": round(c, 2),
                              "tangential": round(tf, 2), "score": round(score, 2)})
        # structural support of each target
        dist = ndimage.distance_transform_edt(~line, sampling=(dy / 1000, dx / 1000))
        for t in s["targets"]:
            (a0, o0), (a1, o1) = meta["bounds"]
            j = int(round((a1 - t["lat"]) / (a1 - a0) * (meta["h"] - 1)))
            i = int(round((t["lon"] - o0) / (o1 - o0) * (meta["w"] - 1)))
            j, i = min(max(j, 0), meta["h"] - 1), min(max(i, 0), meta["w"] - 1)
            t["struct_km"] = round(float(dist[j, i]), 2)
            near = [abs(math.hypot((t["lat"] - r_["lat"]) * 111.32, (t["lon"] - r_["lon"]) * 111.32 * math.cos(math.radians(t["lat"])))
                        - r_["radius_km"]) for r_ in rings_out]
            t["ring_km"] = round(min(near), 2) if near else None
        s["images"]["struct.jpg"] = f"{s['id']}/struct.jpg"
        s["structure"] = {"rose": rs, "dominant_az": dom, "rings": rings_out,
                          "lineament_km": round(float(line.sum()) * (dx + dy) / 2 / 1000, 1),
                          "src": "Copernicus DEM GLO-30, curvature + structure tensor + circular vote"}
        print(f"[struct] {s['id']}: dominant {dom:.0f} deg, {len(rings_out)} rings, "
              f"targets on structure (<0.5 km): {sum(1 for t in s['targets'] if t['struct_km'] < 0.5)}/{len(s['targets'])}")
    (OUT / "minerals.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    page = (ROOT / "minerals_console.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


if __name__ == "__main__":
    main()
