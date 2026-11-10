"""
Terrain for the 3D view of each MADAR Minerals site: Copernicus DEM GLO-30 (ESA, 30 m, free on AWS).

For every site in docs/minerals/minerals.json it writes docs/minerals/<site>/dem.bin (little-endian uint16,
row 0 = north, metres above the site minimum) and dem.json (size, bounds, min/max height).
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "minerals"
URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
MAX_SIDE = 360          # grid points on the longest side (~80-100 m spacing on a 30 km site)


def tile_name(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def site_dem(bounds):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    (lat0, lon0), (lat1, lon1) = bounds
    kx = (lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2))
    ky = lat1 - lat0
    if kx >= ky:
        w, h = MAX_SIDE, max(2, round(MAX_SIDE * ky / kx))
    else:
        h, w = MAX_SIDE, max(2, round(MAX_SIDE * kx / ky))
    tr = from_bounds(lon0, lat0, lon1, lat1, w, h)
    out = np.full((h, w), np.nan, "float32")
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES")
    with env:
        for la in range(math.floor(lat0), math.floor(lat1) + 1):
            for lo in range(math.floor(lon0), math.floor(lon1) + 1):
                n = tile_name(la, lo)
                try:
                    with rasterio.open(URL.format(n=n)) as src, WarpedVRT(
                            src, crs="EPSG:4326", transform=tr, width=w, height=h,
                            resampling=Resampling.bilinear, nodata=-9999) as v:
                        a = v.read(1).astype("float32")
                except Exception as e:  # noqa: BLE001
                    print("[skip]", n, e)
                    continue
                a[a <= -9000] = np.nan
                fill = np.isnan(out) & np.isfinite(a)
                out[fill] = a[fill]
                print("[ok]", n)
    return out


def main():
    data = json.loads((OUT / "minerals.json").read_text(encoding="utf-8"))
    for s in data["sites"]:
        dem = site_dem(s["bounds"])
        if not np.isfinite(dem).any():
            print("[fail]", s["id"])
            continue
        lo, hi = float(np.nanmin(dem)), float(np.nanmax(dem))
        dem = np.nan_to_num(dem, nan=lo)
        q = np.clip(np.round(dem - lo), 0, 65535).astype("<u2")
        folder = OUT / s["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "dem.bin").write_bytes(q.tobytes())
        meta = {"w": int(q.shape[1]), "h": int(q.shape[0]), "min": round(lo, 1), "max": round(hi, 1),
                "bounds": s["bounds"], "src": "Copernicus DEM GLO-30 (ESA), AWS open data"}
        (folder / "dem.json").write_text(json.dumps(meta))
        print(f"[dem] {s['id']}: {meta['w']}x{meta['h']}, {lo:.0f}-{hi:.0f} m")


if __name__ == "__main__":
    main()
