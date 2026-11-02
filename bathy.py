"""
Sea-floor depth from GEBCO (General Bathymetric Chart of the Oceans), 15 arc-second grid (~450 m),
served by NOAA ERDDAP. Downloaded once per area and cached in data/bathy_<name>.npz (committed by the workflow).

GEBCO: free to use and redistribute with attribution. NOT for navigation: it does not show small rocks,
wrecks or recent changes. Depth is shown as an aid only.
"""
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data"
BASES = ["https://coastwatch.pfeg.noaa.gov/erddap/griddap/GEBCO_2020",
         "https://upwell.pfeg.noaa.gov/erddap/griddap/GEBCO_2020"]
UA = {"User-Agent": "Mozilla/5.0 (dakhla-radar; +https://github.com/Negjir/dakhla-radar)"}
ATTRIBUTION = "GEBCO Compilation Group (2020) GEBCO 2020 Grid"


def _download(lat0, lat1, lon0, lon1, stride):
    import xarray as xr

    q = f".nc?elevation[({lat0:.4f}):{stride}:({lat1:.4f})][({lon0:.4f}):{stride}:({lon1:.4f})]"
    err = None
    for base in BASES:
        try:
            req = urllib.request.Request(base + q, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
                f.write(data)
                path = f.name
            ds = xr.open_dataset(path)
            return ds["latitude"].values, ds["longitude"].values, ds["elevation"].values.astype("float32")
        except Exception as e:  # noqa: BLE001
            err = e
            print(f"[skip] GEBCO from {base.split('/')[2]}: {e}")
    raise RuntimeError(err)


def load(name, lat0, lat1, lon0, lon1, stride):
    """Return (lats ascending, lons, elevation m; negative = below sea level), from cache or GEBCO."""
    f = CACHE / f"bathy_{name}.npz"
    want = np.array([lat0, lat1, lon0, lon1, stride], dtype="float64")
    if f.exists():
        z = np.load(f)
        if np.allclose(z["box"], want):
            return z["lats"], z["lons"], z["elev"]
    lats, lons, elev = _download(lat0, lat1, lon0, lon1, stride)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(f, lats=lats, lons=lons, elev=elev, box=want)
    print(f"[ok] GEBCO {name}: {elev.shape} cached in {f.name}")
    return lats, lons, elev


def depth_on(grid, qlats, qlons):
    """Depth in metres (positive) on the query grid; NaN on land."""
    from scipy.interpolate import RegularGridInterpolator

    lats, lons, elev = grid
    if lats[0] > lats[-1]:
        lats, elev = lats[::-1], elev[::-1]
    f = RegularGridInterpolator((lats, lons), elev, bounds_error=False, fill_value=np.nan)
    G1, G2 = np.meshgrid(qlats, qlons, indexing="ij")
    e = f(np.stack([G1, G2], -1))
    d = -e
    d[~(d > 0)] = np.nan
    return d.astype("float32")


def demo_depth(qlats, qlons, coast_lat, coast_lon):
    """Synthetic shelf for --demo runs: shallow near the coast, shelf edge ~60-100 km out, then deep."""
    LON, LAT = np.meshgrid(qlons, qlats)
    coast = np.interp(LAT, coast_lat, coast_lon)
    km = (coast - LON) * 102
    d = np.where(km < 70, 5 + km * 1.6, 120 + np.clip(km - 70, 0, None) ** 1.6 * 3)
    d[km < 0] = np.nan
    return d.astype("float32")


def safe(name, lat0, lat1, lon0, lon1, stride, qlats, qlons):
    """Depth on (qlats, qlons), or None if GEBCO cannot be reached today (the radar still runs)."""
    try:
        return depth_on(load(name, lat0, lat1, lon0, lon1, stride), qlats, qlons)
    except Exception as e:  # noqa: BLE001
        print("[warn] no bathymetry:", e)
        return None
