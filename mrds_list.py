"""
List known metal deposits in Morocco and Mauritania from the USGS Mineral Resources Data System (MRDS),
to build an unbiased test sample for MADAR Minerals. Writes data/mrds_candidates.json.
"""
import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
OUT = Path(__file__).resolve().parent / "data" / "mrds_candidates.json"
METALS = ("Copper", "Cobalt", "Gold", "Silver", "Zinc", "Lead", "Nickel", "Manganese", "Iron")


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (dakhla-radar)"})
    data = urllib.request.urlopen(req, timeout=300).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    rows = csv.DictReader(io.TextIOWrapper(z.open(name), encoding="latin-1"))
    out = []
    for r in rows:
        if r.get("country") not in ("Morocco", "Mauritania", "Western Sahara"):
            continue
        com = " ".join(r.get(k, "") for k in ("commod1", "commod2", "commod3"))
        if not any(m in com for m in METALS):
            continue
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
        except (ValueError, KeyError):
            continue
        out.append({k: r.get(k, "") for k in ("dep_id", "site_name", "country", "commod1", "commod2", "commod3",
                                               "dev_stat", "dep_type", "model", "alteration", "work_type",
                                               "prod_size")} | {"lat": lat, "lon": lon})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print("[ok] %d candidates" % len(out))


if __name__ == "__main__":
    main()
