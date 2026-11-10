"""Madar Offshore - private trip report for one vessel, from its own track.

Reads the vessel's position log (GPX, KML, NMEA, or CSV with time/lat/lon), splits it into trips from the port,
labels every minute as port / transit / searching / fishing, adds up hours, distance, fuel and money, and puts each
fishing set next to the Madar zones of the same day (docs/archive/zones/<day>.json, rebuilt by the back-fill run).

The track is the owner's confidential data: run this on a private machine, keep the files under private/ (git-ignored)
and send the HTML report to the owner directly. Nothing here uploads anything.

    python tools/vessel_report.py private/elouali.gpx --name ELOUALI --config private/elouali.json
    python tools/vessel_report.py --demo            # synthetic two-month track, to try the report

Config (JSON, every key optional):
    {"fuel_lph": {"port": 4, "transit": 140, "search": 100, "fishing": 70},  # litres per hour per state
     "diesel_mad_l": 11.0,                                                   # price paid per litre
     "catch_csv": "private/catch.csv",                                       # date,kg[,value_mad]
     "captain_csv": "private/captain.csv",                                   # the captain's own trip log, to check it
     "fuel_tolerance": 0.25}                                                 # fuel gap accepted before a flag
"""
import argparse
import csv
import datetime as dt
import html
import json
import math
import pathlib
import random
import re
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = {"name": "ميناء الداخلة", "lat": 23.705, "lon": -15.925}
PORT_NM = 1.5            # inside this distance of the port = in port
GAP_H = 3.0              # a hole longer than this in the log is not interpolated
TRANSIT_KN = 7.0
FISH_KN = 2.5
SET_MIN = 15             # a fishing stretch shorter than this is not counted as a set
OTHER_PORTS = {"بوجدور": (26.125, -14.485), "العيون (المرسى)": (27.090, -13.420), "طرفاية": (27.940, -12.925),
               "لكويرة": (20.840, -17.090), "نواذيبو": (20.905, -17.035), "أنزران": (23.89, -15.80)}
MATCH_H = 12             # a declared trip is paired with a tracked one leaving within this many hours
DEFAULT = {"fuel_lph": {"port": 4, "transit": 140, "search": 100, "fishing": 70}, "diesel_mad_l": 11.0}
STATES = ["port", "transit", "search", "fishing"]
AR = {"port": "في الميناء", "transit": "إبحار مباشر", "search": "بحث", "fishing": "صيد"}


def nm(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 3440.065 * math.asin(math.sqrt(min(1, h)))


def bearing(a, b):
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    return math.degrees(math.atan2(math.sin(dl) * math.cos(la2),
                                   math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dl))) % 360


def when(s):
    s = s.strip().replace("Z", "+00:00")
    for f in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            t = dt.datetime.fromisoformat(s) if f is None else dt.datetime.strptime(s, f)
            return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    if re.fullmatch(r"\d{9,11}(\.\d+)?", s):
        return dt.datetime.fromtimestamp(float(s), dt.timezone.utc)
    raise ValueError(f"unknown time format: {s!r}")


# ---------------------------------------------------------------- reading the log
def read_gpx(text):
    out = []
    for el in ET.fromstring(text).iter():
        if el.tag.split("}")[-1] in ("trkpt", "rtept", "wpt") and el.get("lat"):
            t = next((c.text for c in el if c.tag.split("}")[-1] == "time"), None)
            if t:
                out.append((when(t), float(el.get("lat")), float(el.get("lon"))))
    return out


def read_kml(text):
    out, root = [], ET.fromstring(text)
    whens = [e.text for e in root.iter() if e.tag.split("}")[-1] == "when"]
    coords = [e.text for e in root.iter() if e.tag.split("}")[-1] == "coord"]
    for w, c in zip(whens, coords):           # gx:Track
        lon, lat = map(float, c.split()[:2])
        out.append((when(w), lat, lon))
    if not out:
        for pm in root.iter():
            if pm.tag.split("}")[-1] != "Placemark":
                continue
            w = next((e.text for e in pm.iter() if e.tag.split("}")[-1] in ("when", "begin")), None)
            c = next((e.text for e in pm.iter() if e.tag.split("}")[-1] == "coordinates"), None)
            if w and c:
                lon, lat = map(float, c.strip().split(",")[:2])
                out.append((when(w), lat, lon))
    return out


def read_nmea(text):
    out = []
    for line in text.splitlines():
        p = line.strip().split("*")[0].split(",")
        if len(p) < 10 or not p[0].endswith("RMC") or p[2] != "A":
            continue
        try:
            lat = int(p[3][:2]) + float(p[3][2:]) / 60
            lon = int(p[5][:3]) + float(p[5][3:]) / 60
            lat, lon = (-lat if p[4] == "S" else lat), (-lon if p[6] == "W" else lon)
            d, t = p[9], p[1]
            stamp = dt.datetime(2000 + int(d[4:6]), int(d[2:4]), int(d[:2]), int(t[:2]), int(t[2:4]),
                                int(float(t[4:])), tzinfo=dt.timezone.utc)
            out.append((stamp, lat, lon))
        except (ValueError, IndexError):
            continue
    return out


def read_csv(text):
    rows = list(csv.reader(text.splitlines(), delimiter=";" if text.count(";") > text.count(",") else ","))
    head = [h.strip().lower() for h in rows[0]]

    def col(*names):
        return next((i for i, h in enumerate(head) if any(n in h for n in names)), None)
    it, ila, ilo = col("time", "date", "timestamp", "utc"), col("lat"), col("lon", "lng")
    if None in (it, ila, ilo):
        raise SystemExit(f"CSV needs time, lat and lon columns; found {head}")
    out = []
    for r in rows[1:]:
        try:
            out.append((when(r[it]), float(r[ila]), float(r[ilo])))
        except (ValueError, IndexError):
            continue
    return out


def read_track(path):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    ext = pathlib.Path(path).suffix.lower()
    pts = (read_gpx(text) if ext == ".gpx" else read_kml(text) if ext == ".kml"
           else read_nmea(text) if ext in (".nmea", ".txt", ".log") else read_csv(text))
    pts = sorted({p[0]: p for p in pts if abs(p[1]) <= 90 and abs(p[2]) <= 180}.values())
    if len(pts) < 10:
        raise SystemExit(f"only {len(pts)} positions read from {path}")
    return pts


# ---------------------------------------------------------------- labelling
def label(pts):
    """One row per position: speed (kn), heading change, state."""
    rows = []
    for k, (t, la, lo) in enumerate(pts):
        a = pts[max(0, k - 1)]
        b = pts[min(len(pts) - 1, k + 1)]
        h = (b[0] - a[0]).total_seconds() / 3600
        sog = nm(a[1:], b[1:]) / h if h > 0 and h < GAP_H else 0.0
        rows.append({"t": t, "lat": la, "lon": lo, "sog": sog, "hdg": bearing(a[1:], b[1:])})
    for k, r in enumerate(rows):   # median of 5 removes single GPS jumps
        r["sog"] = sorted(x["sog"] for x in rows[max(0, k - 2):k + 3])[len(rows[max(0, k - 2):k + 3]) // 2]
    for k, r in enumerate(rows):
        a, b = rows[max(0, k - 3)], rows[min(len(rows) - 1, k + 3)]
        turn = abs((b["hdg"] - a["hdg"] + 180) % 360 - 180)
        if nm((r["lat"], r["lon"]), (PORT["lat"], PORT["lon"])) <= PORT_NM:
            r["state"] = "port"
        elif r["sog"] < FISH_KN:
            r["state"] = "fishing"
        elif r["sog"] >= TRANSIT_KN and turn < 25:
            r["state"] = "transit"
        else:
            r["state"] = "search"
    # a short stop while searching is not a set, and a set needs SET_MIN minutes
    k = 0
    while k < len(rows):
        if rows[k]["state"] != "fishing":
            k += 1
            continue
        j = k
        while j + 1 < len(rows) and rows[j + 1]["state"] == "fishing":
            j += 1
        if (rows[j]["t"] - rows[k]["t"]).total_seconds() / 60 < SET_MIN:
            for x in rows[k:j + 1]:
                x["state"] = "search"
        k = j + 1
    return rows


def trips_of(rows):
    trips, cur = [], None
    for k, r in enumerate(rows):
        gap = False
        if r["state"] == "port":
            if cur and len(cur) > 5:
                trips.append(cur)
            cur = None
            if r["state"] == "port":
                continue
        if cur is None:
            cur = [rows[k - 1]] if k and rows[k - 1]["state"] == "port" and not gap else []
        cur.append(r)
    if cur and len(cur) > 5:
        trips.append(cur)
    # a few minutes outside the port circle (moving berth, GPS drift) is not a trip
    return [t for t in trips if (t[-1]["t"] - t[0]["t"]).total_seconds() > 2 * 3600
            and max(nm((x["lat"], x["lon"]), (PORT["lat"], PORT["lon"])) for x in t) > 5]


# ---------------------------------------------------------------- Madar zones of the day
def zones_of(day, zdir):
    p = zdir / f"{day}.json"
    if not p.exists():
        return None
    z = json.loads(p.read_text())
    return z if z.get("zones") is not None else None


def score_at(z, la, lo):
    g = z.get("grid") or {}
    if not g:
        return None
    i = round((g["lat0"] - la) / g["cell"])
    j = round((lo - g["lon0"]) / g["cell"])
    if not (0 <= i < g["n"] and 0 <= j < g["m"]):
        return None
    v = g["score"][i * g["m"] + j]
    return None if v is None else v / 99


def percentile(z, s):
    sea = [v for v in z["grid"]["score"] if v is not None]
    return round(100 * sum(1 for v in sea if v / 99 < s) / len(sea)) if sea and s is not None else None


# ---------------------------------------------------------------- the report
def analyse(rows, cfg, zdir, catch):
    fuel = {**DEFAULT["fuel_lph"], **cfg.get("fuel_lph", {})}
    price = cfg.get("diesel_mad_l", DEFAULT["diesel_mad_l"])
    out = []
    for n, tr in enumerate(trips_of(rows), 1):
        hours = {s: 0.0 for s in STATES}
        dist = {s: 0.0 for s in STATES}
        for a, b in zip(tr, tr[1:]):
            h = (b["t"] - a["t"]).total_seconds() / 3600
            if h <= 0 or h > GAP_H:
                continue
            hours[a["state"]] += h
            dist[a["state"]] += nm((a["lat"], a["lon"]), (b["lat"], b["lon"]))
        # fishing sets
        sets, k = [], 0
        while k < len(tr):
            if tr[k]["state"] != "fishing":
                k += 1
                continue
            j = k
            while j + 1 < len(tr) and tr[j + 1]["state"] == "fishing":
                j += 1
            seg = tr[k:j + 1]
            la = sum(x["lat"] for x in seg) / len(seg)
            lo = sum(x["lon"] for x in seg) / len(seg)
            day = seg[0]["t"].date().isoformat()
            s = {"t": seg[0]["t"].strftime("%Y-%m-%d %H:%M"), "day": day, "lat": round(la, 4), "lon": round(lo, 4),
                 "min": round((seg[-1]["t"] - seg[0]["t"]).total_seconds() / 60),
                 "port_nm": round(nm((la, lo), (PORT["lat"], PORT["lon"])), 1)}
            z = zones_of(day, zdir)
            if z:
                near = min(z["zones"], key=lambda q: nm((la, lo), (q["lat"], q["lon"])), default=None)
                if near:
                    s["zone"] = near["id"]
                    s["zone_nm"] = round(nm((la, lo), (near["lat"], near["lon"])), 1)
                    s["zone_conf"] = near.get("confidence")
                sc = score_at(z, la, lo)
                s["score_pct"] = percentile(z, sc)
            sets.append(s)
            k = j + 1
        # how far the boat sailed before the first set, compared with a straight line
        first = next((k for k, x in enumerate(tr) if x["state"] == "fishing"), None)
        detour = None
        if first:
            sailed = sum(nm((a["lat"], a["lon"]), (b["lat"], b["lon"])) for a, b in zip(tr[:first], tr[1:first + 1])
                         if (b["t"] - a["t"]).total_seconds() / 3600 <= GAP_H)
            straight = nm((PORT["lat"], PORT["lon"]), (tr[first]["lat"], tr[first]["lon"]))
            detour = {"sailed_nm": round(sailed, 1), "straight_nm": round(straight, 1),
                      "extra_nm": round(max(0, sailed - straight), 1),
                      "extra_h": round(max(0, sailed - straight) / 9.0, 1)}
        gaps, stops = [], []
        for a, b in zip(tr, tr[1:]):
            h = (b["t"] - a["t"]).total_seconds() / 3600
            if h > 1.0:
                gaps.append({"from": a["t"].strftime("%Y-%m-%d %H:%M"), "h": round(h, 1),
                             "lat": round(a["lat"], 3), "lon": round(a["lon"], 3),
                             "jump_nm": round(nm((a["lat"], a["lon"]), (b["lat"], b["lon"])), 1),
                             "port_nm": round(nm((a["lat"], a["lon"]), (PORT["lat"], PORT["lon"])), 1)})
        k = 0
        while k < len(tr):                         # stationary > 1 h (under 1 kn)
            if tr[k]["sog"] >= 1.0 or tr[k]["state"] == "port":
                k += 1
                continue
            j = k
            while j + 1 < len(tr) and tr[j + 1]["sog"] < 1.0:
                j += 1
            h = (tr[j]["t"] - tr[k]["t"]).total_seconds() / 3600
            if h >= 1.0:
                la, lo = tr[k]["lat"], tr[k]["lon"]
                near = min(OTHER_PORTS.items(), key=lambda q: nm((la, lo), q[1]))
                d = nm((la, lo), near[1])
                if d <= 3 or h >= 6:
                    stops.append({"t": tr[k]["t"].strftime("%Y-%m-%d %H:%M"), "h": round(h, 1), "lat": round(la, 3),
                                  "lon": round(lo, 3), "port": near[0] if d <= 3 else None, "port_nm": round(d, 1)})
            k = j + 1
        litres = sum(hours[s] * fuel[s] for s in STATES)
        d0 = tr[0]["t"].date().isoformat()
        d1 = tr[-1]["t"].date().isoformat()
        c = [x for x in catch if d0 <= x["date"] <= d1]
        kg = sum(x["kg"] for x in c) if c else None
        value = sum(x.get("mad") or 0 for x in c) if c and any(x.get("mad") for x in c) else None
        z0 = zones_of(d0, zdir)
        out.append({
            "n": n, "start": tr[0]["t"].strftime("%Y-%m-%d %H:%M"), "end": tr[-1]["t"].strftime("%Y-%m-%d %H:%M"),
            "day": d0, "hours": {s: round(v, 2) for s, v in hours.items()},
            "nm": round(sum(dist.values()), 1), "nm_by": {s: round(v, 1) for s, v in dist.items()},
            "max_nm": round(max(nm((x["lat"], x["lon"]), (PORT["lat"], PORT["lon"])) for x in tr), 1),
            "litres": round(litres), "mad": round(litres * price), "sets": sets, "detour": detour,
            "gaps": gaps, "stops": stops, "t0": tr[0]["t"].isoformat(), "t1": tr[-1]["t"].isoformat(),
            "kg": kg, "value": value, "zones": (z0 or {}).get("zones"), "has_zones": bool(z0),
            "path": [[round(x["lat"], 4), round(x["lon"], 4), STATES.index(x["state"])] for x in tr[::max(1, len(tr) // 1500)]],
        })
    return out, fuel, price


def summary(trips):
    tot = {s: round(sum(t["hours"][s] for t in trips), 1) for s in STATES}
    sea = sum(tot[s] for s in ("transit", "search", "fishing")) or 1
    sets = [s for t in trips for s in t["sets"]]
    z = [s for s in sets if "zone_nm" in s]
    sp = [s["score_pct"] for s in sets if s.get("score_pct") is not None]
    return {
        "trips": len(trips), "hours": tot, "sea_h": round(sea, 1),
        "search_share": round(100 * tot["search"] / sea), "fishing_share": round(100 * tot["fishing"] / sea),
        "nm": round(sum(t["nm"] for t in trips)), "litres": sum(t["litres"] for t in trips),
        "mad": sum(t["mad"] for t in trips), "sets": len(sets),
        "search_mad": None,  # filled below
        "extra_nm": round(sum((t["detour"] or {}).get("extra_nm", 0) for t in trips)),
        "compared": len(z), "near10": sum(1 for s in z if s["zone_nm"] <= 10),
        "near20": sum(1 for s in z if s["zone_nm"] <= 20),
        "score_pct_median": sorted(sp)[len(sp) // 2] if sp else None,
        "kg": sum(t["kg"] for t in trips if t["kg"]) or None,
    }


def read_catch(path):
    if not path or not pathlib.Path(path).exists():
        return []
    out = []
    for r in csv.DictReader(pathlib.Path(path).read_text(encoding="utf-8").splitlines()):
        try:
            out.append({"date": when(r["date"]).date().isoformat(), "kg": float(r["kg"]),
                        "mad": float(r["value_mad"]) if r.get("value_mad") else None})
        except (KeyError, ValueError):
            continue
    return out


def read_captain(path):
    """Captain's own trip log. Columns (any order, extra columns ignored):
    depart, return, fishing_h, sets, fuel_l, catch_kg, area   (times as 2026-09-14 06:30)"""
    if not path or not pathlib.Path(path).exists():
        return []
    text = pathlib.Path(path).read_text(encoding="utf-8-sig")
    out = []
    for r in csv.DictReader(text.splitlines(), delimiter=";" if text.count(";") > text.count(",") else ","):
        r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
        try:
            c = {"depart": when(r["depart"]), "return": when(r["return"]) if r.get("return") else None}
        except (KeyError, ValueError):
            continue
        for k in ("fishing_h", "sets", "fuel_l", "catch_kg"):
            try:
                c[k] = float(r[k]) if r.get(k) else None
            except ValueError:
                c[k] = None
        c["area"] = r.get("area") or ""
        out.append(c)
    return out


def compare(trips, captain, fuel_tol=0.25):
    """Pair each declared trip with the trip in the track that left nearest in time, and list what does not agree."""
    used, rows = set(), []
    dep = lambda t: dt.datetime.fromisoformat(t["t0"])
    # pair declared and tracked trips closest-first, so one wrong entry cannot shift all the others
    pairs = sorted((abs((dep(t) - c["depart"]).total_seconds()), ci, t["n"])
                   for ci, c in enumerate(captain) for t in trips
                   if abs((dep(t) - c["depart"]).total_seconds()) <= MATCH_H * 3600)
    match, taken = {}, set()
    for _, ci, tn in pairs:
        if ci not in match and tn not in taken:
            match[ci] = tn
            taken.add(tn)
    by_n = {t["n"]: t for t in trips}
    for ci, c in sorted(enumerate(captain), key=lambda q: q[1]["depart"]):
        best = by_n.get(match.get(ci))
        if best is None:
            rows.append({"declared": c["depart"].strftime("%Y-%m-%d %H:%M"), "trip": None, "level": "red", "c": None,
                         "flags": ["رحلة مصرّح بها لا أثر لها في مسار السفينة"]})
            continue
        used.add(best["n"])
        t0, t1 = dt.datetime.fromisoformat(best["t0"]), dt.datetime.fromisoformat(best["t1"])
        sea = (t1 - t0).total_seconds() / 3600
        r = {"declared": c["depart"].strftime("%Y-%m-%d %H:%M"), "trip": best["n"], "flags": [], "level": "ok",
             "c": {k: c[k] for k in ("fishing_h", "sets", "fuel_l", "catch_kg")} | {"area": c["area"],
                   "sea_h": round((c["return"] - c["depart"]).total_seconds() / 3600, 1) if c["return"] else None},
             "t": {"fishing_h": round(best["hours"]["fishing"], 1), "sets": len(best["sets"]), "fuel_l": best["litres"],
                   "sea_h": round(sea, 1), "gap_h": round(sum(g["h"] for g in best["gaps"]), 1)}}
        dep = (c["depart"] - t0).total_seconds() / 3600
        if abs(dep) > 2:
            r["flags"].append(f"وقت الخروج يختلف بـ {abs(dep):.1f} ساعة")
        if c["return"]:
            ret = (c["return"] - t1).total_seconds() / 3600
            if abs(ret) > 2:
                r["flags"].append(f"وقت العودة يختلف بـ {abs(ret):.1f} ساعة")
        if c["fishing_h"] is not None and abs(c["fishing_h"] - r["t"]["fishing_h"]) > max(2, 0.3 * r["t"]["fishing_h"]):
            r["flags"].append(f"ساعات الصيد: صرّح بـ {c['fishing_h']:g} والمسار يُظهر {r['t']['fishing_h']:g}")
        if c["sets"] is not None and c["sets"] != r["t"]["sets"]:
            r["flags"].append(f"عدد الرميات: صرّح بـ {c['sets']:g} والمسار يُظهر {r['t']['sets']}")
        if c["fuel_l"] and best["litres"]:
            ratio = c["fuel_l"] / best["litres"]
            r["t"]["fuel_ratio"] = round(ratio, 2)
            if ratio > 1 + fuel_tol:
                r["flags"].append(f"الوقود المصرّح به أكثر من المتوقع بـ {round(100 * (ratio - 1))}%")
            elif ratio < 1 - fuel_tol:
                r["flags"].append(f"الوقود المصرّح به أقل من المتوقع بـ {round(100 * (1 - ratio))}%")
        if c["catch_kg"] and not best["sets"]:
            r["flags"].append("صيد مصرّح به بينما المسار لا يُظهر أي رمية")
        if r["t"]["gap_h"] >= 2:
            r["flags"].append(f"انقطاع إشارة في البحر مجموعه {r['t']['gap_h']:g} ساعة")
        for s in best["stops"]:
            if s["port"]:
                r["flags"].append(f"توقف {s['h']:g} ساعة قرب {s['port']} ({s['t']})")
        heavy = ("وقت الخروج", "وقت العودة", "الوقود المصرّح به أكثر", "توقف", "صيد مصرّح", "انقطاع")
        r["level"] = ("red" if any(f.startswith(heavy) for f in r["flags"]) else "amber" if r["flags"] else "ok")
        rows.append(r)
    for t in trips:
        if t["n"] not in taken:
            rows.append({"declared": None, "trip": t["n"], "level": "red", "c": None,
                         "flags": [f"رحلة في المسار ({t['start']}) غير مصرّح بها"]})
    return rows


def grids_for(trips, zdir):
    """Madar score grid per trip day, small enough to embed (score only)."""
    out = {}
    for t in trips:
        z = zones_of(t["day"], zdir)
        if z and z.get("grid"):
            g = z["grid"]
            out[t["day"]] = {k: g[k] for k in ("n", "m", "lat0", "lon0", "cell", "score")}
    return out


def render(name, meta, trips, sm, fuel, price, grids, demo, cmp=None, fuel_tol=0.25):
    tpl = (ROOT / "tools" / "vessel_report.html").read_text(encoding="utf-8")
    data = {"name": name, "meta": meta, "trips": trips, "summary": sm, "fuel": fuel, "price": price,
            "port": PORT, "grids": grids, "cmp": cmp, "fuel_tol": fuel_tol, "demo": demo, "states": STATES, "ar": AR,
            "made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    return (tpl.replace("__TITLE__", html.escape(f"{name} · تقرير الرحلات"))
               .replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))


# ---------------------------------------------------------------- synthetic test track
BAY_EXIT = [(23.64, -15.90), (23.56, -15.97), (23.50, -16.10), (23.55, -16.35)]   # demo only


def demo_track(days=60, seed=7):
    """Two months of a 34 m pelagic boat out of Dakhla: trips of 1-3 days, a transit, some searching, a few sets."""
    rnd = random.Random(seed)
    zones = json.loads((ROOT / "docs" / "radar.json").read_text())["zones"]
    t = dt.datetime(2026, 9, 10, 4, 0, tzinfo=dt.timezone.utc)
    end = t + dt.timedelta(days=days)
    pts, trip_no = [], 0

    def sail(a, b, kn, step_min=2, wander=0.0):
        nonlocal t
        d = nm(a, b)
        steps = max(1, int(d / kn * 60 / step_min))
        for k in range(1, steps + 1):
            f = k / steps
            la = a[0] + (b[0] - a[0]) * f + rnd.gauss(0, wander)
            lo = a[1] + (b[1] - a[1]) * f + rnd.gauss(0, wander)
            t += dt.timedelta(minutes=step_min)
            pts.append((t, la, lo))
        return b

    while t < end:
        here = (PORT["lat"], PORT["lon"])
        for _ in range(rnd.randint(8, 30)):          # hours in port, logged every 20 min
            t += dt.timedelta(minutes=20)
            pts.append((t, here[0] + rnd.gauss(0, 0.001), here[1] + rnd.gauss(0, 0.001)))
        z = rnd.choice(zones[:4])
        good = rnd.random() < 0.55                    # half of the trips go where Madar pointed
        aim = ((z["lat"] + rnd.gauss(0, 0.06), z["lon"] + rnd.gauss(0, 0.06)) if good else
               (here[0] + rnd.uniform(-1.0, 1.2), here[1] - rnd.uniform(0.9, 2.0)))
        for w in BAY_EXIT:                            # leave the bay by its southern mouth
            here = sail(here, w, 7.5)
        here = sail(here, aim, rnd.uniform(9, 10.5))
        trip_no += 1
        if trip_no in (9, 22):                        # planted: AIS switched off for 4 h at sea
            t += dt.timedelta(hours=4)
        for _ in range(rnd.randint(1, 3)):            # search, then a set
            for _ in range(rnd.randint(2, 6) if good else rnd.randint(6, 14)):
                nxt = (here[0] + rnd.uniform(-0.08, 0.08), here[1] + rnd.uniform(-0.08, 0.08))
                here = sail(here, nxt, rnd.uniform(4, 6.5), wander=0.002)
            for _ in range(rnd.randint(40, 90)):      # set: drifting under 1.5 kn for 1.5-3 h
                t += dt.timedelta(minutes=2)
                here = (here[0] + rnd.gauss(0, 0.0006), here[1] + rnd.gauss(0, 0.0006))
                pts.append((t, *here))
        if trip_no == 16:                             # planted: 3 h stop off Boujdour on the way back
            here = sail(here, (26.10, -14.53), 10)
            for _ in range(90):
                t += dt.timedelta(minutes=2)
                pts.append((t, here[0] + rnd.gauss(0, 0.0003), here[1] + rnd.gauss(0, 0.0003)))
        here = sail(here, BAY_EXIT[-1], rnd.uniform(9, 10.5))
        for w in BAY_EXIT[-2::-1] + [(PORT["lat"], PORT["lon"])]:
            here = sail(here, w, 7.5)
        if rnd.random() < 0.15:                       # AIS hole
            t += dt.timedelta(hours=5)
    return [p for p in pts if p[0] < end]


def demo_captain(trips, seed=3):
    rnd = random.Random(seed)
    out = []
    for k, t in enumerate(trips):
        t0, t1 = dt.datetime.fromisoformat(t["t0"]), dt.datetime.fromisoformat(t["t1"])
        c = {"depart": t0 + dt.timedelta(minutes=rnd.randint(-40, 40)), "return": t1 + dt.timedelta(minutes=rnd.randint(-40, 40)),
             "fishing_h": round(t["hours"]["fishing"] * rnd.uniform(0.85, 1.15), 1), "sets": len(t["sets"]),
             "fuel_l": round(t["litres"] * rnd.uniform(0.92, 1.12)), "catch_kg": rnd.randint(8, 40) * 1000, "area": ""}
        if k % 7 == 3:
            c["fuel_l"] = round(t["litres"] * 1.45)          # planted: fuel over-declared
        if k % 9 == 5:
            c["sets"] = len(t["sets"]) + 2                  # planted: more sets declared
        if k % 11 == 7:
            c["return"] = t1 + dt.timedelta(hours=5)        # planted: return time off
        if k != 12:                                         # planted: one trip left out of the log
            out.append(c)
    out.append({"depart": dt.datetime.fromisoformat(trips[5]["t0"]) + dt.timedelta(days=1, hours=3), "return": None,
                "fishing_h": 6, "sets": 2, "fuel_l": 4000, "catch_kg": 20000, "area": ""})   # planted: a trip that never happened
    return out


def demo_zones(zdir, days=60):
    """Use today's zones for every demo day (the real report uses the zones rebuilt for each day)."""
    r = json.loads((ROOT / "docs" / "radar.json").read_text())
    zdir.mkdir(parents=True, exist_ok=True)
    d = dt.date(2026, 9, 10)
    for k in range(days + 1):
        day = (d + dt.timedelta(days=k)).isoformat()
        (zdir / f"{day}.json").write_text(json.dumps({"day": day, "zones": r["zones"], "grid": r["grid"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track", nargs="?", help="GPX, KML, NMEA or CSV position log")
    ap.add_argument("--name", default="VESSEL")
    ap.add_argument("--config", help="JSON with fuel_lph, diesel_mad_l, catch_csv")
    ap.add_argument("--zones", default=str(ROOT / "docs" / "archive" / "zones"))
    ap.add_argument("--out", help="output HTML (default private/vessel_report_<name>.html)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.config).read_text()) if a.config else {}
    zdir = pathlib.Path(a.zones)
    if a.demo:
        pts, a.name = demo_track(), "DEMO"
        zdir = ROOT / "private" / "demo_zones"
        demo_zones(zdir)
        cfg.setdefault("catch_csv", None)
    elif not a.track:
        sys.exit("give a track file, or --demo")
    else:
        pts = read_track(a.track)
    rows = label(pts)
    trips, fuel, price = analyse(rows, cfg, zdir, read_catch(cfg.get("catch_csv")))
    if not trips:
        sys.exit("no trip found: does the log start and end at Dakhla port?")
    sm = summary(trips)
    sm["search_mad"] = round(sm["hours"]["search"] * fuel["search"] * price)
    cap = demo_captain(trips) if a.demo else read_captain(cfg.get("captain_csv"))
    cmp = compare(trips, cap, cfg.get("fuel_tolerance", 0.25)) if cap else None
    meta = {"points": len(pts), "first": pts[0][0].strftime("%Y-%m-%d"), "last": pts[-1][0].strftime("%Y-%m-%d"),
            "source": pathlib.Path(a.track).name if a.track else "synthetic"}
    out = pathlib.Path(a.out or ROOT / "private" / f"vessel_report_{a.name.lower()}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(a.name, meta, trips, sm, fuel, price, grids_for(trips, zdir), a.demo, cmp,
                          cfg.get("fuel_tolerance", 0.25)), encoding="utf-8")
    print(f"{sm['trips']} trips · {sm['sea_h']} h at sea · search {sm['search_share']}% · "
          f"{sm['sets']} sets · {sm['near10']}/{sm['compared']} sets within 10 nm of a Madar zone -> {out}")


if __name__ == "__main__":
    main()
