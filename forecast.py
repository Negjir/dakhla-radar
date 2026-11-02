"""
Dakhla Fish Radar - FORECAST layer (D+1, D+2, D+3).

Idea: on the Dakhla coast the chain is
  north wind strengthens -> cold water upwells (1-3 days later) -> fronts form
  -> plankton blooms -> pelagic fish gather.
So we look AHEAD instead of at yesterday's satellite picture:

  1. Ocean model forecast of sea temperature for the next 3 days (Copernicus Marine).
  2. Wind forecast along the coast (Open-Meteo) -> Coastal Upwelling Index (CUI).
  3. Score per day = forecast fronts + chlorophyll (latest satellite) + upwelling boost
     near the coast where recent/forecast winds push upwelling.
  4. Output docs/forecast.png, docs/forecast_message.txt, docs/forecast.json, docs/forecast.html.

Usage:
  python forecast.py            # real data (needs the same Copernicus secrets)
  python forecast.py --demo     # synthetic data, to test the setup
"""
import argparse
import datetime as dt
import html
import json
import math
import urllib.request

import numpy as np

import radar  # reuse config, scoring, zones, formatting

CFG = radar.CFG
OUT = radar.OUT
DAYS_AR = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]

MODEL_CANDIDATES = CFG.get("model_sst_datasets") or [
    ["cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", "thetao"],
]

# Coastal points (lat, lon offshore ~20 km, direction the coast runs toward the south, degrees)
COAST_POINTS = CFG.get("coast_points") or [
    [26.0, -14.95, 212, "Boujdour"],
    [25.2, -15.30, 210, ""],
    [24.3, -15.80, 220, ""],
    [23.5, -16.30, 215, "Dakhla"],
    [22.5, -16.95, 205, ""],
    [21.5, -17.35, 190, "Cap Barbas S"],
]


# ----------------------------------------------------------------- wind / upwelling
def coastal_upwelling_index(speed_ms, dir_from_deg, coast_dir_deg, lat):
    """Offshore Ekman transport per 100 m of coast (m3/s). Positive = upwelling."""
    rho_a, cd, rho_w = 1.22, 1.3e-3, 1025.0
    f = 2 * 7.2921e-5 * math.sin(math.radians(lat))
    to_dir = (dir_from_deg + 180) % 360
    along = speed_ms * math.cos(math.radians(to_dir - coast_dir_deg))
    tau = rho_a * cd * speed_ms * along
    return tau / (rho_w * f) * 100


def wind_forecast(today):
    lats = ",".join(str(p[0]) for p in COAST_POINTS)
    lons = ",".join(str(p[1]) for p in COAST_POINTS)
    url = ("https://api.open-meteo.com/v1/forecast?latitude=" + lats + "&longitude=" + lons
           + "&hourly=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms&past_days=3&forecast_days=4&timezone=UTC")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode())
    if isinstance(data, dict):
        data = [data]
    out = []  # per point: {date: mean CUI}
    for p, d in zip(COAST_POINTS, data):
        h = d["hourly"]
        per_day = {}
        for t, s, w in zip(h["time"], h["wind_speed_10m"], h["wind_direction_10m"]):
            if s is None or w is None:
                continue
            per_day.setdefault(t[:10], []).append(coastal_upwelling_index(s, w, p[2], p[0]))
        out.append({k: float(np.mean(v)) for k, v in per_day.items()})
    return out


def wind_demo(today):
    rng = np.random.default_rng(int(today.strftime("%Y%m%d")))
    out = []
    pulse = rng.uniform(-1, 3)  # day of the strongest wind event
    for i, p in enumerate(COAST_POINTS):
        days = {}
        for k in range(-3, 4):
            day = str(today + dt.timedelta(days=k))
            days[day] = float(70 + 190 * math.exp(-((k - pulse) ** 2) / 1.5) * (1 - 0.08 * i) + rng.normal(0, 15))
        out.append(days)
    return out


def lagged_upwelling(cui, day):
    """Upwelling that should be visible on `day`: weighted winds of the 1-3 days before (and the day itself)."""
    weights = {0: 0.2, 1: 0.35, 2: 0.3, 3: 0.15}
    vals = []
    for pt in cui:
        v = 0.0
        for lag, w in weights.items():
            v += w * pt.get(str(day - dt.timedelta(days=lag)), 0.0)
        vals.append(v)
    return np.array(vals)


# ----------------------------------------------------------------- ocean model
def model_sst(today, box, demo):
    days = [today + dt.timedelta(days=k) for k in (1, 2, 3)]
    if demo:
        frames = []
        for d in days:
            lats, lons, sst, _, _ = radar.load_demo(d, box)
            frames.append(sst)
        return lats, lons, frames, days, "DEMO"
    import copernicusmarine as cm

    err = []
    for dataset_id, var in MODEL_CANDIDATES:
        try:
            ds = cm.open_dataset(
                dataset_id=dataset_id, variables=[var],
                minimum_longitude=box["lon_min"], maximum_longitude=box["lon_max"],
                minimum_latitude=box["lat_min"], maximum_latitude=box["lat_max"],
                minimum_depth=0, maximum_depth=1,
                start_datetime=f"{days[0]}T00:00:00", end_datetime=f"{days[-1]}T23:59:59",
            )
            da = ds[var]
            if "depth" in da.dims:
                da = da.isel(depth=0)
            la = "latitude" if "latitude" in da.coords else "lat"
            lo = "longitude" if "longitude" in da.coords else "lon"
            da = da.sortby(la, ascending=False)
            frames = [da.sel(time=str(d), method="nearest").values.astype("float32") for d in days]
            print(f"[ok] model SST: {dataset_id}")
            return da[la].values, da[lo].values, frames, days, dataset_id
        except Exception as e:  # noqa: BLE001
            err.append(f"{dataset_id}: {e}")
            print(f"[skip] model SST: {dataset_id} -> {e}")
    raise SystemExit("Could not download the ocean model forecast:\n  " + "\n  ".join(err))


def latest_chl(today, box, lats, lons, demo):
    if demo:
        _, dl, _, chl, _ = radar.load_demo(today, box)
        return chl if chl.shape == (len(lats), len(lons)) else None
    try:
        da, _ = radar.fetch(radar.CHL_CANDIDATES, today, box, "CHL")
        la = "latitude" if "latitude" in da.coords else "lat"
        lo = "longitude" if "longitude" in da.coords else "lon"
        return da.interp({la: lats, lo: lons}).values.astype("float32")
    except SystemExit as e:
        print("[warn] no chlorophyll, forecast uses fronts + wind only:", e)
        return None


def upwelling_boost(lats, lons, ocean, up_vals):
    from scipy.ndimage import distance_transform_edt

    step_km = abs(lats[1] - lats[0]) * 111
    dist_km = distance_transform_edt(ocean) * step_km
    pts_lat = np.array([p[0] for p in COAST_POINTS])
    order = np.argsort(pts_lat)
    ref = CFG.get("cui_strong", 200.0)
    along = np.interp(lats, pts_lat[order], np.clip(up_vals[order] / ref, 0, 1.3))
    boost = along[:, None] * np.exp(-dist_km / CFG.get("upwelling_reach_km", 90))
    boost[~ocean] = 0
    return boost


# ----------------------------------------------------------------- outputs
def draw(lats, lons, frames, scores, zones_by_day, days, cui, today, path, demo):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "s", ["#071a2c", "#0f3a55", "#136b6f", "#d39a3a", "#ffb547", "#ff6b4a"])
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    fig = plt.figure(figsize=(15, 8), dpi=110, facecolor="#06121d")
    gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1])
    port = CFG["port"]
    for k in range(3):
        ax = fig.add_subplot(gs[0, k])
        ocean = np.isfinite(frames[k])
        ax.set_facecolor("#2a2418")
        ax.imshow(np.ma.masked_where(~ocean, scores[k]), extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=1,
                  interpolation="bilinear", aspect=1 / math.cos(math.radians(23.7)))
        ax.contour(lons, lats, ocean.astype(float), levels=[0.5], colors="#e6d2aa", linewidths=0.8)
        ax.plot(port["lon"], port["lat"], "o", color="#4fe3d6", ms=6, mec="#06121d")
        for i, z in enumerate(zones_by_day[k]):
            ax.plot(z["lon"], z["lat"], "o", ms=15 if i == 0 else 11, mfc="#06121d", mec="white", mew=2)
            ax.text(z["lon"], z["lat"], z["id"], color="white", ha="center", va="center", fontsize=8, weight="bold")
        ax.set_title(f"D+{k + 1} · {days[k]:%a %d %b}", color="#e6eef3", fontsize=11)
        ax.tick_params(colors="#8aa3b4", labelsize=7)
        for s in ax.spines.values():
            s.set_color("#1c3448")
    # upwelling index along the coast
    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor("#0b1c2b")
    lat_pts = [p[0] for p in COAST_POINTS]
    colors = ["#5b7387", "#4fe3d6", "#ffb547", "#ff6b4a"]
    for k, c in zip(range(0, 4), colors):
        d = today + dt.timedelta(days=k)
        vals = [pt.get(str(d), np.nan) for pt in cui]
        ax.plot(vals, lat_pts, "-o", color=c, lw=2 if k else 1, ms=4, label="today" if k == 0 else f"D+{k}")
    ax.axvline(CFG.get("cui_strong", 200.0), color="#ffb547", ls="--", lw=1)
    ax.set_title("Upwelling index (wind)", color="#e6eef3", fontsize=11)
    ax.set_xlabel("m³/s per 100 m coast", color="#8aa3b4", fontsize=8)
    ax.tick_params(colors="#8aa3b4", labelsize=7)
    ax.legend(fontsize=7, facecolor="#06121d", labelcolor="#e6eef3", edgecolor="#1c3448")
    for s in ax.spines.values():
        s.set_color("#1c3448")
    # timeline of mean upwelling
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor("#0b1c2b")
    ds = [today + dt.timedelta(days=k) for k in range(-3, 4)]
    mean = [np.nanmean([pt.get(str(d), np.nan) for pt in cui]) for d in ds]
    ax.bar([f"{d:%a %d}" for d in ds], mean, color=["#5b7387" if d <= today else "#ffb547" for d in ds])
    ax.axhline(CFG.get("cui_strong", 200.0), color="#ffb547", ls="--", lw=1)
    ax.set_title("Coast-average upwelling: past 3 days (grey) and forecast (amber)", color="#e6eef3", fontsize=10)
    ax.tick_params(colors="#8aa3b4", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#1c3448")
    src = "DEMO DATA" if demo else "Copernicus Marine model + Open-Meteo wind"
    fig.suptitle(f"Dakhla Fish Radar · FORECAST from {today} · {src}", color="#e6eef3", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def message(days, zones_by_day, cui, today):
    best_day, best_val = None, -1e9
    for k in range(1, 4):
        d = today + dt.timedelta(days=k)
        v = np.nanmean([pt.get(str(d), np.nan) for pt in cui])
        if v > best_val:
            best_day, best_val = d, v
    strong = best_val >= CFG.get("cui_strong", 200.0)
    lines = [f"🔭 توقعات رادار الصيد · مساء {today}"]
    if strong:
        lines.append(f"رياح شمالية قوية متوقعة يوم {DAYS_AR[best_day.weekday()]}: صعود مياه قوي وجبهات جديدة قرب الساحل بعدها بيوم أو يومين.")
    else:
        lines.append(f"لا حدث رياح قوي في الأيام الثلاثة القادمة. أقوى يوم: {DAYS_AR[best_day.weekday()]}.")
    lines.append("")
    for k, (d, zs) in enumerate(zip(days, zones_by_day)):
        if not zs:
            lines.append(f"{DAYS_AR[d.weekday()]} (+{k + 1}): لا جبهة قوية متوقعة.")
            continue
        z = zs[0]
        away = radar.ltr("%s nm · %s°" % (z["distance_nm"], z["bearing_deg"]))
        lines.append(f"{DAYS_AR[d.weekday()]} (+{k + 1}): {radar.pos(z['lat'], z['lon'])} · {away} · ثقة {z['confidence']}%")
    lines += ["", "توقعات تقديرية، تتأكد صباحاً بخريطة الأقمار وبالسونار."]
    return "\n".join(lines)


def page(msg, days, today, demo):
    warn = "<p class='w'>بيانات تجريبية (demo).</p>" if demo else ""
    doc = f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>توقعات رادار الصيد</title>
<style>body{{margin:0;background:#06121d;color:#e6eef3;font-family:system-ui,sans-serif;padding:16px}}
main{{max-width:1100px;margin:auto}}img{{width:100%;border-radius:12px;border:1px solid #1c3448}}
pre{{white-space:pre-wrap;background:#0a2a22;border:1px solid #1f5b49;border-radius:12px;padding:12px;font:15px/1.7 system-ui}}
button{{background:#4fe3d6;border:0;border-radius:999px;padding:12px 18px;font-weight:700;font-size:15px}}
a{{color:#4fe3d6}}.w{{color:#ffb547}}.s{{color:#8aa3b4;font-size:13px}}</style></head><body><main>
<h1>توقعات 3 أيام</h1><p class="s">من {today} · <a href="index.html">خريطة اليوم</a></p>{warn}
<pre id="m">{html.escape(msg)}</pre>
<button onclick="navigator.clipboard.writeText(document.getElementById('m').innerText).then(()=>this.textContent='تم النسخ')">نسخ الرسالة</button>
<p><img src="forecast.png?v={today}" alt="توقعات مناطق الصيد"></p>
<p class="s">مؤشر الصعود يُحسب من الرياح الموازية للساحل (نقل إكمان). الخطوط الصفراء المتقطعة = عتبة "صعود قوي" في config.json.
بيانات الرياح: Open-Meteo (CC BY 4.0). نموذج المحيط: Copernicus Marine.</p>
</main></body></html>"""
    (OUT / "forecast.html").write_text(doc, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
    box = CFG["box"]

    cui = wind_demo(today) if a.demo else wind_forecast(today)
    lats, lons, frames, days, src = model_sst(today, box, a.demo)
    chl = latest_chl(today, box, lats, lons, a.demo)
    reports = radar.read_reports(today)

    scores, zones_by_day = [], []
    for k, (d, sst) in enumerate(zip(days, frames)):
        ocean = np.isfinite(sst)
        c = chl if chl is not None else np.full_like(sst, 10 ** CFG.get("chl_best_log10", 0.25))
        base, _ = radar.score_field(lats, lons, sst, c, reports)
        boost = upwelling_boost(lats, lons, ocean, lagged_upwelling(cui, d))
        s = base * (0.7 + 0.6 * boost)
        s[~ocean] = 0
        s = s / (s.max() or 1)
        conf_cap = [0.9, 0.8, 0.7][k]  # forecasts get less certain with time
        scores.append(s)
        zs = radar.pick_zones(lats, lons, s)
        for z in zs:
            z["confidence"] = round(z["confidence"] * conf_cap)
        zones_by_day.append(zs)

    (OUT / "archive").mkdir(parents=True, exist_ok=True)
    draw(lats, lons, frames, scores, zones_by_day, days, cui, today, OUT / "forecast.png", a.demo)
    msg = message(days, zones_by_day, cui, today)
    (OUT / "forecast_message.txt").write_text(msg, encoding="utf-8")
    (OUT / "forecast.json").write_text(json.dumps({
        "issued": str(today), "demo": a.demo, "model": src,
        "days": [{"date": str(d), "zones": zs} for d, zs in zip(days, zones_by_day)],
        "upwelling_index": [{"lat": p[0], "lon": p[1], "daily": c} for p, c in zip(COAST_POINTS, cui)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "archive").joinpath("forecast_history.csv").open("a", encoding="utf-8") as f:
        for k, (d, zs) in enumerate(zip(days, zones_by_day)):
            for z in zs:
                f.write(f"{today},{d},D+{k + 1},{z['id']},{z['lat']},{z['lon']},{z['confidence']}\n")
    page(msg, days, today, a.demo)
    print(msg)


if __name__ == "__main__":
    main()
