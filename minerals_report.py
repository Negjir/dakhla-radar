"""
MADAR Minerals - study report (PDF, French) for each site in docs/minerals/minerals.json.

    python minerals_report.py                      # all sites, generic "document de démonstration"
    python minerals_report.py --site bouazzer --client "Société X" --ref "MM-2026-001"

Writes docs/minerals/<id>/MADAR-<id>-rapport.pdf and adds the link to the console page.
"""
import argparse
import json
import math
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "minerals"
FONTS = ROOT / "fonts"
pdfmetrics.registerFont(TTFont("DV", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FONTS / "DejaVuSansMono.ttf")))

COPPER = colors.HexColor("#c8641e")
GOLD = colors.HexColor("#e09a2c")
INK = colors.HexColor("#1d1712")
MUTE = colors.HexColor("#6b5d50")
LINE = colors.HexColor("#e6dccf")
PAPER = colors.HexColor("#faf6f0")
DARK = colors.HexColor("#140d07")

S = {
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=17, leading=22, textColor=INK, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=12.5, leading=16, textColor=COPPER, spaceBefore=10, spaceAfter=5),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9.6, leading=14, textColor=INK, alignment=TA_LEFT, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8, leading=11, textColor=MUTE),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.3, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=8.3, leading=10.5, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="DVM", fontSize=8, leading=10.5, textColor=INK),
    "kpi": ParagraphStyle("kpi", fontName="DVB", fontSize=18, leading=22, textColor=COPPER),
    "kpil": ParagraphStyle("kpil", fontName="DV", fontSize=8, leading=10, textColor=MUTE),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=9.6, leading=14, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=3),
}

PROFILE_FR = {"alteration": "altération hydrothermale (argiles + oxydes de fer)",
              "iron": "minerai de fer (fer ferrique + ferreux)",
              "carbonatite": "carbonatite (oxydes de fer + indice terres rares expérimental)"}
METAL_FR = {"bouazzer": "Cobalt, nickel, or", "imiter": "Argent", "kediet": "Fer",
            "twihinate": "Terres rares, niobium, fer", "lafhouda": "Niobium, tantale, terres rares, fer"}


def country_fr(c):
    return c.replace("Morocco", "Maroc").replace("Mauritania", "Mauritanie")


def dms(v, pos, neg):
    a = abs(v)
    d = int(a)
    m = int((a - d) * 60)
    s = round(((a - d) * 60 - m) * 60)
    if s == 60:
        s, m = 0, m + 1
    if m == 60:
        m, d = 0, d + 1
    return "%d°%02d'%02d\"%s" % (d, m, s, pos if v >= 0 else neg)


# ----------------------------------------------------------------------------- maps
def _font(size, bold=False):
    return ImageFont.truetype(str(FONTS / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def map_image(site, layer, path, width=1600, known=True):
    """Map figure: true colour (dimmed) + layer overlay, targets, known site, scale bar, north arrow."""
    folder = OUT / site["id"]
    base = Image.open(folder / "truecolor.jpg").convert("RGB")
    W = width
    H = int(base.height * W / base.width)
    base = base.resize((W, H), Image.LANCZOS)
    if layer == "ratios":
        img = Image.open(folder / "ratios.jpg").convert("RGB").resize((W, H), Image.LANCZOS)
    elif layer == "truecolor":
        img = base
    else:
        img = Image.eval(base, lambda v: int(v * 0.62)).convert("RGBA")
        ov = Image.open(folder / (layer + ".png")).convert("RGBA").resize((W, H), Image.LANCZOS)
        img.alpha_composite(ov)
        img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    (la0, lo0), (la1, lo1) = site["bounds"]
    xy = lambda la, lo: ((lo - lo0) / (lo1 - lo0) * W, (la1 - la) / (la1 - la0) * H)
    f = _font(max(14, W // 70), True)
    if known:
        for k in site.get("blind_test", []):
            x, y = xy(k["lat"], k["lon"])
            r = 2.0 / ((la1 - la0) * 111.32) * H
            d.ellipse([x - r, y - r, x + r, y + r], outline=(80, 210, 255), width=3)
            d.polygon([(x, y - 11), (x + 11, y), (x, y + 11), (x - 11, y)], fill=(80, 210, 255), outline=(10, 30, 40))
    for t in reversed(site["targets"]):
        x, y = xy(t["lat"], t["lon"])
        r = 15 if t["rank"] <= 3 else 12
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 138, 61) if t["rank"] <= 3 else (20, 13, 7),
                  outline=(255, 243, 214), width=3)
        d.text((x + r + 4, y), t["label"], font=f, fill=(255, 240, 210), anchor="lm", stroke_width=3, stroke_fill=(20, 13, 7))
    # scale bar
    km_px = W / ((lo1 - lo0) * 111.32 * math.cos(math.radians((la0 + la1) / 2)))
    sk = next(v for v in (1, 2, 5, 10, 20, 50) if v * km_px > W * 0.12)
    x0, y0 = 30, H - 40
    d.rectangle([x0 - 10, y0 - 34, x0 + sk * km_px + 14, y0 + 14], fill=(20, 13, 7))
    d.rectangle([x0, y0 - 4, x0 + sk * km_px, y0 + 4], fill=(255, 243, 214))
    d.text((x0, y0 - 10), "%d km" % sk, font=f, fill=(255, 243, 214), anchor="ls")
    # north arrow
    nx, ny = W - 50, 60
    d.polygon([(nx, ny - 32), (nx + 14, ny + 10), (nx, ny), (nx - 14, ny + 10)], fill=(255, 243, 214), outline=(20, 13, 7))
    d.text((nx, ny + 30), "N", font=f, fill=(255, 243, 214), anchor="mm", stroke_width=3, stroke_fill=(20, 13, 7))
    img.save(path, quality=90)
    return path, W, H


def legend_bar(path, kind):
    W, H = 900, 60
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = _font(20)
    if kind == "heat":
        stops = [(70, 55, 45), (180, 60, 20), (255, 140, 0), (255, 210, 80), (255, 255, 220)]
        for i in range(W - 200):
            x = i / (W - 201) * (len(stops) - 1)
            k = min(int(x), len(stops) - 2)
            fr = x - k
            c = tuple(int(stops[k][j] * (1 - fr) + stops[k + 1][j] * fr) for j in range(3))
            d.line([(100 + i, 8), (100 + i, 30)], fill=c)
        d.text((100, 48), "faible", font=f, fill=(90, 80, 70), anchor="lm")
        d.text((W - 100, 48), "fort", font=f, fill=(90, 80, 70), anchor="rm")
    else:
        img = Image.new("RGB", (1300, H), (255, 255, 255))
        d = ImageDraw.Draw(img)
        items = [((235, 70, 70), "Rouge = argiles (B11/B12)"), ((70, 220, 110), "Vert = oxydes de fer (B4/B2)"),
                 ((80, 140, 255), "Bleu = fer ferreux")]
        x = 10
        for c, t in items:
            d.rectangle([x, 18, x + 24, 42], fill=c)
            d.text((x + 32, 30), t, font=f, fill=(40, 30, 20), anchor="lm")
            x += 440
    img.save(path)
    return path


# ----------------------------------------------------------------------------- pdf
def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page == 1:
        canvas.setFillColor(DARK)
        canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(COPPER)
    canvas.rect(0, 0, w, 3 * mm, fill=1, stroke=0)
    if doc.page > 1:
        canvas.setFont("DVB", 8)
        canvas.setFillColor(COPPER)
        canvas.drawString(18 * mm, h - 12 * mm, "MADAR · MINERALS")
        canvas.setFont("DV", 8)
        canvas.setFillColor(MUTE)
        canvas.drawRightString(w - 18 * mm, h - 12 * mm, doc._madar_title)
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, h - 14 * mm, w - 18 * mm, h - 14 * mm)
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUTE)
    canvas.drawString(18 * mm, 8 * mm, "Confidentiel · " + doc._madar_ref)
    canvas.drawRightString(w - 18 * mm, 8 * mm, "Page %d" % doc.page)
    canvas.restoreState()


def kpis(rows):
    data = [[Paragraph(v, S["kpi"]) for v, _ in rows], [Paragraph(l, S["kpil"]) for _, l in rows]]
    t = Table(data, colWidths=[174 * mm / len(rows)] * len(rows))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PAPER), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                           ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def fig(path, w_mm, W, H, caption, max_h=118):
    h_mm = w_mm * H / W
    if h_mm > max_h:
        w_mm, h_mm = max_h * W / H, max_h
    return KeepTogether([RLImage(str(path), width=w_mm * mm, height=h_mm * mm), Spacer(1, 2), Paragraph(caption, S["small"]), Spacer(1, 8)])


def build(site, data, test, client, ref, tmp):
    sid = site["id"]
    out = OUT / sid / ("MADAR-%s-rapport.pdf" % sid)
    title = "Étude de ciblage satellite · %s" % site["name"]
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=16 * mm,
                            title=title, author="MADAR", subject="MADAR Minerals")
    doc._madar_title, doc._madar_ref = title, ref
    tg = site["targets"]
    bt = site.get("blind_test", [])
    st = []

    # ---------------- cover
    cover_title = ParagraphStyle("ct", fontName="DVB", fontSize=26, leading=31, textColor=colors.white)
    cover_sub = ParagraphStyle("cs", fontName="DV", fontSize=11.5, leading=16, textColor=colors.HexColor("#f2d7b5"))
    cover_tag = ParagraphStyle("cg", fontName="DVB", fontSize=9, leading=12, textColor=GOLD)
    st += [Spacer(1, 8 * mm), Paragraph("MADAR · MINERALS", cover_tag), Spacer(1, 4 * mm),
           Paragraph("Étude de ciblage<br/>par satellite", cover_title), Spacer(1, 4 * mm),
           Paragraph("%s · %s<br/>%s" % (site["name"], country_fr(site["country"]), METAL_FR.get(sid, site["metal"])), cover_sub),
           Spacer(1, 14 * mm)]
    mp, W, H = map_image(site, "score", tmp / (sid + "_cover.jpg"), known=False)
    st.append(RLImage(str(mp), width=174 * mm, height=min(174 * mm * H / W, 105 * mm)))
    st.append(Spacer(1, 6 * mm))
    meta_cell = ParagraphStyle("mc", parent=S["cell"], textColor=colors.HexColor("#f2e6d6"))
    meta_cellb = ParagraphStyle("mcb", parent=S["cellb"], textColor=GOLD)
    meta = [["Préparé pour", client], ["Référence", ref], ["Date", date.today().strftime("%d/%m/%Y")],
            ["Images", "ESA Copernicus Sentinel-2 L2A · " + ", ".join(sorted({s["date"] for s in site["scenes"]}))],
            ["Modèle", PROFILE_FR.get(site["profile"], site["profile"])]]
    t = Table([[Paragraph(a, meta_cellb), Paragraph(b, meta_cell)] for a, b in meta], colWidths=[34 * mm, 140 * mm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#4a3a2a")), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    st += [t, PageBreak()]

    # ---------------- 1. synthese
    st.append(Paragraph("1. Synthèse", S["h1"]))
    st.append(kpis([("%s km²" % format(site["area_km2"], ",").replace(",", " "), "zone analysée"),
                    ("%s km²" % site["anomaly_km2"], "zones prioritaires (top %s %%)" % data.get("top_percent", 1)),
                    ("×%s" % site["reduction"], "réduction de la zone de recherche"),
                    ("%d" % len(tg), "cibles classées")]))
    st.append(Spacer(1, 8))
    top3 = ", ".join("%s (%.1f ha)" % (x["label"], x["area_ha"]) for x in tg[:3])
    st.append(Paragraph(
        "MADAR a analysé l'ensemble de la zone à partir d'images satellite Sentinel-2 sans nuages, avec le modèle "
        "<b>%s</b>. Les pixels les plus anomaux (top %s %% de la zone) ont été regroupés en <b>%d cibles</b> classées par "
        "taille et intensité. Les cibles prioritaires pour une première visite de terrain sont <b>%s</b>."
        % (PROFILE_FR.get(site["profile"], site["profile"]), data.get("top_percent", 1), len(tg), top3), S["p"]))
    if bt:
        k = bt[0]
        verdict = ("une cible (%s) se trouve à <b>%s km</b> du site connu" % (k["nearest_target"], k["nearest_km"])) if k["hit"] else \
                  ("la cible la plus proche (%s) est à %s km du site connu, au-delà du seuil de %s km" % (k["nearest_target"], k["nearest_km"], data.get("hit_km", 2)))
        st.append(Paragraph(
            "<b>Contrôle à l'aveugle.</b> La position du site de référence « %s » n'a pas été utilisée par l'analyse. "
            "Après coup : %s ; le score MADAR autour du site dépasse <b>%s %%</b> de la zone." % (k["name"], verdict, k["percentile"]), S["p"]))
    if test:
        s2 = test["summary"]
        st.append(Paragraph(
            "<b>Validation de la méthode.</b> Sur un test à l'aveugle de %d gîtes connus au Maroc et en Mauritanie, MADAR a placé "
            "une cible à moins de %s km de <b>%d sites</b>, contre <b>%.1f</b> attendus par hasard (×%.1f, probabilité d'un "
            "résultat dû à la chance : %.1f %%)." % (s2["n"], test.get("hit_km", 2), s2["hits"], s2["expected_by_chance"], s2["lift"],
                                                      100 * s2["p_value"]), S["p"]))
    st.append(Paragraph("Recommandations", S["h2"]))
    for b in ["Visite de reconnaissance des cibles <b>%s</b> : cartographie, échantillonnage roche et sol, analyses." % ", ".join(x["label"] for x in tg[:3]),
              "Contrôle des cibles suivantes selon les résultats et l'accès.",
              "Vérifier sur l'image en couleurs réelles que chaque cible n'est pas une piste, un village, une ancienne exploitation ou une sebkha.",
              "Intégrer les fichiers SIG livrés (cibles, polygones, score) aux données géologiques et géophysiques disponibles."]:
        st.append(Paragraph(b, S["bul"], bulletText="•"))
    st.append(PageBreak())

    # ---------------- 2. cartes
    st.append(Paragraph("2. Cartes", S["h1"]))
    mp, W, H = map_image(site, "score", tmp / (sid + "_score.jpg"))
    lb = legend_bar(tmp / "leg_heat.png", "heat")
    st.append(fig(mp, 174, W, H, "Figure 1 · Score MADAR (zones claires = anomalies les plus fortes), cibles T1–T%d, et site de référence "
                  "(losange bleu, cercle de 2 km) utilisé uniquement pour le contrôle à l'aveugle." % len(tg)))
    st.append(RLImage(str(lb), width=90 * mm, height=6 * mm))
    st.append(Spacer(1, 8))
    mp, W, H = map_image(site, "ratios", tmp / (sid + "_ratios.jpg"))
    lr = legend_bar(tmp / "leg_rgb.png", "rgb")
    st.append(fig(mp, 174, W, H, "Figure 2 · Composite d'indices : rouge = argiles, vert = oxydes de fer, bleu = fer ferreux. "
                  "Les couleurs mélangées (jaune, magenta) signalent la présence simultanée de plusieurs signatures."))
    st.append(RLImage(str(lr), width=150 * mm, height=150 * mm * 60 / 1300))
    st.append(Spacer(1, 8))
    mp, W, H = map_image(site, "truecolor", tmp / (sid + "_tc.jpg"))
    st.append(fig(mp, 174, W, H, "Figure 3 · Image Sentinel-2 en couleurs réelles, pour vérifier la nature de chaque cible sur le terrain."))
    st.append(PageBreak())

    # ---------------- 3. cibles
    st.append(Paragraph("3. Cibles classées", S["h1"]))
    st.append(Paragraph("Coordonnées WGS84 du centre pondéré de chaque zone anomale. La force est relative à T1 ; ce n'est pas une "
                        "probabilité de présence de minerai.", S["small"]))
    st.append(Spacer(1, 6))
    head = ["Cible", "Latitude, longitude (DD)", "DMS", "Surface", "Force", "Priorité"]
    rows = [[Paragraph(h, S["cellb"]) for h in head]]
    for x in tg:
        pr = "Haute" if x["rank"] <= 3 else "Moyenne" if x["rank"] <= 6 else "Basse"
        rows.append([Paragraph(x["label"], S["cellb"]), Paragraph("%.5f, %.5f" % (x["lat"], x["lon"]), S["mono"]),
                     Paragraph("%s %s" % (dms(x["lat"], "N", "S"), dms(x["lon"], "E", "W")), S["mono"]),
                     Paragraph("%.1f ha" % x["area_ha"], S["cell"]), Paragraph("%d %%" % x["rel"], S["cell"]), Paragraph(pr, S["cell"])])
    t = Table(rows, colWidths=[14 * mm, 40 * mm, 52 * mm, 20 * mm, 16 * mm, 20 * mm], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PAPER), ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("BACKGROUND", (0, 1), (-1, 3), colors.HexColor("#fff3e6"))]))
    st.append(t)
    st.append(Paragraph("Livrables SIG", S["h2"]))
    st.append(Paragraph("Archive <b>MADAR-%s-GIS.zip</b> : cibles en points et polygones (GeoJSON, KML pour Google Earth), "
                        "score MADAR en GeoTIFF (20 m) et indices en z-scores (GeoTIFF, 40 m), en WGS84 (EPSG:4326), "
                        "directement exploitables dans QGIS ou ArcGIS." % sid, S["p"]))

    # ---------------- 4. methode
    st.append(Paragraph("4. Méthode et données", S["h1"]))
    rows = [["Données", "ESA Copernicus Sentinel-2 niveau 2A (réflectance de surface), résolution 10–20 m, scènes sans nuages : "
             + ", ".join("%s (%s)" % (s["id"], s["date"]) for s in site["scenes"])],
            ["Masques", "Végétation (NDVI), eau et ombres, nuages et pixels saturés."],
            ["Indices", "Argiles / hydroxyles = B11 / B12 · Oxydes de fer (hématite, goethite) = B4 / B2 · Fer ferreux = B12 / B8A + B3 / B4"
             + (" · Terres rares (expérimental) = (B5 + B8A) / (B6 + B7)" if site["profile"] == "carbonatite" else "")],
            ["Score", "Z-scores robustes (médiane / MAD) des indices du modèle « %s », combinés et lissés ; "
                      "top %s %% retenu ; regroupement en zones ; classement par taille × intensité." % (PROFILE_FR.get(site["profile"], site["profile"]), data.get("top_percent", 1))],
            ["Contrôle", "Site(s) de référence publics ajoutés uniquement après l'analyse (test à l'aveugle)."]]
    t = Table([[Paragraph(a, S["cellb"]), Paragraph(b, S["cell"])] for a, b in rows], colWidths=[26 * mm, 148 * mm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    st.append(t)

    # ---------------- 5. limites
    st.append(Paragraph("5. Limites", S["h1"]))
    for b in ["MADAR réduit et hiérarchise la zone de recherche ; il <b>ne prouve pas l'existence d'un gisement</b>. "
              "L'échantillonnage, les analyses et le forage restent déterminants.",
              "Les signatures de surface peuvent être masquées par le sable, la patine désertique ou des dépôts récents, "
              "et certaines minéralisations (filons profonds, magnétite) ne laissent qu'une faible trace visible.",
              "Des formations sédimentaires naturellement riches en argiles ou en fer, les sebkhas et les anciennes exploitations "
              "peuvent produire des anomalies sans intérêt économique : vérification visuelle et terrain indispensables.",
              "Les coordonnées des sites de référence proviennent de sources publiques et peuvent différer de quelques centaines de mètres."]:
        st.append(Paragraph(b, S["bul"], bulletText="•"))
    st.append(Spacer(1, 10))
    st.append(Paragraph("Ce rapport est fourni à titre d'aide à la décision pour l'exploration. Il ne constitue ni une estimation de "
                        "ressources ni une recommandation d'investissement.", S["small"]))
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print("[ok]", out.relative_to(ROOT))
    return out


def main():
    import tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="")
    ap.add_argument("--client", default="Document de démonstration")
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    data = json.loads((OUT / "minerals.json").read_text())
    tf = OUT / "test" / "test.json"
    test = json.loads(tf.read_text()) if tf.exists() else None
    tmp = Path(tempfile.mkdtemp())
    for site in data["sites"]:
        if a.site and site["id"] != a.site:
            continue
        ref = a.ref or "MM-%s-%s" % (date.today().strftime("%Y%m%d"), site["id"].upper()[:4])
        build(site, data, test, a.client, ref, tmp)
        site["report"] = "%s/MADAR-%s-rapport.pdf" % (site["id"], site["id"])
    (OUT / "minerals.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    page = (ROOT / "minerals_console.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


if __name__ == "__main__":
    main()
