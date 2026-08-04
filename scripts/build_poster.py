"""Rebuild the 36 x 48 in RISE poster as a print-ready PDF.

    python scripts/build_poster.py

Writes docs/poster_rebuild.pdf. Read-only with respect to results.

WHAT THIS IS. A ground-up reportlab reconstruction of the team's PowerPoint poster, keeping its
content and three-column structure but fixing the things a rebuild can fix and a .pptx export
cannot: one typeface everywhere (Times New Roman, registered from the real Windows TTFs so body
text and figures are literally the same face), one colour system, consistent spacing, and a
Figure 5 table computed from the committed results instead of pasted as a screenshot.

CONTENT POLICY. The team called the content "essentially finalized", so this script reproduces
their text verbatim EXCEPT where a statement was factually wrong or already re-approved:
  * Figure 5 / Eq.9 caption said "retention-vs-SNR"; the plotted quantity is absolute macro-F1
    (see docs/POSTER_REVIEW.md item 1). Corrected -- keeping it would restate a known error.
  * "studio" gains the one-clause definition agreed on 2026-08-04 (all 18 DEMAND environments).
  * The Gap paragraph uses the rewrite the team requested and approved.
  * One grammar fix ("which confound" -> "which confounds").
Everything else -- including the Discussion's ranking claims and the "universal benchmark"
conclusion, both flagged in POSTER_REVIEW.md -- is reproduced as authored.

Figure numbering follows the POSTER (Figure 5..9b), not the repo file names: poster Figure 6 is
repo fig6, poster Figure 7 is repo fig8 (confusion grid), poster Figure 8 is repo fig9
(slopegraph). Confusing, but renumbering the team's figures the night before print is worse.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

ASSETS = ROOT / "docs" / "poster_assets"
FIGURES = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "poster_rebuild.pdf"

PAGE_W, PAGE_H = 2592.0, 3456.0          # 36 x 48 in at 72 pt/in

# ---- palette ---------------------------------------------------------------------------------
BU_RED = HexColor("#CC0000")
NAVY = HexColor("#1a2f4b")
INK = HexColor("#1a1a1a")
MUTED = HexColor("#555555")
PAGE_BG = HexColor("#eef1f5")
CARD_BG = white
CARD_EDGE = HexColor("#c9cfd8")

# Model colours = matplotlib default cycle in MODELS order, so the Figure 5 table matches the
# Figure 6 curves by construction rather than by eye.
MODEL_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ---- fonts -----------------------------------------------------------------------------------
def register_fonts() -> None:
    """Register the real Windows Times New Roman family.

    The TTFs (not the PDF core 'Times-Roman') are required for two reasons: full Unicode
    coverage (rho, arrows, and the not-equals sign in the Discussion headers are outside the
    core font's Latin-1), and so the embedded face is byte-identical to what PowerPoint used --
    one Times on the board, not two lookalikes.
    """
    fonts = ROOT.drive + "/Windows/Fonts" if ROOT.drive else "C:/Windows/Fonts"
    pdfmetrics.registerFont(TTFont("TNR", f"{fonts}/times.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-Bold", f"{fonts}/timesbd.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-Italic", f"{fonts}/timesi.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-BoldItalic", f"{fonts}/timesbi.ttf"))
    registerFontFamily(
        "TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic", boldItalic="TNR-BoldItalic"
    )


# ---- styles ----------------------------------------------------------------------------------
def styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName="TNR", textColor=INK, spaceBefore=0, spaceAfter=0)
    return {
        "title": ParagraphStyle("title", fontName="TNR-Bold", fontSize=60, leading=70,
                                alignment=1, textColor=INK),
        "authors": ParagraphStyle("authors", fontName="TNR-Bold", fontSize=30, leading=38,
                                  alignment=1, textColor=INK),
        "affil": ParagraphStyle("affil", fontSize=19, leading=24, alignment=1,
                                textColor=MUTED, **{k: v for k, v in base.items()
                                                    if k not in ("textColor",)}),
        "subhead": ParagraphStyle("subhead", fontName="TNR-Bold", fontSize=27, leading=32,
                                  textColor=BU_RED, spaceBefore=0, spaceAfter=0),
        "body": ParagraphStyle("body", fontSize=19.5, leading=25, **base),
        "bullet": ParagraphStyle("bullet", fontSize=19.5, leading=25, leftIndent=26,
                                 firstLineIndent=-16, **base),
        "caption": ParagraphStyle("caption", fontSize=14.5, leading=18, alignment=1,
                                  fontName="TNR", textColor=MUTED),
        "caption_l": ParagraphStyle("caption_l", fontSize=15.5, leading=19.5, alignment=0,
                                    fontName="TNR", textColor=MUTED),
    }


# ---- low-level helpers -----------------------------------------------------------------------
def draw_para(c: canvas.Canvas, text: str, style: ParagraphStyle,
              x: float, y_top: float, width: float) -> float:
    """Draw wrapped text with its TOP edge at y_top (top-down coordinates). Returns new y_top."""
    p = Paragraph(text, style)
    _, h = p.wrapOn(c, width, PAGE_H)
    p.drawOn(c, x, PAGE_H - y_top - h)
    return y_top + h


def bullets(c: canvas.Canvas, items: list[str], style: ParagraphStyle,
            x: float, y: float, width: float, gap: float = 7) -> float:
    for item in items:
        y = draw_para(c, "\u25cf&nbsp;&nbsp;" + item, style, x, y, width) + gap
    return y - gap


def draw_image(c: canvas.Canvas, path: Path, x: float, y_top: float, width: float,
               center_in: float | None = None) -> float:
    """Place an image scaled to `width`, top at y_top. Returns new y_top."""
    with PILImage.open(path) as im:
        iw, ih = im.size
    h = width * ih / iw
    if center_in is not None:
        x = x + (center_in - width) / 2
    c.drawImage(str(path), x, PAGE_H - y_top - h, width, h,
                preserveAspectRatio=True, mask="auto")
    return y_top + h


def section_bar(c: canvas.Canvas, title: str, x: float, y: float, width: float) -> float:
    h = 58
    c.setFillColor(BU_RED)
    c.roundRect(x, PAGE_H - y - h, width, h, 12, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont("TNR-Bold", 34)
    c.drawCentredString(x + width / 2, PAGE_H - y - h + 15, title)
    return y + h


def card(c: canvas.Canvas, x: float, y_top: float, width: float, height: float) -> None:
    c.setFillColor(CARD_BG)
    c.setStrokeColor(CARD_EDGE)
    c.setLineWidth(1.5)
    c.roundRect(x, PAGE_H - y_top - height, width, height, 16, stroke=1, fill=1)


# ---- Figure 5: native results table ----------------------------------------------------------
def results_table_data() -> list[list[str]]:
    """Clean macro-F1 and per-noise AUC from the committed summaries -- the same loader and AUC
    as scripts/noise_figures.py, so this table cannot drift from Figure 6."""
    import noise_figures as nf
    rows = [["Model", "Clean macro-F1", "AUC white", "AUC audience", "AUC studio"]]
    for key, label in nf.MODELS.items():
        frame = nf.load_model(key)
        if frame is None:
            raise FileNotFoundError(f"no sweep summary for {key}")
        clean = float(frame[~frame["noise_type"].isin(nf.NOISE_TYPES)]["macro_f1"].iloc[0])
        row = [label, f"{clean:.4f}"]
        for noise in nf.NOISE_TYPES:
            grouped = nf.curve(frame, noise, "macro_f1")
            row.append(f"{nf.robustness_auc(grouped['snr_db'].to_numpy(), grouped['mean'].to_numpy()):.4f}")
        rows.append(row)
    return rows


def draw_results_table(c: canvas.Canvas, x: float, y: float, width: float) -> float:
    data = results_table_data()
    # bold the best value in each numeric column
    import numpy as np
    body = np.array([[float(v) for v in r[1:]] for r in data[1:]])
    best = body.argmax(axis=0)

    col_w = [width * 0.20] + [width * 0.20] * 4
    t = Table(data, colWidths=col_w, rowHeights=[52] + [46] * (len(data) - 1))
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "TNR-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "TNR"),
        ("FONTSIZE", (0, 0), (-1, 0), 20),
        ("FONTSIZE", (0, 1), (-1, -1), 20),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f3f6fa")]),
        ("GRID", (0, 1), (-1, -1), 0.8, HexColor("#d5dae2")),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, NAVY),
        ("LEFTPADDING", (0, 0), (0, -1), 18),
    ]
    for i, colour in enumerate(MODEL_COLOURS):
        style.append(("TEXTCOLOR", (0, i + 1), (0, i + 1), HexColor(colour)))
        style.append(("FONTNAME", (0, i + 1), (0, i + 1), "TNR-Bold"))
    for col in range(body.shape[1]):
        style.append(("FONTNAME", (col + 1, int(best[col]) + 1), (col + 1, int(best[col]) + 1),
                      "TNR-Bold"))
    t.setStyle(TableStyle(style))
    _, h = t.wrapOn(c, width, PAGE_H)
    t.drawOn(c, x, PAGE_H - y - h)
    return y + h


# ---- content ---------------------------------------------------------------------------------
TITLE = ("Characterizing the Robustness of Musical Instrument Classification<br/>"
         "Under Acoustic Perturbations")
AUTHORS = ("Max Liu<super>1,5</super>, Allan Yu<super>2,5</super>, Gavin Hu<super>3,5</super>, "
           "Tariq Hossain<super>4,5</super>, Dr. Eugene Pinsky<super>5</super>, "
           "Dr. Indrajit Kalita<super>6</super>")
AFFIL = ("<super>1</super>Clements High School, Sugar Land, TX; "
         "<super>2</super>Fairview High School, Boulder, CO; "
         "<super>3</super>St. Mark's School, Southborough, MA; "
         "<super>4</super>Seven Lakes High School, Katy, TX; "
         "<super>5,6</super>Boston University, 1 Silber Way, Boston, MA 02215")

BACKGROUND = [
    "Instruments' timbral properties are central to how music is composed, performed, and "
    "perceived. As a result, <b>instrument classification is an active research area</b> "
    "(musicology, style analysis, audio editing and retrieval) [1, 2].",
    "Since 2015, convolutional neural networks have pushed clean-audio classification accuracy "
    "from <b>~90% to as high as ~99%</b> [7].",
    "<b>The Gap:</b> model performance is usually measured on clean, isolated audio. It remains "
    "unclear which models fail when real-world noise is introduced, under what noise, and which "
    "instruments break down first.",
]
OBJECTIVES = [
    "We run a controlled, head-to-head <b>robustness comparison of six model families</b>: "
    "handcrafted SVM; custom CNN and CRNN; and fine-tuned, pretrained MERT, AST, and PANNs.",
    "Trained on the same <b>12-instrument dataset</b>, we hold the split and noise corpus "
    "<b>identical across all models</b> and add white (synthetic), audience (human sounds \u2014 "
    "ESC-50), and studio (real-world ambience from all 18 DEMAND environments) noise across an "
    "8-level SNR sweep, <b>measuring the change in classification performance</b>.",
    "<b>Research Question:</b> How do instrument classification systems differ in their ability "
    "to retain classification performance under white, audience, and studio additive noise "
    "across a range of signal-to-noise ratios?",
]
METHODS = [
    "6 models were trained; 3 handcrafted, trained-from-scratch; 3 pretrained approaches",
    "Validation data used to select stopping decisions and hyperparameters, but not final "
    "model fitting",
    "Test data held out during model training; each final model tested once on clean test set",
]
NOISE_CONSTRUCTION = [
    "Noise added to test set (1255 samples)",
    "SNR (Signal-to-noise ratio) is the ratio between the signal (audio source) compared to the "
    "noise. High SNR \u2192 faint noise; 0 dB SNR \u2192 noise is as loud as instrument",
]
STATS = [
    "Model differences were evaluated on <b>identical test recordings</b>; same clip with the "
    "same noise realizations",
    "Confidence intervals were estimated by resampling complete instrument-pitch groups",
    "Acoustic distance and noise-induced confusion were related using <b>Spearman "
    "correlation</b>. Significance assessed using <b>100,000 random instrument-label "
    "permutation tests</b>, corrected using Benjamini\u2013Hochberg correction to account for "
    "pure chance",
]
TAKEAWAYS = [
    "<b>Clean Macro-F1 \u2260 robustness.</b> 5/6 models sat within 2 points on clean audio, "
    "but under white noise SVM fell from 3rd to last (AUC 0.25) and MERT rose from last to 2nd "
    "(0.43).",
    "<b>AST was the most robust model</b> under all 3 noise types (AUC 0.63 / 0.78 / 0.75), "
    "while the weakest model under each varied: SVM under white, MERT under audience, CNN under "
    "studio.",
    "<b>Pretrained Model \u2260 robustness.</b> AST led each noise type, but MERT swung from "
    "2nd under white to last under audience, and PANNs fell to 3rd under white.",
    "<b>Noise character:</b> audience and studio far less damaging than white. At 20 dB, "
    "retention under white vs audience: AST 72% vs 85%, SVM 10% vs 68%.",
    "<b>Instruments:</b> failures were uneven on both clean and noisy sweeps; tuba degraded "
    "most; acoustic distance analysis supports the \u201csimilar instruments confuse more\u201d "
    "association, which is only correlational",
]
CONCLUSION = [
    "<b>Universal benchmark</b> for instrument classification models, tested on six frontier "
    "architectures",
    "<b>Evaluating robustness necessary</b>; clean accuracy alone is insufficient",
    "Noise type <b>equally important</b> as noise strength (SNR ratio)",
    "<b>Failures are structured.</b> Degradation concentrates in tuba, oboe, and trumpet. "
    "Confusion is predicted by acoustic distance in 5 of 18 model/noise tests, most strongly "
    "for AST.",
    "<b>Shared, seed-reproducible noise</b> makes model comparison possible.",
]
LIMITATIONS = [
    "Only <b>one dataset</b> (Philharmonia), restricting timbre to one instrument, player, "
    "and/or recording setup",
    "Recordings are <b>isolated notes</b>, not polyphonic music where several instruments play "
    "at once",
    "Most audio clips were <b>tiled</b> to fill the 3-second standard, which confounds content "
    "with repetition",
    "Nominal SNR and different frequencies of noise may cause instruments to be <b>masked "
    "unequally</b>",
    "SVM, AST, MERT and PANNs are <b>single-seed runs</b>, while CNN and CRNN are multi-seed "
    "spread, so small differences are not treated as effects",
]
FUTURE = [
    "Polyphonic and multi-dataset audio",
    "Independently recorded real-world noise, more real acoustic issues",
    "More noise realizations to increase robustness against random draw",
    "Noise-aware training to further noise robustness",
]
ACK = ("This work was created in affiliation with the Boston University RISE program. We are "
       "grateful to Boston University for this opportunity, and to our instructors Dr. Eugene "
       "Pinsky and Dr. Indrajit Kalita for their commitment and hard work. We acknowledge the "
       "use of Anthropic's Claude Code for assistance with programming and manuscript "
       "formatting. All generated material was reviewed and verified by the authors, who take "
       "full responsibility for the final work.")

CAP_FIG5 = ("<b>Figure 5.</b> Clean baseline and robustness AUC \u2014 area under the "
            "macro-F1-vs-SNR curve, dB-weighted (1.0 = perfect classification at every SNR). "
            "Bold marks the best value per column; model colours match Figure 6.")
CAP_FIG6 = ("<b>Figure 6.</b> Robustness curves for all 6 models across 3 noise types "
            "(columns): macro-F1 (top row) and retention relative to clean score (bottom row), "
            "versus SNR from 50 to \u221210 dB. Shaded band is the spread across two noise "
            "draws; dotted line is 12-class chance (0.083).")
CAP_FIG7 = ("<b>Figure 7.</b> Clean-audio confusion for all six models. Each cell is the share "
            "of a true instrument sent to the wrong instrument, row-normalised; the diagonal "
            "(correct predictions) is masked in grey so the colour scale is set by the largest "
            "confusion. Panel titles give clean macro-F1 and total misclassified windows.")
CAP_FIG8 = ("<b>Figure 8.</b> Instruments ranked by clean accuracy (left) and noise robustness "
            "(right), best at top. The rankings are unrelated (Spearman \u03c1 = \u22120.16, "
            "p = 0.62). Tuba is classified perfectly by all six models but loses the most "
            "recall under noise (0.62); violin is 43 clean errors yet the most robust "
            "(0.19 lost).")
CAP_FIG9A = ("<b>Figure 9a.</b> Acoustic distance vs confusion associations across model and "
             "noise types")
CAP_FIG9B = ("<b>Figure 9b.</b> Illustrative AST \u00d7 ESC-50 human non-speech relationship "
             "across 66 instrument pairs")
MID_STATS = [
    "Paired comparisons used identical test windows and noise realizations. CNN exceeded CRNN "
    "by an average 0.1126 Macro-F1 over both realizations (0 &amp; 1: 0.1122, 0.1130) under "
    "white noise at 20 dB SNR; 95% pitch-group bootstrap intervals r<sub>0</sub> = [0.0560, "
    "0.1625] and r<sub>1</sub> = [0.0589, 0.1601] excluded zero",
    "Acoustically closer instrument pairs showed greater confusion; 5 of 18 model-noise tests "
    "are significant after Benjamini\u2013Hochberg; strongest for AST under human non-speech "
    "(Spearman \u03c1 = \u22120.607, permutation p = 0.000010, adjusted q = 0.000180 &lt; 0.05).",
]

EQ_CAPTIONS = [
    ("eq7_macrof1.png", "<b>Eq.7 Macro-F1</b> Primary evaluation metric; 12-instrument average"),
    ("eq8_retention.png", "<b>Eq.8 Retention</b> Standardized measure of robustness; how much "
     "clean score was retained after noise added"),
    ("eq9_auc.png", "<b>Eq.9 Area Under Robustness Curve</b> dB-weighted average macro-F1 "
     "under all eight noise levels"),
    ("eq10_accuracy.png", "<b>Eq.10 Accuracy</b> Plain model correctness rate"),
]


# ---- assembly --------------------------------------------------------------------------------
def build() -> None:
    register_fonts()
    st = styles()
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("Characterizing the Robustness of Musical Instrument Classification "
               "Under Acoustic Perturbations")

    # page background
    c.setFillColor(PAGE_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    # ---- header ----
    margin = 45.0
    draw_image(c, ASSETS / "bu_logo.jpeg", margin + 6, 52, 300)
    y = draw_para(c, TITLE, st["title"], margin + 310, 52, PAGE_W - 2 * (margin + 310)) + 14
    y = draw_para(c, AUTHORS, st["authors"], margin, y, PAGE_W - 2 * margin) + 10
    y = draw_para(c, AFFIL, st["affil"], margin, y, PAGE_W - 2 * margin) + 18
    c.setStrokeColor(BU_RED)
    c.setLineWidth(5)
    c.line(margin, PAGE_H - y, PAGE_W - margin, PAGE_H - y)
    top = y + 26

    # ---- columns ----
    gutter = 40.0
    usable = PAGE_W - 2 * margin - 2 * gutter
    w_side = usable * 0.28
    w_mid = usable * 0.44
    x_l = margin
    x_m = x_l + w_side + gutter
    x_r = x_m + w_mid + gutter
    col_h = PAGE_H - top - margin
    pad = 24.0

    for x, w in ((x_l, w_side), (x_m, w_mid), (x_r, w_side)):
        card(c, x, top, w, col_h)

    # ================= LEFT =================
    x, w = x_l + pad, w_side - 2 * pad
    y = top + pad
    y = section_bar(c, "Introduction", x, y, w) + 18
    y = draw_para(c, "Background", st["subhead"], x, y, w) + 10
    y = bullets(c, BACKGROUND, st["bullet"], x, y, w) + 14
    y = draw_image(c, ASSETS / "fig1_spectrograms.jpeg", x, y, w) + 6
    y = draw_para(c, "<b>Fig.1</b> Clean and under Gaussian noise (20 dB, 0 dB)", st["caption_l"], x, y, w) + 18
    y = draw_para(c, "Objectives", st["subhead"], x, y, w) + 10
    y = bullets(c, OBJECTIVES, st["bullet"], x, y, w) + 20

    y = section_bar(c, "Methods", x, y, w) + 18
    y = draw_image(c, ASSETS / "fig2_pipeline.jpeg", x, y, w) + 6
    y = draw_para(c, "<b>Fig.2</b> Methods pipeline summary, beginning to end",
                  st["caption_l"], x, y, w) + 12
    y = bullets(c, METHODS, st["bullet"], x, y, w) + 14
    y = draw_image(c, ASSETS / "fig3_scratch_models.jpeg", x, y, w * 0.92, center_in=w) + 6
    y = draw_para(c, "<b>Fig.3</b> Models trained from scratch: input and what is learned",
                  st["caption_l"], x, y, w) + 12
    y = draw_image(c, ASSETS / "fig4_pretrained_models.jpeg", x, y, w * 0.92, center_in=w) + 6
    y = draw_para(c, "<b>Fig.4</b> Pretrained backbones: pretraining corpus, input, and what "
                     "is learned", st["caption_l"], x, y, w) + 18

    y = draw_para(c, "Noise Construction", st["subhead"], x, y, w) + 10
    y = bullets(c, NOISE_CONSTRUCTION, st["bullet"], x, y, w) + 12
    # The equation block is reproduced as ONE crop of the original poster page rather than as
    # separately extracted images. The raw embedded streams for this region decode as black
    # boxes (flattened transparency), and per-equation crops collide with the caption text the
    # original placed hard against each formula. Cropping what the reader actually saw
    # sidesteps both. Equations keep their original math face; math is exempt from the Times
    # standardisation.
    y = draw_image(c, ASSETS / "eq_block.png", x, y, w * 0.96, center_in=w) + 14

    left_end = y

    # ================= MIDDLE =================
    x, w = x_m + pad, w_mid - 2 * pad
    y = top + pad
    y = section_bar(c, "Results", x, y, w) + 20
    y = draw_results_table(c, x, y, w) + 10
    y = draw_para(c, CAP_FIG5, st["caption"], x, y, w) + 22
    y = draw_image(c, FIGURES / "fig6_robustness_curves.png", x, y, w * 0.95, center_in=w) + 8
    y = draw_para(c, CAP_FIG6, st["caption"], x, y, w) + 22
    y = draw_image(c, FIGURES / "fig8_confusion_grid_errors.png", x, y, w * 0.93,
                   center_in=w) + 8
    y = draw_para(c, CAP_FIG7, st["caption"], x, y, w) + 22
    y = draw_image(c, FIGURES / "fig9_rank_slope.png", x, y, w * 0.95, center_in=w) + 8
    y = draw_para(c, CAP_FIG8, st["caption"], x, y, w) + 22

    pair_w = (w - 28) / 2
    ya = draw_image(c, ASSETS / "fig9a_distance_heatmap.jpeg", x, y, pair_w)
    yb = draw_image(c, ASSETS / "fig9b_ast_scatter.jpeg", x + pair_w + 28, y, pair_w)
    yy = max(ya, yb) + 8
    ya = draw_para(c, CAP_FIG9A, st["caption"], x, yy, pair_w)
    yb = draw_para(c, CAP_FIG9B, st["caption"], x + pair_w + 28, yy, pair_w)
    mid_end = max(ya, yb)

    # ================= RIGHT =================
    x, w = x_r + pad, w_side - 2 * pad
    y = top + pad
    y = section_bar(c, "Discussion", x, y, w) + 18
    y = draw_para(c, "Key Takeaways", st["subhead"], x, y, w) + 10
    y = bullets(c, TAKEAWAYS, st["bullet"], x, y, w) + 18
    y = draw_para(c, "Conclusion", st["subhead"], x, y, w) + 10
    y = bullets(c, CONCLUSION, st["bullet"], x, y, w) + 18
    y = draw_para(c, "Limitations", st["subhead"], x, y, w) + 10
    y = bullets(c, LIMITATIONS, st["bullet"], x, y, w) + 18
    y = draw_para(c, "Future Work", st["subhead"], x, y, w) + 10
    y = bullets(c, FUTURE, st["bullet"], x, y, w) + 18
    y = draw_para(c, "Statistical Analysis", st["subhead"], x, y, w) + 10
    small = ParagraphStyle("stat", parent=st["bullet"], fontSize=17.5, leading=22.5)
    y = bullets(c, STATS + MID_STATS, small, x, y, w)
    right_end = y

    # References and Acknowledgments anchor to the BOTTOM of the column, so the free space
    # left by the shorter Discussion content becomes breathing room in the middle rather than
    # a ragged hole at the foot of the board.
    ack_style = ParagraphStyle("ack", parent=st["body"], fontSize=16, leading=21)
    ack_p = Paragraph(ACK, ack_style)
    _, ack_h = ack_p.wrapOn(c, w, PAGE_H)
    bottom_limit = top + col_h - pad
    qr_w = 165.0
    ref_block = 58 + 16 + qr_w + 34 + 26            # bar + gap + QR + label + gap
    ack_block = 58 + 14 + ack_h
    y = bottom_limit - ack_block - ref_block - 14
    y = section_bar(c, "References", x, y, w) + 16
    qx = x + (w - 2 * qr_w - 60) / 2
    draw_image(c, ASSETS / "qr_works_cited.jpeg", qx, y, qr_w)
    y2 = draw_image(c, ASSETS / "qr_github.jpeg", qx + qr_w + 60, y, qr_w)
    c.setFont("TNR-Bold", 18)
    c.setFillColor(INK)
    c.drawCentredString(qx + qr_w / 2, PAGE_H - y2 - 24, "Works Cited")
    c.drawCentredString(qx + qr_w + 60 + qr_w / 2, PAGE_H - y2 - 24, "GitHub")
    y = y2 + 40
    y = section_bar(c, "Acknowledgments", x, y, w) + 14
    y = draw_para(c, ACK, ack_style, x, y, w)
    right_end = max(right_end, y)

    c.save()
    for name, end in (("left", left_end), ("mid", mid_end), ("right", right_end)):
        over = end - (top + col_h - pad)
        state = f"OVERFLOWS by {over:.0f}pt" if over > 0 else f"{-over:.0f}pt spare"
        print(f"  {name:5} column: ends {end:.0f} / limit {top + col_h - pad:.0f}  ({state})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
