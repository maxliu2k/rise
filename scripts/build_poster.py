"""Rebuild the 36 x 48 in RISE poster as a print-ready PDF.

    python scripts/build_poster.py

Writes output/pdf/poster_rebuild_revised_v3.pdf. Read-only with respect to results.

WHAT THIS IS. A ground-up reportlab reconstruction of the team's PowerPoint poster, keeping its
content and three-column structure while making every text element on the board Times New Roman
(registered from the real Windows TTFs, so body text and figures are literally the same face).
Beyond restyling, the rebuild replaces raster elements with native ones wherever the underlying
data or text is available:

  * Figure 5 results table  -- computed from the committed sweep summaries (same loader and AUC
    as noise_figures.py; cannot drift from Figure 6)
  * Fig.2 pipeline chart    -- redrawn natively (was a DejaVu Sans image)
  * Fig.3 / Fig.4 tables    -- native reportlab tables (were images)
  * Fig.1 labels            -- the baked title/label rows are cropped off and re-set in Times
  * Figures 8 / 9           -- generated from the canonical failure-analysis artifacts;
                               statistics are read, never recomputed
  * Eq.7-10                 -- typeset with STIX mathtext (Times-compatible); the originals
                               carried heavy black/blue borders. Eq.1-6 remain one crop of the
                               original page: math is exempt from the face standardisation.

Figure 7 uses the BLUE diagonal-shown confusion grid (team preference, 2026-08-04), not the
errors-masked variant; the caption describes what is actually shown.

CONTENT POLICY. The team called the content "essentially finalized", so text is reproduced
verbatim EXCEPT where a statement was factually wrong or already re-approved (see
docs/POSTER_REVIEW.md): the Figure 5 / Eq.9 wording says macro-F1 (it said retention, which the
plotted number is not), "studio" carries the agreed 18-environment definition, the approved Gap
rewrite, one grammar fix, and affiliation 6 (Dr. Kalita) is defined -- the original left it
dangling. The flagged-but-unapproved items (ranking claims, "universal benchmark") are
reproduced as authored. The Dataset & split box and the at-a-glance strip were added on request
to fill free space; both restate verified numbers already in the repo.

Figure numbering follows the POSTER (Figure 5..9b), not repo file names: poster Figure 6 is
repo fig6, poster Figure 7 is repo fig8 (confusion grid), poster Figure 8 is repo fig9
(slopegraph).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily, stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

ASSETS = ROOT / "docs" / "poster_assets"
FIGURES = ROOT / "docs" / "figures"
OUT = ROOT / "output" / "pdf" / "poster_rebuild_revised_v3.pdf"

PAGE_W, PAGE_H = 2592.0, 3456.0          # 36 x 48 in at 72 pt/in

# BU logo width. Sized against the TITLE, not chosen for its own sake: the title is centred on the
# page, so the logo and the title font size compete for the same horizontal space (~40pt of logo
# per 2pt of title). Raising this without dropping the title size trips the assert in main().
LOGO_W = 185.0                           # 2.6 in

# ---- palette ---------------------------------------------------------------------------------
BU_RED = HexColor("#CC0000")
NAVY = HexColor("#1a2f4b")
INK = HexColor("#1a1a1a")
MUTED = HexColor("#555555")
PAGE_BG = HexColor("#eef1f5")
CARD_BG = white
CARD_EDGE = HexColor("#c9cfd8")
BOX_BLUE = HexColor("#d9ecf7")
BOX_EDGE = HexColor("#8fb8d8")
ROW_TINT = HexColor("#eaf1f9")

# Model colours = matplotlib default cycle in MODELS order, so the Figure 5 table matches the
# Figure 6 curves by construction rather than by eye.
MODEL_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ---- fonts -----------------------------------------------------------------------------------
def register_fonts() -> None:
    """Register the real Windows Times New Roman family.

    The TTFs (not the PDF core 'Times-Roman') are required for two reasons: full Unicode
    coverage (rho, arrows, and the not-equals sign in the Discussion are outside the core
    font's Latin-1), and so the embedded face is byte-identical to what PowerPoint used.
    """
    windows = Path("C:/Windows/Fonts")
    macos = Path("/System/Library/Fonts/Supplemental")
    if windows.is_dir():
        paths = {
            "TNR": windows / "times.ttf",
            "TNR-Bold": windows / "timesbd.ttf",
            "TNR-Italic": windows / "timesi.ttf",
            "TNR-BoldItalic": windows / "timesbi.ttf",
        }
    elif macos.is_dir():
        paths = {
            "TNR": macos / "Times New Roman.ttf",
            "TNR-Bold": macos / "Times New Roman Bold.ttf",
            "TNR-Italic": macos / "Times New Roman Italic.ttf",
            "TNR-BoldItalic": macos / "Times New Roman Bold Italic.ttf",
        }
    else:
        raise FileNotFoundError("Times New Roman fonts were not found")
    for name, path in paths.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    registerFontFamily(
        "TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic", boldItalic="TNR-BoldItalic"
    )


# ---- styles ----------------------------------------------------------------------------------
def styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName="TNR", textColor=INK, spaceBefore=0, spaceAfter=0)
    return {
        # 50/60 rather than 60/70. The title is 91 characters, and at 60pt it wraps to a second
        # line -- which costs 70pt off every column's height against a layout that ships with 3pt
        # of slack (see the overflow check in main). So it has to fit on one line.
        #
        # One line means the text must clear the logo, and the two trade directly: the title is
        # page-centred, so every 2pt of title size costs ~40pt of logo width. 50pt is the largest
        # size that leaves the logo legible (LOGO_W below, 2.6in) with a ~49pt gap, and it is
        # still 1.67x the 30pt author line. Changing either number requires re-checking the other.
        "title": ParagraphStyle("title", fontName="TNR-Bold", fontSize=50, leading=60,
                                alignment=1, textColor=INK),
        "authors": ParagraphStyle("authors", fontName="TNR-Bold", fontSize=30, leading=38,
                                  alignment=1, textColor=INK),
        "affil": ParagraphStyle("affil", fontSize=19, leading=24, alignment=1,
                                textColor=MUTED, fontName="TNR"),
        "subhead": ParagraphStyle("subhead", fontName="TNR-Bold", fontSize=27, leading=32,
                                  textColor=BU_RED),
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
    rows = [["Model", "Clean macro-F1", "AUC white", "AUC human", "AUC environ."]]
    for key, label in nf.MODELS.items():
        frame = nf.load_model(key)
        if frame is None:
            raise FileNotFoundError(f"no sweep summary for {key}")
        clean = float(frame[~frame["noise_type"].isin(nf.NOISE_TYPES)]["macro_f1"].iloc[0])
        row = [label, f"{clean:.4f}"]
        for noise in nf.NOISE_TYPES:
            grouped = nf.curve(frame, noise, "macro_f1_retention")
            row.append(f"{nf.robustness_auc(grouped['snr_db'].to_numpy(), grouped['mean'].to_numpy()):.4f}")
        rows.append(row)
    return rows


def draw_results_table(c: canvas.Canvas, x: float, y: float, width: float) -> float:
    data = results_table_data()
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


# ---- native Fig.1 / Fig.2 / Fig.3-4 ----------------------------------------------------------
def fig1_block(c: canvas.Canvas, st: dict, x: float, y: float, w: float) -> float:
    """The spectrogram strip with its DejaVu title/label rows cropped off and re-set in Times."""
    label_y = y
    c.setFont("TNR", 16)
    c.setFillColor(INK)
    for i, label in enumerate(("Clean", "20 dB SNR", "0 dB SNR")):
        c.drawCentredString(x + w * (i * 2 + 1) / 6.0, PAGE_H - label_y - 14, label)
    y = label_y + 22
    y_img = draw_image(c, ASSETS / "fig1_panels.png", x, y, w)
    # rotated axis label, as the original had, but in Times italic
    c.saveState()
    c.setFont("TNR-Italic", 13)
    c.setFillColor(MUTED)
    c.translate(x - 8, PAGE_H - (y + (y_img - y) / 2))
    c.rotate(90)
    c.drawCentredString(0, 0, "White (Gaussian)")
    c.restoreState()
    return y_img


PIPELINE = [
    ("Source Recordings",
     "Philharmonia Orchestra\n12 instruments\nOne labeled note / file"),
    ("Preprocessing",
     "Filter articulation \u00b7 tile silence\nResample \u00b7 one 3 s window\n"
     "Normalize loudness\n\u2192 8,374 retained recordings"),
    ("Grouped Split",
     "Stratified by instrument,\nsplit by pitch 70/15/15\n"
     "Train 5,861 \u00b7 Val 1,258\nTest 1,255"),
    ("Evaluate / Score Models",
     "Identical noisy mixtures\nper model\nMacro-F1 retention across SNR"),
    ("Construct Noisy Test Set",
     "White (Gaussian)\nHuman non-speech (ESC-50)\nEnvironmental (DEMAND)\n"
     "8 SNRs \u00d7 2 independent draws"),
    ("Train Models",
     "Trained: SVM \u00b7 CNN \u00b7 CRNN\nFine-tuned: MERT \u00b7 AST \u00b7 PANNs"),
]


def pipeline_chart(c: canvas.Canvas, x: float, y: float, w: float) -> float:
    """Native redraw of the Fig.2 flowchart: top row left-to-right, down, bottom row
    right-to-left -- the same S-shape as the original, with numbered step badges so the
    serpentine reading order is explicit.

    Text is measured and vertically CENTRED in each box. The first version top-anchored it,
    which left three-line boxes with a slab of dead space at the foot and made the whole
    chart read as slumped.
    """
    import math
    gap = 20.0
    box_w = (w - 2 * gap) / 3
    box_h = 142.0
    row_gap = 42.0
    # step number by process order, not grid order: the bottom row runs right-to-left
    step_of = {0: 1, 1: 2, 2: 3, 3: 6, 4: 5, 5: 4}

    def box(i: int, title: str, body: str) -> None:
        col, row = i % 3, i // 3
        bx = x + col * (box_w + gap)
        by = y + row * (box_h + row_gap)
        c.setFillColor(BOX_BLUE)
        c.setStrokeColor(BOX_EDGE)
        c.setLineWidth(1.4)
        c.roundRect(bx, PAGE_H - by - box_h, box_w, box_h, 12, stroke=1, fill=1)
        # step badge
        c.setFillColor(BU_RED)
        c.circle(bx + 22, PAGE_H - by - 22, 13, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont("TNR-Bold", 15)
        c.drawCentredString(bx + 22, PAGE_H - by - 27, str(step_of[i]))
        # measured, centred text block
        lines = body.split("\n")
        line_h = 15.5
        # ONE title height for every box: per-box vertical centring made titles land at
        # different heights depending on how many body lines followed, and the row read as
        # misaligned. Title is fixed; the body block centres in the space beneath it.
        ty = by + 52
        c.setFillColor(NAVY)
        c.setFont("TNR-Bold", 16)
        c.drawCentredString(bx + box_w / 2, PAGE_H - ty, title)
        region_top, region_bottom = ty + 10, by + box_h - 10
        body_h = len(lines) * line_h
        y0 = region_top + (region_bottom - region_top - body_h) / 2 + 11
        c.setFont("TNR", 12)
        c.setFillColor(INK)
        for k, line in enumerate(lines):
            c.drawCentredString(bx + box_w / 2, PAGE_H - (y0 + k * line_h), line)

    for i, (title, body) in enumerate(PIPELINE):
        box(i, title, body)

    c.setStrokeColor(NAVY)
    c.setLineWidth(2.5)

    def arrow(x1, y1, x2, y2):
        c.line(x1, PAGE_H - y1, x2, PAGE_H - y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        for side in (-0.5, 0.5):
            c.line(x2, PAGE_H - y2,
                   x2 - 11 * math.cos(angle + side), PAGE_H - y2 + 11 * math.sin(angle + side))

    mid_y1 = y + box_h / 2
    mid_y2 = y + box_h + row_gap + box_h / 2
    arrow(x + box_w + 3, mid_y1, x + box_w + gap - 3, mid_y1)
    arrow(x + 2 * box_w + gap + 3, mid_y1, x + 2 * box_w + 2 * gap - 3, mid_y1)
    cx3 = x + 2 * box_w + 2 * gap + box_w / 2
    arrow(cx3, y + box_h + 3, cx3, y + box_h + row_gap - 3)
    arrow(x + 2 * box_w + 2 * gap - 3, mid_y2, x + 2 * box_w + gap + 3, mid_y2)
    arrow(x + box_w + gap - 3, mid_y2, x + box_w + 3, mid_y2)
    return y + 2 * box_h + row_gap


SCRATCH_ROWS = [
    ("SVM", "", "88 handcrafted audio features", "Timbre, spectrum, pitch, loudness",
     "RBF decision boundary", ""),
    ("CNN", "", "128-band log-Mel spectrogram", "", "Convolutional features + classifier", ""),
    ("CRNN", "", "128-band log-Mel spectrogram", "", "Convolutional and recurrent",
     "Features plus classifier"),
]
PRETRAINED_ROWS = [
    ("MERT", "Music audio", "24 kHz waveform", "13 time-averaged layer outputs",
     "Fine-tuned backbone", "Plus 12-class head"),
    ("AST", "AudioSet", "16 kHz waveform", "Official extractor to log-Mel",
     "Transformer backbone", "Plus 12-class head"),
    ("PANNs", "AudioSet", "32 kHz waveform", "CNN14 computes 64-band log-Mel",
     "CNN14 backbone", "Plus 12-class head"),
]


def model_table(c: canvas.Canvas, st: dict, x: float, y: float, w: float,
                title: str, subtitle: str, rows: list[tuple]) -> float:
    y = draw_para(c, f'<para alignment="center">{title}</para>',
                  ParagraphStyle("mt", parent=st["body"], fontName="TNR-Bold", fontSize=20,
                                 textColor=NAVY), x, y, w) + 2
    y = draw_para(c, f'<para alignment="center">{subtitle}</para>',
                  ParagraphStyle("ms", parent=st["caption_l"], fontSize=14, alignment=1),
                  x, y, w) + 8

    def cell(main: str, sub: str, bold: bool = False, colour: str = "#1a1a1a") -> Paragraph:
        face = "TNR-Bold" if bold else "TNR"
        sub_html = (f'<br/><font size="12" color="#5a6b7d">{sub}</font>') if sub else ""
        return Paragraph(
            f'<para alignment="center"><font name="{face}" size="15" color="{colour}">'
            f"{main}</font>{sub_html}</para>", st["body"])

    header = [Paragraph(f'<para alignment="center"><font name="TNR-Bold" size="15" '
                        f'color="white">{h}</font></para>', st["body"])
              for h in ("Model", "Input to classifier", "What's learned on this dataset")]
    data = [header]
    for model, msub, in_main, in_sub, learn_main, learn_sub in rows:
        data.append([cell(model, msub, bold=True), cell(in_main, in_sub),
                     cell(learn_main, learn_sub)])
    t = Table(data, colWidths=[w * 0.20, w * 0.40, w * 0.40])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_TINT, white]),
        ("GRID", (0, 1), (-1, -1), 0.7, HexColor("#c4d2e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    _, h = t.wrapOn(c, w, PAGE_H)
    t.drawOn(c, x, PAGE_H - y - h)
    return y + h


# ---- content ---------------------------------------------------------------------------------
TITLE = "Characterizing Robustness of Musical Instrument Classification Under Acoustic Perturbations"
SUP = '<super><font size="17">{}</font></super>'
AUTHORS = (f"Max Liu{SUP.format('1,5')}, Allan Yu{SUP.format('2,5')}, "
           f"Gavin Hu{SUP.format('3,5')}, Tariq Hossain{SUP.format('4,5')}, "
           f"Dr. Eugene Pinsky{SUP.format('5')}, Dr. Indrajit Kalita{SUP.format('5')}")
ASUP = '<super><font size="11">{}</font></super>'
AFFIL = (f"{ASUP.format('1')}Clements High School, Sugar Land, TX; "
         f"{ASUP.format('2')}Fairview High School, Boulder, CO; "
         f"{ASUP.format('3')}St. Mark's School, Southborough, MA; "
         f"{ASUP.format('4')}Seven Lakes High School, Katy, TX; "
         f"{ASUP.format('5')}Boston University, 1 Silber Way, Boston, MA 02215")

BACKGROUND = [
    "Instruments' timbral properties are central to how music is composed, performed, and "
    "perceived. As a result, instrument classification is an active research area "
    "(musicology, style analysis, audio editing and retrieval) [1, 2].",
    "Since 2015, convolutional neural networks have pushed clean-audio classification accuracy "
    "from ~90% to as high as ~99% [7].",
    "<b>The Gap:</b> model performance is usually measured on clean, isolated audio. It remains "
    "unclear which models fail when real-world noise is introduced, under what noise, and which "
    "instruments break down first.",
]
OBJECTIVES = [
    "We run a controlled, head-to-head robustness comparison of six model families: "
    "handcrafted SVM; custom CNN and CRNN; and fine-tuned, pretrained MERT, AST, and PANNs.",
    "Trained on the same 12-instrument dataset, we hold the split and noise corpus "
    "identical across all models and add white (synthetic), human non-speech (ESC-50), and "
    "environmental (all 18 DEMAND environments) noise across an "
    "8-level SNR sweep, measuring the change in classification performance.",
    "<b>Research Question:</b> How do instrument classification systems differ in their ability "
    "to retain classification performance under white, human non-speech, and environmental "
    "additive noise "
    "across a range of signal-to-noise ratios?",
]
METHODS = [
    "6 models were trained; 3 handcrafted, trained-from-scratch; 3 pretrained approaches",
    "Validation data used to select stopping decisions and hyperparameters, but not final "
    "model fitting",
    "Test data held out during model training; each final model tested once on clean test set",
]
DATASET_STATS = [("12", "instruments"), ("8,374", "windows"), ("3 s", "@ 22.05 kHz"),
                 ("5,861", "train"), ("1,258", "validation"), ("1,255", "test")]
DATASET_NOTE = ("Split is grouped by (instrument, note): every recording of a pitch lands in "
                "exactly one split, so near-duplicate takes cannot leak between train and "
                "test. A no-leak assertion runs on every dataset build.")
NOISE_CONSTRUCTION = [
    "Noise added to test set (1255 samples)",
    "SNR (Signal-to-noise ratio) is the ratio between the signal (audio source) compared to the "
    "noise. High SNR \u2192 faint noise; 0 dB SNR \u2192 noise is as loud as instrument",
]
STATS = [
    "Model differences were evaluated on identical test recordings; same clip with the "
    "same noise realizations",
    "Confidence intervals were estimated by resampling complete instrument-pitch groups",
    "Acoustic distance and noise-induced confusion were related using Spearman "
    "correlation. Significance assessed using 100,000 random instrument-label "
    "permutation tests, corrected using Benjamini\u2013Hochberg correction to account for "
    "pure chance",
]
MID_STATS = [
    "Paired comparisons used identical test windows and noise realizations. CNN exceeded CRNN "
    "by an average 0.1126 Macro-F1 over both realizations (0 &amp; 1: 0.1122, 0.1130) under "
    "white noise at 20 dB SNR; 95% pitch-group bootstrap intervals r<sub>0</sub> = [0.0560, "
    "0.1625] and r<sub>1</sub> = [0.0589, 0.1601] excluded zero",
    "Acoustically closer instrument pairs showed greater confusion; 7 of 18 model-noise tests "
    "are significant after Benjamini\u2013Hochberg; strongest for AST under human non-speech "
    "(Spearman \u03c1 = \u22120.607, permutation p = 0.000010, adjusted q = 0.000180 &lt; 0.05).",
]
TAKEAWAYS = [
    "<b>Clean macro-F1 did not predict robustness.</b> All six clean scores spanned only 2.0 "
    "points, but white-noise retention AUC ranged from 0.259 (SVM) to 0.636 (AST).",
    "<b>AST had the highest observed robustness</b> under all 3 noise types "
    "(AUC 0.636 / 0.791 / 0.753). The weakest model varied: SVM under white and human "
    "non-speech, CNN under environmental noise.",
    "<b>Pretrained systems occupied the top three positions</b> under every noise type, but "
    "the study does not isolate pretraining from architecture, input, or sample rate.",
    "<b>Noise category mattered:</b> recorded noise was less damaging than white at matched "
    "nominal SNR. At 20 dB, AST retained 72.3% vs 85.3%; SVM retained 10.4% vs 68.0%.",
    "<b>Instrument failures were uneven:</b> tuba had the greatest overall recall-loss AUC. "
    "Acoustic distance was associated with confusion in 7 of 18 corrected tests, but the "
    "relationship is correlational.",
]
CONCLUSION = [
    "<b>Reproducible benchmark</b> spanning six instrument-classification systems",
    "<b>Evaluating robustness necessary</b>; clean accuracy alone is insufficient",
    "Noise category changes degradation at matched nominal SNR",
    "<b>Failures are structured.</b> Degradation concentrates in tuba, oboe, and trumpet. "
    "Confusion is associated with acoustic distance in 7 of 18 model/noise tests, most strongly "
    "for AST.",
    "<b>Shared, seed-reproducible noise</b> makes model comparison possible.",
]
LIMITATIONS = [
    "Only one dataset (Philharmonia), restricting timbre to one instrument, player, "
    "and/or recording setup",
    "Recordings are isolated notes, not polyphonic music where several instruments play "
    "at once",
    "Most audio clips were tiled to fill the 3-second standard, which confounds content "
    "with repetition",
    "Nominal full-band SNR and model-specific frontends can change effective masking; "
    "cross-model differences describe complete systems",
    "SVM, AST, MERT and PANNs are single-seed runs, while CNN and CRNN are multi-seed "
    "spread, so small differences are not treated as effects",
]
FUTURE = [
    "Polyphonic and multi-dataset audio",
    "Independently recorded real-world noise, more real acoustic issues",
    "More noise realizations to increase robustness against random draw",
    "Noise-aware training to further noise robustness",
]
NOISE_SOURCES = [
    ("white", "generated Gaussian noise — synthetic broadband control"),
    ("human", "ESC-50 human non-speech (targets 20–29): 400 clips, 10 classes — "
     "clapping, coughing, laughing, footsteps …"),
    ("environmental", "DEMAND: 18 real environments (domestic, nature, office, public, street, "
     "transport), one microphone per array"),
]
SEED_NOTE = ("Every mixture is seeded by sha256(dataset fingerprint | window | noise type | "
             "replicate), so the 60,240-file corpus is bit-reproducible and identical for "
             "every model.")
REPRO = [
    "All per-window predictions, per-condition metrics, and the failure analysis are "
    "committed to the repository",
    "Every figure and table on this poster regenerates from one script each "
    "(scripts/*.py) against those files",
    "The noise corpus rebuilds bit-identically from the sealed dataset fingerprint",
]
RESULT_HIGHLIGHTS = [
    "<b>Noise category mattered:</b> at 20 dB, SVM retained 10% under white noise and 68% "
    "under human non-speech noise.",
    "<b>Clean accuracy hid fragility:</b> tuba had no clean errors across the six models but "
    "the greatest mean recall-loss AUC under noise.",
    "<b>Failures were structured:</b> 7 of 18 distance-confusion tests remained significant "
    "after BH correction.",
    "<b>At 20 dB white noise,</b> CNN exceeded CRNN by 0.1126 macro-F1; both "
    "realization-specific bootstrap intervals excluded zero.",
]
ACK = ("This work was created in affiliation with the Boston University RISE program. We are "
       "grateful to Boston University for this opportunity, and to our instructors Dr. Eugene "
       "Pinsky and Dr. Indrajit Kalita for their commitment and hard work. We acknowledge the "
       "use of Anthropic's Claude Code for assistance with programming and manuscript "
       "formatting. All generated material was reviewed and verified by the authors, who take "
       "full responsibility for the final work.")

CAP_FIG5 = ("<b>Figure 5.</b> Clean baseline and normalized robustness AUC \u2014 area under the "
            "retention-vs-SNR curve, dB-weighted (1.0 = no degradation across SNR). "
            "Bold marks the best value per column; model colours match Figure 6.")
CAP_FIG6 = ("<b>Figure 6.</b> Macro-F1 (top) and retention relative to clean macro-F1 "
            "(bottom) versus SNR for all six models and three noise categories. Each point "
            "averages two noise realizations; shading shows their standard deviation.")
CAP_FIG7 = ("<b>Figure 7.</b> Clean-audio confusion for all six models, row-normalised: the "
            "diagonal is per-class recall, off-diagonal cells are the share of a true "
            "instrument sent elsewhere, on one colour scale across panels. Panel titles give "
            "clean macro-F1 and total misclassified windows.")
CAP_FIG8 = ("<b>Figure 8.</b> Instruments ranked by clean errors (left) and mean recall-loss "
            "AUC under noise (right), best at top. The rankings were unrelated "
            "(Spearman \u03c1 = \u22120.18, p = 0.59). Tuba had no clean errors but the greatest "
            "recall loss (0.60); violin had 22 clean errors but the smallest loss (0.21).")
CAP_FIG9 = ("<b>Figure 9.</b> Acoustic distance and noise-induced confusion. Seven of 18 "
            "model-noise tests were significant after BH correction; AST under human "
            "non-speech noise had the strongest association.")

EQ_CAPTIONS = [
    ("eq7.png", "Eq.7 Macro-F1 — primary evaluation metric; 12-instrument average"),
    ("eq8.png", "Eq.8 Retention — standardized measure of robustness; how much clean "
     "score was retained after noise added"),
    ("eq9.png", "Eq.9 Area Under Robustness Curve — dB-weighted average retention across "
     "all eight noise levels"),
    ("eq10.png", "Eq.10 Accuracy — plain model correctness rate"),
]


# ---- filler blocks ---------------------------------------------------------------------------
def dataset_box(c: canvas.Canvas, st: dict, x: float, y: float, w: float) -> float:
    """Stat grid for the dataset and its pitch-grouped split -- methods the board never stated,
    and the first thing a skeptical reader asks about a 0.99 clean score."""
    inner = 16.0
    stat_rows = []
    for r in range(2):
        cells = []
        for value, label in DATASET_STATS[r * 3:(r + 1) * 3]:
            cells.append(Paragraph(
                f'<para alignment="center"><font name="TNR-Bold" size="26" '
                f'color="#CC0000">{value}</font><br/><font size="14" color="#555555">'
                f'{label}</font></para>', st["body"]))
        stat_rows.append(cells)
    note = Paragraph(DATASET_NOTE, ParagraphStyle("dnote", parent=st["caption_l"],
                                                  fontSize=14, leading=17.5))
    grid = Table(stat_rows, colWidths=[(w - 2 * inner) / 3] * 3, rowHeights=[62, 62])
    grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    _, gh = grid.wrapOn(c, w - 2 * inner, PAGE_H)
    _, nh = note.wrapOn(c, w - 2 * inner, PAGE_H)
    head_h = 34.0
    box_h = inner + head_h + gh + 10 + nh + inner
    c.setFillColor(HexColor("#f3f6fa"))
    c.setStrokeColor(CARD_EDGE)
    c.setLineWidth(1.2)
    c.roundRect(x, PAGE_H - y - box_h, w, box_h, 12, stroke=1, fill=1)
    c.setFillColor(NAVY)
    c.setFont("TNR-Bold", 22)
    c.drawCentredString(x + w / 2, PAGE_H - y - inner - 20, "Dataset & Split")
    grid.drawOn(c, x + inner, PAGE_H - (y + inner + head_h) - gh)
    note.drawOn(c, x + inner, PAGE_H - (y + inner + head_h + gh + 10) - nh)
    return y + box_h


def result_highlights_box(c: canvas.Canvas, st: dict, x: float, y: float,
                          w: float) -> float:
    """Full-width results summary with height determined by its wrapped text."""
    box_style = ParagraphStyle("result_box", parent=st["bullet"], fontSize=17,
                               leading=21.5, leftIndent=15, firstLineIndent=-10)
    paras = []
    for item in RESULT_HIGHLIGHTS:
        para = Paragraph(f'•&nbsp;&nbsp;{item}', box_style)
        _, ph = para.wrapOn(c, w - 40, PAGE_H)
        paras.append((para, ph))
    h = 26 + sum(ph + 8 for _, ph in paras)
    c.setFillColor(HexColor("#f3f6fa"))
    c.setStrokeColor(CARD_EDGE)
    c.setLineWidth(1.2)
    c.roundRect(x, PAGE_H - y - h, w, h, 12, stroke=1, fill=1)
    yy = y + 14
    for para, ph in paras:
        para.drawOn(c, x + 20, PAGE_H - yy - ph)
        yy += ph + 8
    return y + h


# ---- assembly --------------------------------------------------------------------------------
def build() -> None:
    register_fonts()
    st = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H))
    c.setTitle(TITLE)

    c.setFillColor(PAGE_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    # ---- header ----
    margin = 45.0
    draw_image(c, ASSETS / "bu_logo.jpeg", margin + 6, 52, LOGO_W)
    # Full-width box, so the title centres on the PAGE and lines up with the authors and
    # affiliations beneath it. It clears the logo by ~49pt only because LOGO_W was cut to suit
    # the 50pt title -- see the note on the title style. Asserted rather than trusted:
    title_w = stringWidth(TITLE, "TNR-Bold", st["title"].fontSize)
    assert (PAGE_W - title_w) / 2 - (margin + 6 + LOGO_W) > 30, (
        "the centred title collides with the BU logo: shrink LOGO_W or the title font size")
    y = draw_para(c, TITLE, st["title"], margin, 64, PAGE_W - 2 * margin) + 20
    y = draw_para(c, AUTHORS, st["authors"], margin, y, PAGE_W - 2 * margin) + 12
    y = draw_para(c, AFFIL, st["affil"], margin, y, PAGE_W - 2 * margin) + 22
    c.setStrokeColor(BU_RED)
    c.setLineWidth(5)
    c.line(margin, PAGE_H - y, PAGE_W - margin, PAGE_H - y)
    top = y + 40

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
    y = fig1_block(c, st, x, y, w) + 6
    y = draw_para(c, "<b>Fig.1</b> Flute A6 mel spectrogram; clean and under Gaussian noise",
                  st["caption_l"], x, y, w) + 18
    y = draw_para(c, "Objectives", st["subhead"], x, y, w) + 10
    y = bullets(c, OBJECTIVES, st["bullet"], x, y, w) + 20

    y = section_bar(c, "Methods", x, y, w) + 18
    y = pipeline_chart(c, x, y, w) + 6
    y = draw_para(c, "<b>Fig.2</b> Methods pipeline", st["caption_l"], x, y, w) + 14
    y = bullets(c, METHODS, st["bullet"], x, y, w) + 16
    y = dataset_box(c, st, x, y, w) + 6
    y = draw_para(c, "<b>Fig.3</b> Dataset and grouped split", st["caption_l"],
                  x, y, w) + 8
    y = model_table(c, st, x, y, w, "Trained from scratch",
                    "No external pretraining", SCRATCH_ROWS) + 6
    y = draw_para(c, "<b>Fig.4a</b> Models trained from scratch", st["caption_l"],
                  x, y, w) + 8
    y = model_table(c, st, x, y, w, "Pretrained backbones",
                    "Pretraining corpus listed beneath each model", PRETRAINED_ROWS) + 6
    y = draw_para(c, "<b>Fig.4b</b> Models using pretrained backbones", st["caption_l"],
                  x, y, w) + 6

    y = draw_para(c, "Noise Construction", st["subhead"], x, y, w) + 10
    y = bullets(c, NOISE_CONSTRUCTION, st["bullet"], x, y, w) + 12
    y = draw_image(c, ASSETS / "eq_block.png", x, y, w * 0.92, center_in=w) + 14
    tag = ParagraphStyle("tag", parent=st["bullet"], fontSize=17, leading=21.5)
    for name, desc in NOISE_SOURCES:
        y = draw_para(c, f'●&nbsp;&nbsp;<b><font color="#CC0000">{name}</font></b> '
                         f'— {desc}', tag, x, y, w) + 5
    y = draw_para(c, SEED_NOTE, ParagraphStyle("seed", parent=st["caption_l"], fontSize=14.5,
                                               leading=18), x, y, w)
    left_end = y

    # ================= MIDDLE =================
    x, w = x_m + pad, w_mid - 2 * pad
    y = top + pad
    y = section_bar(c, "Results", x, y, w) + 22
    y = draw_results_table(c, x, y, w) + 12
    y = draw_para(c, CAP_FIG5, st["caption"], x, y, w) + 32
    y = draw_image(c, FIGURES / "fig6_robustness_curves.png", x, y,
                   w * 0.98, center_in=w) + 10
    y = draw_para(c, CAP_FIG6, st["caption"], x, y, w) + 32
    y = draw_image(c, FIGURES / "fig8_confusion_grid.png", x, y, w * 0.96,
                   center_in=w) + 10
    y = draw_para(c, CAP_FIG7, st["caption"], x, y, w) + 32
    y = draw_image(c, FIGURES / "fig9_rank_slope.png", x, y,
                   w * 0.98, center_in=w) + 10
    y = draw_para(c, CAP_FIG8, st["caption"], x, y, w) + 32
    y = draw_image(c, FIGURES / "fig_distance_confusion.png", x, y,
                   w * 0.96, center_in=w) + 10
    y = draw_para(c, CAP_FIG9, st["caption"], x, y, w) + 24
    mid_end = result_highlights_box(c, st, x, y, w)

    # ================= RIGHT =================
    x, w = x_r + pad, w_side - 2 * pad
    y = top + pad
    bullet_r = ParagraphStyle("bullet_r", parent=st["bullet"], fontSize=20.5, leading=26.5)
    y = section_bar(c, "Discussion", x, y, w) + 18
    # Evaluation metrics BEFORE the statistics that use them (reordered on request 2026-08-04).
    # All four equations share ONE scale factor: they were rendered at the same font size, so
    # equal pixels-per-point everywhere means equal glyph size everywhere. Placing each by
    # column width instead made wide formulas shrink their maths relative to short ones.
    y = draw_para(c, "Evaluation Metrics", st["subhead"], x, y, w) + 12
    eq_w = (w - 26) / 2
    dims = {}
    for fname, _cap in EQ_CAPTIONS:
        with PILImage.open(ASSETS / fname) as im:
            dims[fname] = im.size
    k = min((eq_w * 0.92) / iw for iw, _ih in dims.values())
    eq_caption = ParagraphStyle("eqcap", parent=st["caption_l"], fontSize=13.5,
                                leading=16.5, alignment=1)
    for row in range(2):
        row_items = EQ_CAPTIONS[row * 2:(row + 1) * 2]
        heights = [dims[fname][1] * k for fname, _cap in row_items]
        row_height = max(heights)
        caption_ends = []
        for col, ((fname, cap), image_height) in enumerate(zip(row_items, heights)):
            ex = x + col * (eq_w + 26)
            # Bottom-align equations within each row. Equations 8 and 10 are shorter images,
            # so this intentionally moves them downward to line up with Equations 7 and 9.
            image_top = y + row_height - image_height
            iw, _ih = dims[fname]
            draw_image(c, ASSETS / fname, ex, image_top, iw * k, center_in=eq_w)
            caption_ends.append(draw_para(c, cap, eq_caption, ex, y + row_height + 6, eq_w))
        y = max(caption_ends) + 14
    y += 24
    y = draw_para(c, "Statistical Analysis", st["subhead"], x, y, w) + 10
    small = ParagraphStyle("stat", parent=st["bullet"], fontSize=18.5, leading=24)
    y = bullets(c, STATS + MID_STATS, small, x, y, w) + 26
    y = draw_para(c, "Key Takeaways", st["subhead"], x, y, w) + 10
    y = bullets(c, TAKEAWAYS, bullet_r, x, y, w) + 26
    y = draw_para(c, "Conclusion", st["subhead"], x, y, w) + 10
    y = bullets(c, CONCLUSION, bullet_r, x, y, w) + 26
    y = draw_para(c, "Limitations", st["subhead"], x, y, w) + 10
    y = bullets(c, LIMITATIONS, bullet_r, x, y, w) + 26
    y = draw_para(c, "Future Work", st["subhead"], x, y, w) + 10
    y = bullets(c, FUTURE, bullet_r, x, y, w) + 26
    y = draw_para(c, "Reproducibility", st["subhead"], x, y, w) + 10
    y = bullets(c, REPRO, bullet_r, x, y, w)
    right_flow_end = y

    # References and Acknowledgments anchor to the BOTTOM of the column, so leftover space
    # becomes breathing room in the middle rather than a ragged hole at the foot of the board.
    ack_style = ParagraphStyle("ack", parent=st["body"], fontSize=16, leading=21)
    ack_p = Paragraph(ACK, ack_style)
    _, ack_h = ack_p.wrapOn(c, w, PAGE_H)
    bottom_limit = top + col_h - pad
    qr_w = 215.0
    ref_block = 58 + 16 + qr_w + 34 + 26
    ack_block = 58 + 14 + ack_h
    y = bottom_limit - ack_block - ref_block - 14
    anchor_start = y
    y = section_bar(c, "References", x, y, w) + 16
    qx = x + (w - qr_w) / 2
    y2 = draw_image(c, ASSETS / "qr_works_cited.jpeg", qx, y, qr_w)
    c.setFont("TNR-Bold", 18)
    c.setFillColor(INK)
    c.drawCentredString(x + w / 2, PAGE_H - y2 - 24, "Works Cited")
    y = y2 + 40
    y = section_bar(c, "Acknowledgments", x, y, w) + 14
    y = draw_para(c, ACK, ack_style, x, y, w)

    c.save()
    limit = top + col_h - pad
    for name, end, cap in (("left", left_end, limit), ("mid", mid_end, limit),
                           ("right", right_flow_end, anchor_start)):
        over = end - cap
        state = f"OVERFLOWS by {over:.0f}pt" if over > 0 else f"{-over:.0f}pt spare"
        print(f"  {name:5} column: ends {end:.0f} / limit {cap:.0f}  ({state})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
