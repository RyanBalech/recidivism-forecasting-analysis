"""Build reports/ISAF_Recidivism_Presentation.pptx, following reports/presentation_outline.md.

Light theme, Calibri throughout, palette matched to the figures (navy / coral / teal).
Numbers are read from artifacts/ so the deck stays in sync with the pipeline; native tables and
one native chart (slide 8) keep the deck editable in PowerPoint.
"""
from pathlib import Path
import json
import re
import sys

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.parts.image import Image as PImage
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
ART, FIG, REPORTS = ROOT / "artifacts", ROOT / "artifacts" / "figures", ROOT / "reports"
# An optional path writes a copy instead, e.g. to try a slide without touching the team deck.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS / "ISAF_Recidivism_Presentation.pptx"

FONT = "Calibri"
INK, NAVY, CORAL, TEAL, TEAL_L, GOLD = "1B2B3A", "24506E", "D9573A", "1F7A70", "2A9D8F", "C99A2E"
MUTED, FAINT, LINE = "5F6E7C", "8795A2", "D5DDE5"
CARD, CARD_T, CARD_C, CARD_N, BG, BG_TITLE = "F1F5F8", "E6F2F0", "FBEBE6", "E7EEF4", "FFFFFF", "F3F7FA"

W, H = 13.333, 7.5
LM = 0.6              # left/right margin
CW = W - 2 * LM       # content width
SECTIONS = ["Intro", "Features", "Models", "Interpretability", "Fairness", "Stability", "Demo"]


def rgb(hex_):
    return RGBColor.from_string(hex_)


# ---------------------------------------------------------------- TEXT HELPERS
def _runs(p, text, size, color, bold=False, italic=False):
    """Add runs to paragraph p; **x** marks a bold segment."""
    for i, seg in enumerate(re.split(r"\*\*", text)):
        if not seg:
            continue
        r = p.add_run()
        r.text = seg
        f = r.font
        f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold or i % 2 == 1, italic
        f.color.rgb = rgb(color)


def _bullet(p, color):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(0.2))))
    pPr.set("indent", str(-int(Inches(0.2))))
    clr = pPr.makeelement(qn("a:buClr"), {})
    clr.append(clr.makeelement(qn("a:srgbClr"), {"val": color}))
    pPr.append(clr)
    pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
    pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))


def text(slide, x, y, w, h, paras, size=15, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, bullets=False, space=6, bullet_color=TEAL_L, line=None):
    """paras: a string, or a list of strings / (string, options) pairs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate([paras] if isinstance(paras, str) else paras):
        t, o = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        p.space_after = Pt(o.get("space", space))
        if line:
            p.line_spacing = line
        _runs(p, t, o.get("size", size), o.get("color", color), o.get("bold", bold), o.get("italic", italic))
        if o.get("bullet", bullets):
            _bullet(p, o.get("bcolor", bullet_color))
    return tb


def box(slide, x, y, w, h, fill=CARD, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def circle(slide, x, y, d, fill, label, size=16):
    s = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    _runs(tf.paragraphs[0], label, size, "FFFFFF", bold=True)


def arrow(slide, x, y):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(0.28), Inches(0.32))
    a.shadow.inherit = False
    a.fill.solid(); a.fill.fore_color.rgb = rgb("A3AFBA"); a.line.fill.background()


def picture(slide, name, x, y, w, h, align="center"):
    """Fit a figure inside the (x, y, w, h) box, keeping its aspect ratio."""
    path = str(FIG / name)
    pw, ph = PImage.from_file(path).size
    iw, ih = (h * pw / ph, h) if w / h > pw / ph else (w, w * ph / pw)
    ix = x + (w - iw) / 2 if align == "center" else x
    slide.shapes.add_picture(path, Inches(ix), Inches(y + (h - ih) / 2), Inches(iw), Inches(ih))
    return ix, iw


def stat(slide, x, y, w, h, big, label, color=NAVY, fill=CARD, big_size=34, label_size=12):
    """Big number with a small label, on a tinted card."""
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.2, y + 0.14, w - 0.4, 0.62, big, size=big_size, color=color, bold=True)
    text(slide, x + 0.2, y + 0.14 + big_size / 72 * 1.25, w - 0.4, 0.7, label, size=label_size, color=MUTED)


def card(slide, x, y, w, h, head, body, fill=CARD, head_color=NAVY, size=13, bullets=False, space=5):
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.22, y + 0.16, w - 0.44, 0.3, head.upper(), size=12, color=head_color, bold=True)
    text(slide, x + 0.22, y + 0.5, w - 0.44, h - 0.62, body, size=size, bullets=bullets,
         bullet_color=head_color, space=space)


def table(slide, x, y, rows, col_w, row_h=0.4, size=13, highlight=()):
    """Light table: navy header, zebra rows, rows in `highlight` tinted teal."""
    gf = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                Inches(sum(col_w)), Inches(row_h * len(rows)))
    tbl = gf.table
    tbl._tbl.tblPr.set("firstRow", "0"); tbl._tbl.tblPr.set("bandRow", "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(NAVY if i == 0 else CARD_T if i in highlight else BG if i % 2 else CARD)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            _runs(p, str(value), size, "FFFFFF" if i == 0 else INK, bold=i == 0 or j == 0)


def cards3(slide, y, h, items, size=13.5, gap=0.3):
    """Three equal cards in a row: items = [(head, head_color, fill, body), ...]."""
    w = (CW - 2 * gap) / 3
    for i, (head, color, fill, body) in enumerate(items):
        card(slide, LM + i * (w + gap), y, w, h, head, body, fill=fill, head_color=color, size=size)


# ---------------------------------------------------------------- SLIDE CHROME
def new_slide(kicker, title, section, number, title_size=28):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    text(s, LM, 0.42, 6.2, 0.3, kicker.upper(), size=12, color=TEAL, bold=True)
    if section:  # section navigator, top right: the deck's wayfinding motif
        tb = s.shapes.add_textbox(Inches(6.4), Inches(0.4), Inches(W - LM - 6.4), Inches(0.3))
        tf = tb.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.RIGHT
        for i, name in enumerate(SECTIONS):
            _runs(p, name, 10.5, CORAL if name == section else "A3AFBA", bold=name == section)
            if i < len(SECTIONS) - 1:
                _runs(p, "   ·   ", 10.5, "C5CED6")
    text(s, LM, 0.78, CW, 0.75, title, size=title_size, bold=True)
    text(s, LM, 7.02, 7, 0.25, "Team 11  ·  Trustworthy Recidivism Forecasting", size=10, color=FAINT)
    text(s, W - LM - 1.5, 7.02, 1.5, 0.25, str(number), size=10, color=FAINT, align=PP_ALIGN.RIGHT)
    return s


def source(slide, txt, y=6.72):
    text(slide, LM, y, CW, 0.25, "Source: " + txt, size=9.5, color=FAINT)


# ---------------------------------------------------------------- INPUTS FROM artifacts/
metrics = pd.read_csv(ART / "model_metrics.csv").set_index("model")
inc_auc = pd.read_csv(ART / "incumbent_discrimination.csv").set_index("ranker")["roc_auc"]
econ = pd.read_csv(ART / "incumbent_economics.csv").set_index("ranker")
inc_econ = econ["assumed_net_value"]
baselines = pd.read_csv(ART / "validation_baselines.csv").set_index("model")
brier_skill, base_brier = baselines.brier_skill_vs_prevalence, baselines.loc["training_prevalence", "brier"]
inf = pd.read_csv(ART / "fairness_inference.csv")
top20 = inf[inf.rule == "top_20pct"]
fnr20 = top20[top20.metric == "fnr"].pivot_table(index="model", columns="attribute", values="gap")
race_equiv = top20[(top20.attribute == "Race") & top20.metric.isin(["selection_rate", "fnr", "fpr"])] \
    .groupby("model").equivalent_within_delta.all()
paired = pd.read_csv(ART / "paired_comparisons.csv").set_index(["model_a", "model_b", "metric"])
xgb_minus_logit = -paired.loc[("logistic", "xgboost", "roc_auc"), "difference_a_minus_b"]
calib = pd.read_csv(ART / "fairness_impossibility.csv").set_index(["attribute", "group"])
mitigation = pd.read_csv(ART / "fairness_mitigation.csv")
drop_gang = mitigation[(mitigation.attribute == "Gender") & (mitigation.feature == "Gang_Affiliated")
                       & mitigation.panel.str.startswith("A")].set_index("model")
base_gender = mitigation[(mitigation.panel == "baseline") & (mitigation.attribute == "Gender")].set_index("model")
gang_cost = base_gender.loc["logistic", "auc"] - drop_gang.loc["logistic", "auc"]
by_group = pd.read_csv(ART / "fairness_by_group.csv").set_index(["model", "attribute", "group"])
race_auc = [by_group.loc[(m, "Race", g), "roc_auc"] for m in ["logistic", "xgboost", "tabicl"] for g in ["BLACK", "WHITE"]]

# Proxy recovery and per-person decision stability.
_proxy_path = ART / "proxy_recovery.csv"
if _proxy_path.exists():
    _proxy = pd.read_csv(_proxy_path)
    _get = lambda a, f: float(_proxy[(_proxy.attribute == a) & (_proxy.feature_set == f)].recovery_auc.iloc[0])
    leak_auc, gender_proxy_auc, race_proxy_auc = _get("Gender", "with_missingness"), _get("Gender", "shipped"), _get("Race", "shipped")
else:
    leak_auc = gender_proxy_auc = race_proxy_auc = float("nan")

_indiv_path = ART / "individual_stability_summary.csv"
if _indiv_path.exists():
    _indiv = pd.read_csv(_indiv_path).set_index("model")
    contested_share = float(_indiv.share_contested.mean())
    contested_of_selected = float(_indiv.contested_share_of_selected.mean())
else:
    contested_share = contested_of_selected = float("nan")

# Tested comparison: calibration, captured events and the selected-set overlap.
_cal_path = ART / "calibration_paired_tests.csv"
if _cal_path.exists():
    _cal = pd.read_csv(_cal_path)
    _cap = _cal[(_cal.model_a == "logistic") & (_cal.model_b == "xgboost")
                & (_cal.metric == "captured_events")].iloc[0]
    cap_ci = (_cap.ci_low, _cap.ci_high)
    overlap_share = float(pd.read_csv(ART / "selected_set_overlap.csv")
                          .query("model_a == 'logistic' and model_b == 'xgboost'").jaccard.iloc[0])
else:
    cap_ci, overlap_share = (float("nan"), float("nan")), float("nan")

# Out-of-fold mitigation evidence: selection inside training folds, assessment on held-out folds.
_nested_path = ART / "mitigation_nested_summary.csv"
if _nested_path.exists():
    _nested = pd.read_csv(_nested_path).set_index("model")
    nested_sel = float(_nested.selection_agreement.mean())
    nested_base = float(_nested.baseline_fnr_gap.mean())
    nested_mit = float(_nested.mitigated_fnr_gap.mean())
    nested_auc = float(_nested.auc_change.mean())
else:  # keep the deck buildable before the nested run exists
    nested_sel, nested_base, nested_mit, nested_auc = float("nan"), float("nan"), float("nan"), float("nan")


def span(values):
    """Range across the three models, e.g. '-0.10 to -0.12'."""
    lo, hi = sorted(values, key=abs)[0], sorted(values, key=abs)[-1]
    return f"{lo:+.2f} to {hi:+.2f}".replace("+-", "-")
pred = pd.read_csv(ART / "test_predictions.csv")
interp = json.loads((ART / "interpretability_summary.json").read_text())
surrogate_r2 = float(interp["surrogate_fidelity_r2_test"])
_agree = pd.read_csv(ART / "explanation_agreement.csv")
agree_lo, agree_hi = _agree.spearman_rank_correlation.min(), _agree.spearman_rank_correlation.max()
stability = pd.read_csv(ART / "stability_summary.csv").set_index("model")

# ---------------------------------------------------------------- EXTRA INPUTS FOR THE 19-SLIDE OUTLINE
# Structure follows reports/presentation_outline.md: six speaking parts, 19 core slides.
_pc = pd.read_csv(ART / "deep_review" / "INELIGIBLE_timing_positive_control.csv").set_index("condition")
posctrl_base, posctrl_leaky = _pc.loc["eligible_baseline", "roc_auc"], _pc.loc["INELIGIBLE_post_release_control", "roc_auc"]
_mlc = pd.read_csv(ART / "ml_model_comparison.csv")
_others = _mlc[~_mlc.model.isin(["logistic", "xgboost"])]
ml_lo, ml_hi, ml_n = _others.test_roc_auc.min(), _others.test_roc_auc.max(), len(_others)
_con = pd.read_csv(ART / "probability_contrasts.csv")
_con = _con[_con.model == "logistic"].set_index(["feature", "level"]).average_probability_contrast
age_contrast, gang_contrast = _con[("Age_at_Release", "48 or older")], _con[("Gang_Affiliated", "Yes")]
_pdpc = pd.read_csv(ART / "pdp_contrasts.csv").set_index(["contrast", "model"]).pdp_difference
pdp_age = [_pdpc[("Age 23-27 -> 48 or older", m)] * 100 for m in ["logistic", "xgboost", "tabicl"]]
pdp_gang = [_pdpc[("Gang No -> Yes", m)] * 100 for m in ["logistic", "xgboost", "tabicl"]]
_lime = pd.read_csv(ART / "lime_fidelity.csv").groupby("model").local_r2.mean()
lime_lo, lime_hi = _lime.min(), _lime.max()
_cal = pd.read_csv(ART / "calibration_tests.csv").set_index("model")
_sel = top20[top20.metric == "selection_rate"].set_index(["model", "attribute"])
race_ratio = [_sel.loc[(m, "Race"), "rate_b"] / _sel.loc[(m, "Race"), "rate_a"] for m in ["logistic", "xgboost", "tabicl"]]
gender_ratio = [_sel.loc[(m, "Gender"), "rate_b"] / _sel.loc[(m, "Gender"), "rate_a"] for m in ["logistic", "xgboost", "tabicl"]]
_ab = pd.read_csv(ART / "race_ab_test.csv").pivot_table(index="model", columns="variant", values="roc_auc")
race_ab_max = float((_ab["A_with_race"] - _ab["B_without_race"]).abs().max())
_curve = pd.read_csv(ART / "abstention_curve.csv")
_lc = _curve[_curve.model == "logistic"]
abst_full, abst_strict = _lc.loc[_lc.coverage.idxmax()], _lc.loc[_lc.max_contested_votes.idxmin()]
_ages = pd.read_csv(ART / "fairness_age_bands.csv")
age_old_fnr = _ages[(_ages.model == "logistic") & (_ages.age_band == "48 or older")].fnr.iloc[0]
age_young_fnr = _ages[(_ages.model == "logistic") & (_ages.age_band == "18-22")].fnr.iloc[0]
m3 = ["logistic", "xgboost", "tabicl"]
_xper = pd.read_csv(ART / "xper_values.csv")
_bench = _xper[_xper.feature.str.startswith("benchmark")].set_index("model")
xper_bench = _bench.xper.mean()
xper_auc = _bench.sample_auc
xper_age = _xper[_xper.feature == "Age_at_Release"].set_index("model").xper


def slash(values, fmt="{:.3f}"):
    """Three model values as 'a / b / c' in logistic, XGBoost, TabICL order."""
    return " / ".join(fmt.format(v) for v in values)


# Training re-arrest rate, as in the EDA chart.
_raw = pd.read_csv(ROOT / "data" / "nij-challenge2021_full_dataset.csv", usecols=["Training_Sample", "Recidivism_Within_3years"])
train_rate = _raw.loc[_raw.Training_Sample == 1, "Recidivism_Within_3years"].astype(str).eq("True").mean() if _raw.Recidivism_Within_3years.dtype == bool else _raw.loc[_raw.Training_Sample == 1, "Recidivism_Within_3years"].eq("Yes").mean()

m3 = ["logistic", "xgboost", "tabicl"]
_ci = paired.loc[("logistic", "xgboost", "roc_auc")]
xgb_ci = (-_ci.ci_high, -_ci.ci_low)
agree_sp = _agree[(_agree.method_a == "shap") & (_agree.method_b == "permutation_importance")].spearman_rank_correlation
agree_xp = _agree[_agree.method_b == "xper"].spearman_rank_correlation
top10_lo, top10_hi = _agree.top10_overlap.min(), _agree.top10_overlap.max()
_overlap = pd.read_csv(ART / "selected_set_overlap.csv").query("model_a == 'logistic' and model_b == 'xgboost'").iloc[0]
_nested_rows = pd.read_csv(ART / "mitigation_nested_summary.csv").set_index("model")
_indiv_l = pd.read_csv(ART / "individual_stability_summary.csv").set_index("model").loc["logistic"]
speedup = metrics.loc["xgboost", "fit_predict_seconds"] / metrics.loc["logistic", "fit_predict_seconds"]
pct = lambda v: f"{v * 100:+.0f}".replace("-", "−")
num = lambda v, fmt="{:.3f}": fmt.format(v).replace("-", "−")
WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
sl = lambda vals, fmt="{:.3f}": " / ".join(num(v, fmt) for v in vals)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)

# ================================================================ P1 · INTRO + EDA (1.5 min)
# 1 — Title
s = prs.slides.add_slide(prs.slide_layouts[6])
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG_TITLE)
text(s, LM + 0.1, 1.2, 7, 0.35, "TRUSTWORTHY AI  ·  TEAM 11", size=14, color=TEAL, bold=True)
text(s, LM + 0.1, 1.75, 7.2, 2.2, "Trustworthy\nRecidivism Forecasting", size=48, bold=True, line=0.95)
text(s, LM + 0.1, 3.95, 6.6, 1.2, "Which three-year re-arrest model should a risk-tool vendor ship? "
     "Judged on four dimensions, not accuracy alone.", size=18, color=MUTED)
text(s, LM + 0.1, 6.35, 6.5, 0.35, "NIJ Recidivism Forecasting Challenge  ·  Georgia 2013–2015", size=12, color=FAINT)
for k, (name, sub, col) in enumerate([("Performance", "AUC · Brier · $ value", NAVY),
                                      ("Interpretability", "SHAP · PDP/ICE · LIME · XPER", TEAL_L),
                                      ("Fairness", "Race · Gender · FNR", CORAL),
                                      ("Stability", "Bootstrap refits · flips", GOLD)]):
    cx, cy = 7.75 + (k % 2) * 2.6, 1.3 + (k // 2) * 2.5
    box(s, cx, cy, 2.35, 2.25, fill="FFFFFF", radius=0.08)
    circle(s, cx + 0.3, cy + 0.3, 0.55, col, str(k + 1))
    text(s, cx + 0.3, cy + 1.1, 1.85, 0.4, name, size=18, bold=True)
    text(s, cx + 0.3, cy + 1.5, 1.85, 0.6, sub, size=12, color=MUTED)

# 2 — Client and decision
s = new_slide("Intro · Client", "Our client, and the decision the score supports", "Intro", 2)
for i, (head, body) in enumerate([("Client", "Vendor of risk tools to US community-supervision agencies"),
                                  ("Rank", "Score every person at the start of supervision"),
                                  ("Prioritise", "Top 20% by predicted risk"),
                                  ("Offer", "Voluntary re-entry support: jobs, housing, treatment")]):
    x = LM + i * 3.14
    box(s, x, 1.85, 2.7, 1.95, fill=CARD_N if i < 3 else CARD_T)
    circle(s, x + 0.22, 2.05, 0.46, NAVY if i < 3 else TEAL, str(i + 1), size=14)
    text(s, x + 0.82, 2.1, 1.7, 0.4, head, size=17, bold=True)
    text(s, x + 0.22, 2.68, 2.26, 1.0, body, size=13.5)
    if i < 3:
        arrow(s, x + 2.78, 2.66)
card(s, LM, 4.15, 5.9, 1.75, "In scope", ["**Support allocation only.** Being selected means being offered help.",
                                          "The harm to avoid: people who need support but are not selected."],
     fill=CARD_T, head_color=TEAL, size=14, bullets=True)
card(s, LM + 6.23, 4.15, 5.9, 1.75, "Never in scope",
     ["Sanctions, detention or surveillance.",
      "The same score used for control would turn every conclusion here upside down."],
     fill=CARD_C, head_color=CORAL, size=14, bullets=True)
text(s, LM, 6.15, CW, 0.3, "Caveat: the target is **recorded re-arrest**, which is only a proxy for need.",
     size=13, color=MUTED, italic=True)

# 3 — Data and EDA
s = new_slide("Intro · Data", "Georgia parolees, official split, training-only EDA", "Intro", 3)
for y, big, label, col, size in [(1.8, "18,028 / 7,807", "training / evaluation records\n(official NIJ split, released 2013–2015)", NAVY, 26),
                                 (3.25, f"{train_rate:.1%}", "re-arrested within 3 years\n(training data)", CORAL, 34),
                                 (4.7, "12.3% · 12.9%", "missing: Gang_Affiliated ·\nPrison_Offense", TEAL, 26)]:
    text(s, LM, y, 3.0, 0.62, big, size=size, color=col, bold=True)
    text(s, LM, y + size / 72 * 1.25, 3.0, 0.7, label, size=12, color=MUTED)
picture(s, "eda_overview.png", 3.85, 1.7, 8.88, 3.35)
box(s, 3.85, 5.3, 8.88, 1.25)
text(s, 4.07, 5.42, 8.44, 1.05,
     [(f"Base rates: **Black ≈ White**; women **{calib.loc[('Gender', 'F'), 'base_rate']:.2f}** vs men "
       f"**{calib.loc[('Gender', 'M'), 'base_rate']:.2f}**.", {"bullet": True}),
      ("Target = new arrest within 3 years (cumulative). NIJ scored annual forecasts, so results are "
       "**not comparable** with the leaderboard.", {"bullet": True})], size=13.5, space=4)
text(s, LM, 6.62, 3.1, 0.3, "“The missingness is structured” → P2", size=12.5, color=TEAL, italic=True, bold=True)

# ================================================================ P2 · FEATURES + LEAKAGE (1.5 min)
# 4 — Only information known when supervision starts
s = new_slide("Features · Eligibility", "Only information known when supervision starts", "Features", 4)
w3 = (CW - 0.7) / 3
for i, (head, big, body, fill, col) in enumerate([
        ("In the model", "29", "baseline fields known at the start of supervision", CARD_T, TEAL),
        ("Audit only", "3", "race, gender and residence PUMA: excluded from the models, used to audit outcomes", CARD_N, NAVY),
        ("Excluded", "post-release", "employment, drug tests, violations… anything recorded after release", CARD_C, CORAL)]):
    x = LM + i * (w3 + 0.35)
    box(s, x, 1.85, w3, 2.55, fill=fill)
    text(s, x + 0.25, 2.02, w3 - 0.5, 0.3, head.upper(), size=12, color=col, bold=True)
    text(s, x + 0.25, 2.38, w3 - 0.5, 0.75, big, size=40 if len(big) < 5 else 32, color=col, bold=True)
    text(s, x + 0.25, 3.25, w3 - 0.5, 1.1, body, size=14)
box(s, LM, 4.7, CW, 1.6)
text(s, LM + 0.3, 4.88, 3.5, 0.3, "POSITIVE CONTROL", size=12, color=NAVY, bold=True)
text(s, LM + 0.3, 5.25, 4.4, 0.8, f"AUC {posctrl_base:.3f} → {posctrl_leaky:.3f}", size=32, color=CORAL, bold=True)
text(s, LM + 5.0, 4.7, 6.9, 1.6, [f"Adding the post-release fields back lifts AUC by {posctrl_leaky - posctrl_base:.2f}.",
                                  "They really do leak outcome information — which is **why they are excluded**."],
     size=15, bullets=True, bullet_color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
source(s, "src/recidivism/config.py · artifacts/deep_review/INELIGIBLE_timing_positive_control.csv", y=6.7)

# 5 — The leak
s = new_slide("Features · Leakage", "The leak we found: gender was encoded in missingness", "Features", 5)
cards3(s, 1.8, 1.85, [
    ("What happened", CORAL, CARD_C, "Gang_Affiliated is missing for **every woman** and **no man**. Keep missingness as a "
     f"feature → gender recovered at **AUC {leak_auc:.3f}**; TabICL treats NaN as a category, so it effectively saw gender."),
    ("The fix", TEAL, CARD_T, "Fill missing values with the mode, plus a **regression test** so the leak cannot return."),
    ("What remains", NAVY, CARD_N, f"After the fix, gender is **still recoverable at AUC {gender_proxy_auc:.3f}** from the "
     "other inputs. P5 picks this up.")])
picture(s, "proxy_recovery.png", LM, 3.85, CW, 2.8)
source(s, "artifacts/figures/proxy_recovery.png · artifacts/proxy_recovery.csv")

# ================================================================ P3 · MODELS + PERFORMANCE (2.5 min)
# 6 — Three models and tuning
s = new_slide("Models · Design", "Three models, each tuned by cross-validation", "Models", 6)
for i, (name, col, fill, enc, tune) in enumerate([
        ("L1 logistic regression", NAVY, CARD_N, "One-hot", "5-fold grid search\nC = 0.2154"),
        ("XGBoost", TEAL, CARD_T, "Ordinal encoding for count fields", "5-fold random search, 60 draws\nDeeper trees overfit → rejected"),
        ("TabICLv2", CORAL, CARD_C, "Mixed types, handled natively", "16 ensemble members\nResults level off at about 8")]):
    x = LM + i * (w3 + 0.35)
    box(s, x, 1.85, w3, 3.05, fill=fill)
    circle(s, x + 0.25, 2.05, 0.5, col, str(i + 1), size=15)
    text(s, x + 0.9, 2.1, w3 - 1.1, 0.45, name, size=18, bold=True)
    for dy, lab, val in [(0.95, "ENCODING", enc), (1.75, "TUNING", tune)]:
        text(s, x + 0.25, 1.85 + dy, w3 - 0.5, 0.25, lab, size=11, color=col, bold=True)
        text(s, x + 0.25, 2.12 + dy, w3 - 0.5, 0.9, val, size=14)
box(s, LM, 5.2, CW, 1.3)
text(s, LM + 0.3, 5.36, 3.3, 0.8, f"{ml_lo:.3f}–{ml_hi:.3f}", size=34, color=NAVY, bold=True)
text(s, LM + 3.8, 5.38, 8.0, 1.0, f"Test AUC of **{WORDS.get(ml_n, ml_n)} other ML candidates** — CatBoost, LightGBM, EBM, HistGB, random forest. "
     "Model family barely matters on these features.", size=15, anchor=MSO_ANCHOR.MIDDLE)
source(s, "artifacts/ml_model_comparison.csv · artifacts/figures/estimator_sweep.png", y=6.7)

# 7 — Performance
s = new_slide("Models · Performance", "Statistical performance: effectively a tie", "Models", 7)
table(s, LM, 1.85, [["Model", "ROC AUC", "Brier ↓", "ECE ↓", "Fit + predict"]] +
      [[n, f"{metrics.loc[m, 'roc_auc']:.4f}", f"{metrics.loc[m, 'brier']:.4f}", f"{metrics.loc[m, 'ece_10']:.4f}",
        f"{metrics.loc[m, 'fit_predict_seconds']:.1f} s"]
       for n, m in zip(["Logistic regression", "XGBoost", "TabICLv2"], m3)],
      [2.3, 1.2, 1.2, 1.2, 1.2], row_h=0.44, size=14)
sx, swd = 8.15, 4.58   # AUC quality scale (Desmarais & Singh 2013)
text(s, sx, 1.85, swd, 0.3, f"HOW GOOD IS {metrics.roc_auc.max():.2f}?", size=12, color=NAVY, bold=True)
lo, hi, by = 0.50, 0.80, 2.8
for name, a, b, col in [("Poor", .50, .55, "E4E8EC"), ("Fair", .55, .64, "F3D9A4"),
                        ("Good", .64, .71, "BFDCD6"), ("Excellent", .71, .80, "7CC2B7")]:
    bx, bw = sx + (a - lo) / (hi - lo) * swd, (b - a) / (hi - lo) * swd
    box(s, bx, by, bw, 0.42, fill=col, shape=MSO_SHAPE.RECTANGLE)
    text(s, bx, by + 0.08, bw, 0.28, name, size=10.5, bold=True, align=PP_ALIGN.CENTER)
for val, lab, col in [(inc_auc["incumbent"], f"Georgia score {inc_auc['incumbent']:.2f}", CORAL),
                      (metrics.roc_auc.max(), f"Our models {metrics.roc_auc.max():.2f}", TEAL)]:
    mx = sx + (val - lo) / (hi - lo) * swd
    m = s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(mx - 0.1), Inches(by - 0.24), Inches(0.2), Inches(0.18))
    m.rotation = 180; m.shadow.inherit = False
    m.fill.solid(); m.fill.fore_color.rgb = rgb(col); m.line.fill.background()
    text(s, mx - 1.0, by - 0.55, 2.0, 0.28, lab, size=11.5, color=col, bold=True, align=PP_ALIGN.CENTER)
text(s, sx, by + 0.55, swd, 0.55, "Bands from Desmarais & Singh (2013), CSG Justice Center, Table 2; anchored via "
     "Rice & Harris (2005).", size=9.5, color=MUTED)
for i, (head, col, fill, big, body) in enumerate([
        ("Real but small", NAVY, CARD_N, f"+{xgb_minus_logit:.4f}",
         f"XGBoost vs logistic AUC; paired 95% CI [{xgb_ci[0]:.4f}, {xgb_ci[1]:.4f}]."),
        ("Brier: NIJ's score", TEAL, CARD_T, f"≈{brier_skill[m3].min():.0%} better",
         f"than predicting the base rate for everyone ({base_brier:.3f}). Brier = mean (p − y)², lower is better."),
        ("Calibration", CORAL, CARD_C, " · ".join(f"{_cal.loc[m, 'slope']:.3f}" for m in m3),
         "Cox slopes: does 70% mean 70%? LR and XGB ≈ perfect; TabICL slightly too extreme.")]):
    x = LM + i * ((CW - 0.6) / 3 + 0.3)
    box(s, x, 4.1, (CW - 0.6) / 3, 2.15, fill=fill)
    text(s, x + 0.22, 4.25, 3.4, 0.3, head.upper(), size=12, color=col, bold=True)
    text(s, x + 0.22, 4.6, 3.4, 0.6, big, size=26, color=col, bold=True)
    text(s, x + 0.22, 5.2, 3.4, 1.0, body, size=13)
source(s, "paired_comparisons.csv · validation_baselines.csv · calibration_tests.csv", y=6.45)

# 8 — Incumbent, in dollars
SUPPORT_COST, EVENT_COST, EFFECT = 5_000, 50_000, 0.20
offers = int(econ.loc["logistic", "selected"])
break_even = lambda r: SUPPORT_COST / (econ.loc[r, "precision_at_capacity"] * EVENT_COST)
s = new_slide("Models · The client's real question", "Better than the current tool — in dollars?", "Models", 8)
box(s, LM, 1.85, 4.6, 4.7)
text(s, LM + 0.25, 2.0, 4.1, 0.3, "SCENARIO (ASSUMED, NOT ESTIMATED)", size=12, color=NAVY, bold=True)
for k, (big, label) in enumerate([(f"{offers:,}", "offers (20% capacity)"), (f"${SUPPORT_COST:,}", "support cost / person"),
                                  (f"${EVENT_COST:,}", "cost of a re-arrest"), (f"{EFFECT:.0%}", "re-arrests prevented")]):
    ax, ay = LM + 0.25 + (k % 2) * 2.1, 2.45 + (k // 2) * 1.0
    text(s, ax, ay, 2.0, 0.45, big, size=24, color=NAVY, bold=True)
    text(s, ax, ay + 0.45, 2.0, 0.35, label, size=11.5, color=MUTED)
text(s, LM + 0.25, 4.5, 4.1, 1.95,
     [(f"Net value = re-arrested among offers × ${EVENT_COST * EFFECT:,.0f} − offers × ${SUPPORT_COST:,}", {"bold": True}),
      f"Logistic: {econ.loc['logistic', 'captured_events']:,} × ${EVENT_COST * EFFECT / 1000:.0f}k − {offers:,} × "
      f"${SUPPORT_COST / 1000:.0f}k = **${inc_econ['logistic'] / 1e6:.3f}M**",
      ("Programme cost is the same for every ranking → only **precision** differs.", {"color": MUTED})], size=13, space=7)
_names = {"random": "Random", "incumbent": "Historical Georgia score", "logistic": "Logistic",
          "xgboost": "XGBoost", "tabicl": "TabICL"}
_order = ["random", "incumbent", "logistic", "xgboost", "tabicl"]
_auc = lambda r: inc_auc[r] if r in inc_auc else metrics.loc[r, "roc_auc"]
cd = CategoryChartData()
cd.categories = [f"{_names[r]}  (AUC {_auc(r):.2f} · prec. {econ.loc[r, 'precision_at_capacity']:.2f})"
                 for r in reversed(_order)]
cd.add_series("Net value ($M)", [inc_econ[r] / 1e6 for r in reversed(_order)])
ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(5.5), Inches(1.8), Inches(7.25), Inches(3.55), cd).chart
ch.has_legend, ch.has_title = False, True
ch.chart_title.text_frame.text = "Net value at 20% capacity ($M)"
for r in ch.chart_title.text_frame.paragraphs[0].runs:
    r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(13), True, FONT, rgb(INK)
ch.font.name, ch.font.size, ch.font.color.rgb = FONT, Pt(11), rgb(INK)
plot = ch.plots[0]
plot.gap_width, plot.vary_by_categories, plot.has_data_labels = 55, False, True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked, dl.position = '"$"0.000"M"', False, XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold, dl.font.name = Pt(11.5), True, FONT
for idx, col in enumerate(reversed(["B8C2CC", CORAL, NAVY, TEAL_L, "7FA7C2"])):
    pt = plot.series[0].points[idx]
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(col)
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 0, 6.2
va.major_gridlines.format.line.color.rgb = rgb("E3E8ED")
va.format.line.fill.background()
va.tick_labels.font.size, va.tick_labels.font.color.rgb = Pt(10), rgb(MUTED)
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = '"$"0"M"', False
ch.category_axis.format.line.color.rgb = rgb(LINE)
box(s, 5.5, 5.5, 7.23, 1.05, fill=CARD_T)
text(s, 5.72, 5.58, 1.9, 0.9, f"{break_even('logistic'):.0%} vs {break_even('incumbent'):.0%}", size=24, color=TEAL,
     bold=True, anchor=MSO_ANCHOR.MIDDLE)
text(s, 7.75, 5.58, 4.85, 0.9, "Break-even effect (= 10% / precision) with our ranking vs the historical score. "
     "Models win at every capacity 5–50% (A2).", size=12.5, anchor=MSO_ANCHOR.MIDDLE)
source(s, "incumbent_economics.csv · incumbent_capacity_sweep.csv · incumbent_effectiveness_sweep.csv. "
          "Dollar figures are scenarios, not causal estimates.", y=6.7)

# ================================================================ P4 · INTERPRETABILITY (2.7 min)
# 9 — Global drivers: each model read with the tool that suits it
_or = pd.read_csv(ART / "logistic_odds_ratios.csv").set_index("contrast").odds_ratio
_ice = pd.read_csv(ART / "ice_age_gang.csv", dtype={"value": str})
_fel = _ice[_ice.feature == "Prior_Arrest_Episodes_Felony"].groupby(["model", "value"]).p.mean()
pdp_felony = [(_fel[(m, "10 or more")] - _fel[(m, "1")]) * 100 for m in m3]
s = new_slide("Interpretability · Global", "Global drivers: each model, read with its own tool", "Interpretability", 9)
col_w, gap = (CW - 2 * 0.25) / 3, 0.25
for i, (name, tool, col, fill, fig) in enumerate([
        ("Logistic", "coefficients → odds ratios", NAVY, CARD_N, "logistic_odds_ratios.png"),
        ("XGBoost", "SHAP summary, all 7,807 people", TEAL, CARD_T, "shap_summary_xgboost.png"),
        ("TabICLv2", "PDP + ICE, 200 people", CORAL, CARD_C, "pdp_ice_tabicl.png")]):
    x = LM + i * (col_w + gap)
    box(s, x, 1.62, col_w, 0.42, fill=fill)
    text(s, x + 0.15, 1.62, col_w - 0.3, 0.42, f"**{name}** · {tool}", size=13, color=col, anchor=MSO_ANCHOR.MIDDLE)
    picture(s, fig, x, 2.1, col_w, 3.55)
box(s, LM, 5.78, CW, 0.82, fill=CARD)
text(s, LM + 0.25, 5.78, CW - 0.5, 0.82,
     f"**Three tools, one answer: age · gang · prior record.** Logistic odds: 18–22 vs 48+ ×{_or['18-22 vs 48+']:.1f}, "
     f"gang ×{_or['Gang affiliated: Yes vs No']:.1f}, 10+ felony arrests vs 1 ×{_or['10 or more vs 1']:.1f}. "
     f"PDP (logistic / XGBoost / TabICL): age 23–27 → 48+ {sl(pdp_age, '{:+.0f}')} pts; gang No → Yes {sl(pdp_gang, '{:+.0f}')} pts; "
     f"felony arrests 1 → 10+ {sl(pdp_felony, '{:+.0f}')} pts.",
     size=13, anchor=MSO_ANCHOR.MIDDLE)
source(s, "logistic_odds_ratios.csv · shap_summary_xgboost.png · ice_age_gang.csv · pdp_contrasts.csv. Logistic SHAP: A12 · all three PDP/ICE: A13", y=6.7)

# 10 — One person on the cut-off, each model with its own local tool
_lp = pd.read_csv(ART / "local_person.csv").set_index("model")
_ll = pd.read_csv(ART / "local_person_lime.csv")
_ll_top = _ll.loc[_ll.groupby(["model", "seed"]).weight.apply(lambda w: w.abs().idxmax())].condition.unique()
_ll_cmp = _ll[_ll.shap.notna()]  # TabICLv2 has no SHAP to compare signs with
s = new_slide("Interpretability · Local", "One person on the cut-off: why in, why out", "Interpretability", 10)
for i, (name, tool, col, fill, fig) in enumerate([
        ("Logistic", "coefficient × value (= SHAP)", NAVY, CARD_N, "local_logistic.png"),
        ("XGBoost", "TreeSHAP", TEAL, CARD_T, "local_xgboost.png"),
        ("TabICLv2", "this person's ICE", CORAL, CARD_C, "local_tabicl.png")]):
    x = LM + i * (col_w + gap)
    box(s, x, 1.62, col_w, 0.42, fill=fill)
    text(s, x + 0.15, 1.62, col_w - 0.3, 0.42, f"**{name}** · {tool}", size=13, color=col, anchor=MSO_ANCHOR.MIDDLE)
    picture(s, fig, x, 2.1, col_w, 3.55)
box(s, LM, 5.78, 8.1, 0.82, fill=CARD_C)
text(s, LM + 0.25, 5.78, 7.6, 0.82,
     f"**Age 28–32, gang Yes, 3 felony arrests: risk {slash(_lp.risk.loc[m3])} vs cut-off {slash(_lp.cutoff.loc[m3])}.** "
     f"Gang is the largest push in every model; set it to No and risk falls to {slash(_lp.risk_if_gang_no.loc[m3], '{:.2f}')}: "
     "no model would offer support.", size=12.5, anchor=MSO_ANCHOR.MIDDLE)
box(s, LM + 8.3, 5.78, CW - 8.3, 0.82, fill=CARD)
text(s, LM + 8.5, 5.78, CW - 8.7, 0.82,
     f"**LIME on the same person:** top reason {'gang = Yes' if list(_ll_top) == ['Gang_Affiliated=Yes'] else ', '.join(_ll_top)} "
     f"in all three models; {int((_ll_cmp.weight.gt(0) == _ll_cmp.shap.gt(0)).sum())}/{len(_ll_cmp)} conditions point the same way as SHAP. "
     f"Local R² only {_ll.local_r2.mean():.2f}: SHAP for size.", size=11.5, anchor=MSO_ANCHOR.MIDDLE)
source(s, "local_person.csv · local_person_contributions.csv · ice_age_gang.csv · local_person_lime.csv (3 models × 3 seeds). "
          "Contributions in log-odds, relative to the average person", y=6.7)

# 11 — Explaining performance: XPER and permutation importance
xper_felony = _xper[_xper.feature == "Prior_Arrest_Episodes_Felony"].set_index("model").xper
s = new_slide("Interpretability · Performance", "Explaining performance: which fields earn the accuracy", "Interpretability", 11)
picture(s, "performance_explanations.png", LM, 1.65, CW, 3.6)
cards3(s, 5.35, 1.3, [
    ("XPER (course)", NAVY, CARD_N, f"Splits the AUC itself. A model with no information scores **≈ {xper_bench:.2f}**; "
     f"age adds **+{xper_age.mean():.2f}**, prior felony arrests **+{xper_felony.mean():.2f}**."),
    ("Permutation importance", TEAL, CARD_T, "Shuffle one field, measure the loss. **Age first in all three models**, TabICL "
     "included; gang and prior felonies next."),
    ("Takeaway", CORAL, CARD_C, f"Same drivers, order varies (Spearman {agree_sp.min():.2f}–{agree_sp.max():.2f} SHAP vs "
     f"permutation, {agree_xp.min():.2f}–{agree_xp.max():.2f} vs XPER): **quote the set, not the rank.**")], size=12.5)
source(s, "xper_values.csv · permutation_importance.csv · explanation_agreement.csv. XPER not run on TabICLv2 and "
          "permutation on 10 of its 29 fields: each needs many full prediction passes on CPU", y=6.72)

# 12 — Interpretability verdict
_n_tab_perm = int((pd.read_csv(ART / "permutation_importance.csv").model == "tabicl").sum())
s = new_slide("Interpretability · Summary", "Verdict: logistic explains itself, TabICL only from outside", "Interpretability", 12)
EXACT, APPROX = CARD_T, "FBF1D9"
head_w, cell_w, gx, top = 2.05, (CW - 2.05 - 3 * 0.12) / 3, 0.12, 1.62
for j, (name, col) in enumerate([("Logistic", NAVY), ("XGBoost", TEAL), ("TabICLv2", CORAL)]):
    text(s, LM + head_w + gx + j * (cell_w + gx), top, cell_w, 0.36, name, size=15, color=col, bold=True, align=PP_ALIGN.CENTER)
grid = [
    ("Global · slide 9", [("Coefficients → odds ratios", EXACT), ("SHAP summary", EXACT), ("PDP / ICE, from outside", APPROX)]),
    ("Local · slide 10", [("Coefficient × value", EXACT), ("TreeSHAP", EXACT), ("ICE + LIME, approximate", APPROX)]),
    ("Performance · slide 11", [("XPER + permutation", EXACT), ("XPER + permutation", EXACT),
                                (f"Permutation only, {_n_tab_perm} of 29 fields", APPROX)]),
    ("Cost to explain", [("None: read the model", EXACT), ("Seconds", EXACT),
                         ("≈ 1 h per 200 ICE curves; SHAP, XPER out of reach", APPROX)]),
]
for i, (label, cells) in enumerate(grid):
    y = top + 0.44 + i * 0.6
    text(s, LM, y, head_w, 0.52, label, size=12.5, color=MUTED, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    for j, (body, fill) in enumerate(cells):
        x = LM + head_w + gx + j * (cell_w + gx)
        box(s, x, y, cell_w, 0.52, fill=fill)
        text(s, x + 0.12, y, cell_w - 0.24, 0.52, body, size=12.5, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
cards3(s, 4.8, 1.55, [
    ("All three agree", NAVY, CARD_N, "Age, gang affiliation and prior record lead **in every model and every method**."),
    ("Interpretability favours logistic", TEAL, CARD_T, "Only logistic answers **“why me?” exactly and for free**; "
     "TabICL can only be probed from outside, at a cost."),
    ("Hand-over to fairness", CORAL, CARD_C, "The person on the cut-off is in **because of gang affiliation**, a field never "
     "recorded for women. → Fairness")], size=13)
source(s, "Green = exact, amber = approximate or partial. Full method × model matrix: A10", y=6.72)

# ================================================================ P5 · FAIRNESS (3 min)
# 13 — Race (1)
s = new_slide("Fairness · Race", "Race (1): excluding race is not the same as being fair", "Fairness", 13)
text(s, LM, 1.62, CW, 0.35, "Primary metric: **FNR** — re-arrested but not selected for support, i.e. missed help. "
     "All results at the deployed top-20% rule.", size=13.5, color=MUTED)
box(s, LM, 2.15, 3.6, 2.75)
text(s, LM + 0.25, 2.3, 3.1, 0.3, "BASE RATES", size=12, color=NAVY, bold=True)
for dy, grp, lab, col in [(0.55, "BLACK", "Black", NAVY), (1.3, "WHITE", "White", TEAL_L)]:
    text(s, LM + 0.25, 2.15 + dy, 1.6, 0.6, f"{calib.loc[('Race', grp), 'base_rate']:.3f}", size=36, color=col, bold=True)
    text(s, LM + 1.75, 2.33 + dy, 1.6, 0.4, lab, size=14, color=MUTED)
text(s, LM + 0.25, 4.15, 3.1, 0.7, "Almost equal: the data does not force a gap. (Contrast: COMPAS / ProPublica 2016.)", size=12)
cw2 = (CW - 3.9 - 0.3) / 2
card(s, LM + 3.9, 2.15, cw2, 2.75, "What exclusion guarantees",
     ["Two people who differ only in race get **exactly the same score**.",
      f"Adding race back changes AUC by **≤ {race_ab_max:.3f}**."], fill=CARD_T, head_color=TEAL, size=14, bullets=True, space=8)
card(s, LM + 4.2 + cw2, 2.15, cw2, 2.75, "What it does not guarantee",
     [f"Race is still recoverable at **AUC {race_proxy_auc:.3f}** from the remaining features.",
      "Criminal-history variables carry part of the same information — and are the core of risk assessment."],
     fill=CARD_C, head_color=CORAL, size=14, bullets=True, space=8)
box(s, LM, 5.15, CW, 1.25, fill=CARD_N)
text(s, LM + 0.3, 5.15, CW - 0.6, 1.25, "“Excluding race guarantees the model never uses it directly. It cannot guarantee "
     "equal outcomes. That is why we **audit outcomes, not inputs**.”", size=16, color=NAVY, italic=True, anchor=MSO_ANCHOR.MIDDLE)
source(s, "race_ab_test.csv · proxy_recovery.csv · fairness_impossibility.csv")


def results(slide, head, rows):
    """Right-hand column of three big numbers with labels (slides 14 and 15)."""
    text(slide, 7.2, 1.75, 5.5, 0.3, head, size=12, color=NAVY, bold=True)
    for i, (big, label, col) in enumerate(rows):
        text(slide, 7.2, 2.12 + i * 0.88, 5.5, 0.4, big, size=19, color=col, bold=True)
        text(slide, 7.2, 2.52 + i * 0.88, 5.5, 0.45, label, size=12, color=MUTED)


# 14 — Race (2)
s = new_slide("Fairness · Race", "Race (2): the outcome audit, with three caveats", "Fairness", 14)
picture(s, "fairness_race_gaps.png", LM, 1.7, 6.3, 3.06, align="left")
results(s, "RESULT  (LOGISTIC / XGBOOST / TABICL)", [
    (sl(fnr20.loc[m3, "Race"]), "FNR gap, Black − White", NAVY),
    ("within ±5 pts", "every gap, by TOST equivalence; none significant after Holm", TEAL),
    (sl(race_ratio, "{:.2f}"), "selection-rate ratio White : Black — above the four-fifths rule", NAVY)])
for i, (head, body) in enumerate([
        ("Direction is reversed", "XGBoost and TabICL select ~2 pts **more** Black people — here that means more help. "
         "Used for sanctions, the same gap would be harm."),
        ("The label may be biased", "We measure **recorded arrest**, not reoffending. Differences in policing intensity "
         "cannot be seen in this data."),
        ("Prediction quality differs", f"Within-group AUC **{min(race_auc[0::2]):.3f}–{max(race_auc[0::2]):.3f}** for Black "
         f"vs **{min(race_auc[1::2]):.3f}–{max(race_auc[1::2]):.3f}** for White people.")]):
    x = LM + i * ((CW - 0.6) / 3 + 0.3)
    box(s, x, 4.95, (CW - 0.6) / 3, 1.65, fill=CARD_C)
    circle(s, x + 0.2, 5.1, 0.38, CORAL, str(i + 1), size=12)
    text(s, x + 0.7, 5.1, 3.0, 0.38, head, size=14, color=CORAL, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.2, 5.55, 3.44, 1.05, body, size=12)
source(s, "fairness_inference.csv · fairness_tests_holm.csv · fairness_by_group.csv")

# 15 — Gender (1)
s = new_slide("Fairness · Gender", "Gender (1): women are over-predicted, yet selected less", "Fairness", 15)
picture(s, "fairness_gender_gaps.png", LM, 1.7, 6.3, 3.06, align="left")
results(s, "LOGISTIC / XGBOOST / TABICL", [
    (sl(fnr20.loc[m3, "Gender"]), "FNR gap, men − women: shared by all three models (feature set)", CORAL),
    (sl(gender_ratio, "{:.2f}"), "selection-rate ratio women : men — about half", NAVY),
    (f"{calib.loc[('Gender', 'F'), 'mean_score_logistic']:.2f} vs {calib.loc[('Gender', 'F'), 'base_rate']:.3f}",
     "women's mean predicted score vs observed re-arrest rate", TEAL)])
card(s, LM, 4.95, 5.95, 1.65, "Impossibility result",
     f"Base rates differ by ~{(calib.loc[('Gender', 'M'), 'base_rate'] - calib.loc[('Gender', 'F'), 'base_rate']) * 100:.0f} pts "
     f"(men **{calib.loc[('Gender', 'M'), 'base_rate']:.3f}**, women **{calib.loc[('Gender', 'F'), 'base_rate']:.3f}**): "
     "calibration and equal error rates **cannot both hold**.", fill=CARD_N, head_color=NAVY, size=13.5)
card(s, LM + 6.18, 4.95, 5.95, 1.65, "Over-prediction helps women here",
     "Recalibrating by gender would lower their scores, select fewer women and **widen** the FNR gap. "
     "(Cf. State v. Loomis.)", fill=CARD_C, head_color=CORAL, size=13.5)
source(s, "fairness_impossibility.csv · fairness_inference.csv")

# 16 — Gender (2)
s = new_slide("Fairness · Gender", "Gender (2): cause found, mitigated, validated out of sample", "Fairness", 16)
picture(s, "fpdp_gender_focus.png", LM, 1.7, 8.2, 2.6, align="left")
card(s, 9.1, 1.75, 3.63, 2.5, "FPDP → Gang_Affiliated", "Never recorded for women; raw field has **Cramér's V = 1.0** "
     "with gender. It reflects record-keeping, not behaviour.", fill=CARD_C, head_color=CORAL, size=13)
text(s, LM, 4.5, 6.5, 0.3, "MITIGATION: DROP THE FIELD, REFIT  ·  5-FOLD NESTED, OUT-OF-FOLD", size=11.5, color=NAVY, bold=True)
table(s, LM, 4.85, [["Model", "Gender FNR gap (before → after)", "AUC cost"]] +
      [[n, f"{num(_nested_rows.loc[m, 'baseline_fnr_gap'])} → {num(_nested_rows.loc[m, 'mitigated_fnr_gap'])}",
        num(_nested_rows.loc[m, "auc_change"])] for n, m in [("Logistic", "logistic"), ("XGBoost", "xgboost")]],
      [1.5, 3.5, 1.6], row_h=0.42, size=13.5)
stat(s, 7.45, 4.5, 2.35, 2.1, f"≈{round((1 - nested_mit / nested_base) * 20) / 20:.0%}",
     "of the gap closed;\nsame field chosen in " + ("every fold" if nested_sel == 1 else f"{nested_sel:.0%} of folds"), color=TEAL, fill=CARD_T, big_size=32)
card(s, 10.0, 4.5, 2.73, 2.1, "Costs · decision", ["Men's FNR +2 pts (partly levelling down).",
                                                   "Proposed as a **second pilot arm** (A11)."],
     head_color=NAVY, size=12, bullets=True, space=3)
source(s, "fpdp_gender.png · mitigation_nested_summary.csv · fairness_mitigation.csv")

# ================================================================ P6 · STABILITY + TRADE-OFFS + RECOMMENDATION (2.5 min)
# 17 — Structural stability
s = new_slide("Stability · Structural", "Structural stability: 8 bootstrap refits", "Stability", 17)
table(s, LM, 1.8, [["Model", "Mean |Δp| between refits", "Top-20% Jaccard"]] +
      [[n, f"{stability.loc[m, 'mean_abs_prob_diff']:.3f}", f"{stability.loc[m, 'top20_jaccard']:.2f}"]
       for n, m in zip(["Logistic", "XGBoost", "TabICLv2"], m3)], [1.8, 2.6, 2.0], size=13.5, highlight=(1,))
stat(s, 7.3, 1.8, 2.6, 1.62, "13–15%", "of selected people change between refits (not 23%)", color=NAVY, fill=CARD_N,
     big_size=30, label_size=11.5)
stat(s, 10.13, 1.8, 2.6, 1.62, "28 / 28", "refit pairs where logistic is more stable than XGBoost", color=TEAL,
     fill=CARD_T, big_size=30, label_size=11.5)
picture(s, "structural_stability.png", LM, 3.65, CW, 2.95)
source(s, "structural_stability.png · stability_summary.csv")

# 18 — Per-person stability and abstention
s = new_slide("Stability · Individual", "Stability for one person: abstaining is not fairness-neutral", "Stability", 18)
w4 = (CW - 0.75) / 4
for i, (big, label, col, fill) in enumerate([
        (f"≈{_indiv_l.share_contested:.0%}", f"of decisions flip between refits; {_indiv_l.contested_share_of_selected:.0%} of "
         "selected sit at that margin", NAVY, CARD_N),
        (f"{abst_full.precision_at_capacity:.3f} → {abst_strict.precision_at_capacity:.3f}",
         "precision when keeping only unanimous decisions (8/8)", TEAL, CARD_T),
        (f"{num(abst_full.fnr_gap_gender)} → {num(abst_strict.fnr_gap_gender)}", "gender FNR gap: abstention widens it",
         CORAL, CARD_C),
        ("47% vs 30%", "of selected women vs men sit at the margin", CORAL, CARD_C)]):
    stat(s, LM + i * (w4 + 0.25), 1.8, w4, 1.35, big, label, color=col, fill=fill, big_size=24)
picture(s, "individual_stability.png", LM, 3.35, CW, 3.25)
source(s, "individual_stability.png · individual_stability_summary.csv · abstention_curve.csv  ·  links back to P5 (gender gap)")

# 19 — Trade-off matrix and recommendation
s = new_slide("Conclusion · Trade-offs", "Pilot logistic regression; run XGBoost as the challenger", "Stability", 19)
picture(s, "tradeoff_matrix.png", LM, 1.65, 6.6, 4.95, align="left")
rx, rw = 7.35, W - LM - 7.35
box(s, rx, 1.7, rw, 2.95, fill=CARD_T)
text(s, rx + 0.25, 1.83, rw - 0.5, 0.3, "RECOMMENDATION", size=12, color=TEAL, bold=True)
text(s, rx + 0.25, 2.15, rw - 0.5, 0.65, "L1 logistic regression for a prospective **shadow pilot**; XGBoost as challenger.", size=15)
text(s, rx + 0.25, 2.88, rw - 0.5, 1.75, [
    f"Same people: Jaccard {overlap_share:.3f}, only {int(_overlap.chosen_by_only_one)} differ; captured re-arrest "
    "difference CI includes 0.",
    f"More stable, directly interpretable, ~{speedup:.0f}× faster.",
    "Calibration and fairness are tied — not a reason to choose.",
    "**Second arm without gang affiliation** (A11)."], size=12.5, space=3, bullets=True, bullet_color=TEAL)
card(s, rx, 4.8, rw, 1.0, "What would reverse it", "Scores quoted numerically to supervisees, or a scale where a few "
     "extra captured events matter.", fill=CARD_C, head_color=CORAL, size=12)
text(s, rx, 5.92, rw, 0.7, "**Limits:** evaluation set inspected repeatedly; no temporal or external validation; no "
     "evidence yet that the support programme helps.", size=11.5, color=MUTED)

# 20 — App
s = new_slide("Demo · App", "The Streamlit app makes every trade-off testable", "Demo", 20)
for i, (head, body) in enumerate([("Scores", "One person's score from all three models"),
                                  ("Explanation", "Why this score: SHAP / LIME for that person"),
                                  ("Stability", "How many of the 8 refits select them"),
                                  ("What-if", "Edit an input and watch the score change")]):
    x = LM + i * 3.14
    box(s, x, 2.2, 2.7, 3.0, fill=[CARD_N, CARD_T, CARD, CARD_C][i])
    circle(s, x + 0.3, 2.5, 0.8, [NAVY, TEAL, GOLD, CORAL][i], str(i + 1), size=24)
    text(s, x + 0.3, 3.55, 2.1, 0.45, head, size=20, bold=True)
    text(s, x + 0.3, 4.05, 2.1, 1.0, body, size=14)
    if i < 3:
        arrow(s, x + 2.78, 3.55)
box(s, LM, 5.55, CW, 0.9)
text(s, LM + 0.3, 5.55, CW - 0.6, 0.9, "Backup: screenshots and a screen recording are ready if the live app fails.   ·   "
     "Run: streamlit run app.py", size=13.5, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- APPENDIX (Q&A only)
s = prs.slides.add_slide(prs.slide_layouts[6])
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG_TITLE)
text(s, LM + 0.1, 1.5, 6, 0.35, "APPENDIX", size=14, color=TEAL, bold=True)
text(s, LM + 0.1, 2.0, 7, 1.6, "Supporting evidence\nfor questions", size=44, bold=True, line=0.95)
text(s, LM + 0.1, 3.75, 6, 0.4, "Not presented — kept ready for Q&A.", size=16, color=MUTED)
for k, item in enumerate(["A1  Learning curve", "A2  Economic sensitivity", "A3  Explanation disagreement",
                          "A4  Global surrogate", "A5  Threshold frontier", "A6  Age", "A7  Process log", "A8  LIME",
                          "A9  Permutation importance", "A10  Method overview", "A11  No-gang pilot arm"]):
    text(s, 7.9 + (k // 6) * 2.6, 1.55 + (k % 6) * 0.62, 2.5, 0.5, item, size=14)

APPENDIX = [
    ("Learning curve", "Which model for which agency size?", "learning_curve.png",
     "Subsamples of one Georgia cohort test sample-size sensitivity; they do not validate transfer to other agencies."),
    ("Economic sensitivity", "Model value is a scenario, stress-tested", "incumbent_benchmark.png",
     f"At 20% capacity: logistic ${inc_econ['logistic'] / 1e6:.3f}M, XGBoost ${inc_econ['xgboost'] / 1e6:.3f}M, historical score "
     f"${inc_econ['incumbent'] / 1e6:.3f}M, random ${inc_econ['random'] / 1e6:.3f}M. Models beat the historical score at every "
     "capacity from 5% to 50%; a randomised pilot must estimate the real effect."),
    ("Explanation disagreement", "SHAP, permutation importance and XPER rank differently", "explanation_agreement.png",
     f"Spearman {agree_lo:.2f}–{agree_hi:.2f}; {top10_lo}–{top10_hi} of the top ten features shared. The set of drivers is "
     "robust, the order is method-dependent."),
    ("Global surrogate", "A depth-3 tree that mimics XGBoost", "global_surrogate.png",
     f"Test fidelity R² = {surrogate_r2:.2f}: readable, but it misses much of what XGBoost does. A surrogate can give an "
     "illusion of interpretability."),
    ("Threshold frontier", "Group thresholds: the fairness/utility frontier", "fairness_frontier.png",
     "Optimised on the evaluation set, so optimistic. Group-specific thresholds by race or gender are disparate treatment."),
    ("Age", "Age: the largest gap, and a policy choice", "fpdp_age_focus.png",
     f"FNR gap <33 − 33+: {span(fnr20['Age']).replace('-', '−')}, mostly following base rates. Re-arrested 48+: {age_old_fnr:.0%} not selected, "
     f"vs {age_young_fnr:.0%} at 18–22. Age is a validated, legally accepted risk factor and FPDP finds no candidate: it runs "
     "through criminal history. Rank by risk or by need is the client's policy choice."),
    ("Process", "How we got here: what we tried, where we landed", "improvement_journey.png",
     "Full log in docs/JOURNEY.md — every step measured, including the attempts we rejected."),
    ("LIME", "LIME on the slide-10 person, all three models", "lime_person.png",
     "Gang = Yes is the top reason in every model and seed; for logistic and XGBoost every condition has SHAP's sign. "
     "Local R² 0.23–0.25 here (0.33–0.49 on the highest-, median- and lowest-risk people): quote LIME for direction, SHAP for size."),
    ("Permutation importance", "Permutation importance: shuffle one field, measure the loss", "permutation_importance_readable.png",
     "Same drivers as SHAP and XPER. TabICLv2 was measured on 10 of 29 fields only: each shuffle needs a full "
     "foundation-model prediction pass."),
    ("Method overview", "Which interpretability method, for which model", "interpretability_matrix.png",
     "Method × model, grouped into global, local and performance explanations."),
    ("Pilot arm 2", "Dropping gang affiliation, re-evaluated on all four dimensions", "gang_variant_comparison.png",
     "Gender gap closes; stability and drivers barely change; race stays within ±5 pts but flips sign. "
     "Cost: AUC −0.014, 27 fewer re-arrests captured."),
    ("Logistic SHAP", "SHAP on logistic regression only redraws the coefficients", "shap_summary_logistic.png",
     "For a linear model SHAP = coefficient × (value − average): everyone in the same category gets the same "
     "contribution, hence vertical bars. The odds ratios on slide 9 say the same thing more directly."),
    ("PDP/ICE", "Age, gang and prior felony arrests: PDP/ICE for all three models", "pdp_ice_age_gang_felony.png",
     "Same 200 evaluation people for every model. Near-identical average effects: the three models learned the "
     "same shape, not only the same ranking."),
]
for k, (kicker, title, fig, caption) in enumerate(APPENDIX, start=1):
    s = new_slide(f"A{k} · {kicker}", title, None, f"A{k}")
    pw, ph = PImage.from_file(str(FIG / fig)).size
    if pw / ph < 2.2:  # tall figure: caption beside it
        ix, iw = picture(s, fig, LM, 1.7, 8.2, 5.0, align="left")
        cx = ix + iw + 0.4
        ch_ = 0.9 + 0.3 * (len(caption) // 45)
        box(s, cx, 1.8, W - LM - cx, ch_)
        text(s, cx + 0.25, 1.8, W - LM - cx - 0.5, ch_, caption, size=14, anchor=MSO_ANCHOR.MIDDLE)
    else:            # wide figure: caption below it
        picture(s, fig, LM, 1.7, CW, 4.1)
        box(s, LM, 5.95, CW, 0.8)
        text(s, LM + 0.25, 5.95, CW - 0.5, 0.8, caption, size=13, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- SPEAKER NOTES
# Timings follow reports/presentation_outline.md: 14:25 of talk across 20 core slides,
# leaving a buffer in the 15-minute slot. [P1]…[P6] marks who speaks.
NOTES = {
 1: """[P1 · 0:00 · 20s] Team 11. Our client sells risk-assessment tools to US community-supervision agencies. Their question: which three-year re-arrest model should they ship? We judged four dimensions — performance, interpretability, stability, fairness — not accuracy alone.""",
 2: """[P1 · 0:20 · 30s] The score prioritises VOLUNTARY re-entry support: employment, housing, treatment. Never sanctions or detention. Everything later depends on this: being selected means being offered help. Our target is recorded re-arrest, which is only a proxy for need.""",
 3: """[P1 · 0:50 · 40s] NIJ's own split: 18,028 training, 7,807 evaluation, Georgia 2013-2015. About 58% are re-arrested. Black and White men have almost the same rate; women are lower. On the right, missing values: the gang field is missing for EVERY woman and no man. The missingness is structured — over to [P2].""",
 4: """[P2 · 1:30 · 40s] We only use what is known when supervision starts: 29 fields. Race, gender and residence are never inputs; we keep them to audit who gets support. Everything recorded after release is excluded — and we proved why: adding those fields lifts AUC from 0.73 to 0.81, because they leak the outcome.""",
 5: """[P2 · 2:10 · 50s] The leak we found. Gang affiliation is missing for every woman. If missingness is kept, gender is recovered from our own inputs at AUC 1.00 — perfectly. TabICL treats a missing value as its own category, so it effectively saw gender. We fixed it with mode-filling and a regression test. But even after the fix, gender is still recoverable at 0.78: excluding an attribute is not removing it. [P5] picks this up.""",
 6: """[P3 · 3:00 · 40s] Three families, as the brief requires, each tuned by cross-validation on training data only. Logistic with L1, C chosen by grid search. XGBoost with a 60-draw random search; deeper trees overfit and were rejected. TabICL with 16 ensemble members; gains level off near 8. Five other ML models all land in the same narrow AUC band.""",
 7: """[P3 · 3:40 · 45s] Finding one: performance cannot pick the model. The three ROC curves lie on top of each other: AUC 0.730 to 0.733. Is 0.73 good? Reviews of US recidivism tools call an AUC above 0.71 "excellent"; the historical Georgia score, at 0.60, is only "fair". XGBoost beats logistic by 0.0025 on a paired bootstrap: real, but small. Brier, the score NIJ used for this challenge, agrees: about 16% better than predicting the base rate, for all three. It also checks calibration, whether a predicted 70% means 70%: logistic and XGBoost pass; TabICL is slightly overconfident.""",
 8: """[P3 · 4:25 · 65s] The client's real question: is this worth paying for, compared with the score they already have? The scenario: 20% capacity, 1,561 offers. Support costs $5,000 a person, a re-arrest costs $50,000, and we ASSUME support prevents 20% of re-arrests. So net value is the number of offered people later re-arrested, times $10,000, minus $7.8M of programme cost. The cost is identical for every ranking, so only precision matters: 82% of our offers reach someone later re-arrested, against 67% for the historical score. That turns $2.7M into $5.0M. Put differently: with our ranking the programme breaks even if it prevents 12% of re-arrests; with the historical score it needs 15%. The dollars are a scenario, not a causal estimate: the data cannot tell us whether support works. Over to [P4].""",
 9: """[P4 · 5:30 · 40s] What drives the score, and each model is read with the tool that suits it. Logistic regression is its own explanation: its coefficients become odds ratios - an 18-to-22-year-old has almost six times the odds of re-arrest of someone over 48, gang affiliation more than doubles them, and so do ten or more prior felony arrests compared with one. XGBoost has no coefficients, so we use SHAP on all 7,807 people: each dot is a person, red is a high value, right raises risk - age first, then gang, then prior record. TabICL is a foundation model with neither, so we look from outside with PDP and ICE: the black line is the average, each faint line one person, and almost every line falls with age and rises with gang. The bottom line puts the three on one scale: moving everyone from 23-27 to 48+ lowers predicted risk by 26 to 29 points, gang adds 16 to 17. Three tools, one answer.""",
 10: """[P4 · 6:10 · 40s] Now one person - the kind of case where an explanation matters most, because they sit right on the top-20% cut-off. Aged 28 to 32, gang affiliated, three prior felony arrests. Logistic puts them just below the line, XGBoost and TabICL just above. Why? Each model with its own tool again. For logistic, each bar is simply coefficient times how this person differs from the average - that is exactly SHAP for a linear model. For XGBoost, TreeSHAP. For TabICL, we move one field at a time for this person and watch the risk against the cut-off. All three give the same reason: gang affiliation is by far the largest push. Set it to No and the risk falls to about 0.57 - no model would offer support. That is the field we saw is never recorded for women. As a check we ran LIME on the same person, for all three models: every one puts gang first, and for logistic and XGBoost every condition points the same way as SHAP. But its local fit is low, so we quote SHAP for the size and LIME only for the direction.""",
 11: """[P4 · 6:50 · 40s] So far we explained predictions. Now we explain PERFORMANCE: which fields actually earn the model its accuracy. Left, XPER from the course: it splits the AUC itself. A model with no information would score about 0.47; age alone adds about 0.09, prior felony arrests about 0.04. Right, permutation importance: shuffle one field and see how much worse the model gets. Age comes first in all three models, including TabICL, then gang and prior felonies. The two methods, and SHAP, agree on which fields matter but not on the exact order, because they answer different questions. So we quote the set of drivers, not a ranking.""",
 12: """[P4 · 7:30 · 40s] To sum up interpretability. Across all three models and all the methods we used, the same drivers come out on top: age, gang affiliation and prior record. But explaining them is not equally easy. Logistic regression explains itself: its coefficients are the explanation, exact and free. XGBoost needs SHAP, which is exact for trees and takes seconds. TabICL can only be probed from outside, with PDP, ICE and LIME, and the exact tools - SHAP, XPER - are out of reach on our hardware. So if someone asks "why was I not offered support?", logistic gives the most direct answer. On this dimension, logistic wins. And remember the person on the cut-off: they were in because of gang affiliation - a field that is never recorded for women. That is where [P5] picks up.""",
 13: """[P5 · 8:10 · 45s] Fairness. Our primary metric is FNR: people later re-arrested but not offered support — the real harm here. Race first, the question everyone expects after COMPAS. Base rates are almost equal, so the data does not force a gap. Excluding race guarantees twins get the same score, but race is still recoverable at 0.71 through criminal history. So we audit outcomes, not inputs.""",
 14: """[P5 · 8:55 · 45s] The outcome audit: every race gap is within five points by an equivalence test, and none survives Holm correction. Three caveats: the small gap runs toward MORE support for Black people; the label is recorded arrest, which may carry policing bias; and prediction quality is lower for Black people, AUC 0.72 vs 0.75.""",
 15: """[P5 · 9:40 · 45s] Gender is where we found a problem. Re-arrested women miss support 10 to 12 points more often, in all three models — so it comes from the features. Every model over-predicts women, yet selects them less: with different base rates, calibration and equal FNR cannot both hold. Age shows an even larger gap, but by design — age is a validated risk factor — so it is the client's policy choice; details in the appendix.""",
 16: """[P5 · 10:25 · 45s] We located the cause with the course's FPDP: gang affiliation, never recorded for women. Dropping it, selected in every one of five training folds, closes about 70% of the gap out of sample for about one AUC point — at the cost of men's FNR rising two points. We then re-evaluated the fix on all four dimensions (appendix A11): the gender gap closes, stability and the main drivers barely move, race stays within tolerance, and it costs about 0.014 AUC. So we propose it as a second pilot arm rather than deciding it for the client. Over to [P6].""",
 17: """[P6 · 11:10 · 40s] Stability, in the course's sense: two samples from the same population should give the same model. Eight bootstrap refits, same resamples for all three models. Logistic drifts less than XGBoost on all 28 pairs. Jaccard around 0.77 means about 13% of the selected people change per refit.""",
 18: """[P6 · 11:50 · 50s] At the level of one person, about one decision in eight flips across refits. Referring contested cases to a human raises precision — but WIDENS the gender gap, because women sit at the margin more often. Abstention is not fairness-neutral.""",
 19: """[P6 · 12:40 · 60s] Reading across the four dimensions: performance is a tie; interpretability and stability favour logistic; fairness gaps are shared by all three. So: logistic for a shadow pilot, XGBoost as challenger. The two select 85% of the same people and the difference in captured re-arrests includes zero. We do NOT claim logistic is fairer or better calibrated. It reverses if the client quotes scores numerically, or operates at a scale where a dozen extra offers matter. And the pilot runs a second arm without gang affiliation, the field never recorded for women; appendix A11 shows it on all four dimensions.""",
 20: """[P1 · 13:40 · 45s] Demo one person: scores from all three models, the explanation, how many of eight refits select them, then edit an input and watch the score move. If the app fails, switch to the screenshots.

[APPENDIX CUES] A1 learning curve · A2 economics · A3 explanation disagreement · A4 global surrogate · A5 threshold frontier · A6 age · A7 process · A8 LIME · A9 permutation importance · A10 method overview · A11 no-gang pilot arm""",
}


for index, slide in enumerate(prs.slides, start=1):
    if index in NOTES:
        slide.notes_slide.notes_text_frame.text = NOTES[index]

REPORTS.mkdir(exist_ok=True)
prs.save(OUT)
print(f"{OUT}  ({len(prs.slides)} slides, speaker notes on {len(NOTES)})")
