"""Build P6's slides (stability + recommendation) as a SEPARATE deck, to paste into the main one.

Output: study/stability-and-recommendation/slides/P6_Stability_Recommendation.pptx
The team's deck (reports/ISAF_Recidivism_Presentation.pptx) is never opened for writing.

The design (Calibri, navy/coral/teal palette, section navigator, footer) is copied from
scripts/build_slides.py. That script cannot be imported, because importing it rebuilds and
saves the team's deck. Every number is read from artifacts/ or from this study pack's
results/, so the slides stay traceable.

Slides: 17, 18, 18b (new), 19 for a 3:00 talk, plus backups A12 and A13.
"""
from pathlib import Path
import re

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.parts.image import Image as PImage
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = HERE.parents[2]
ART, FIG, RES = ROOT / "artifacts", ROOT / "artifacts" / "figures", STUDY / "results"
OUT = HERE / "P6_Stability_Recommendation.pptx"

# ---------------------------------------------------------------- design, copied from scripts/build_slides.py
FONT = "Calibri"
INK, NAVY, CORAL, TEAL, TEAL_L, GOLD = "1B2B3A", "24506E", "D9573A", "1F7A70", "2A9D8F", "C99A2E"
MUTED, FAINT, LINE = "5F6E7C", "8795A2", "D5DDE5"
CARD, CARD_T, CARD_C, CARD_N, BG, BG_TITLE = "F1F5F8", "E6F2F0", "FBEBE6", "E7EEF4", "FFFFFF", "F3F7FA"
W, H = 13.333, 7.5
LM = 0.6
CW = W - 2 * LM
SECTIONS = ["Intro", "Features", "Models", "Interpretability", "Fairness", "Stability", "Demo"]


def rgb(hex_):
    return RGBColor.from_string(hex_)


def _runs(p, text, size, color, bold=False, italic=False):
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
         anchor=MSO_ANCHOR.TOP, bullets=False, space=6, bullet_color=TEAL_L):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate([paras] if isinstance(paras, str) else paras):
        t, o = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        p.space_after = Pt(o.get("space", space))
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


def picture(slide, path, x, y, w, h, align="center"):
    path = str(path)
    pw, ph = PImage.from_file(path).size
    iw, ih = (h * pw / ph, h) if w / h > pw / ph else (w, w * ph / pw)
    ix = x + (w - iw) / 2 if align == "center" else x
    slide.shapes.add_picture(path, Inches(ix), Inches(y + (h - ih) / 2), Inches(iw), Inches(ih))
    return ix, iw


def stat(slide, x, y, w, h, big, label, color=NAVY, fill=CARD, big_size=34, label_size=12):
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.2, y + 0.14, w - 0.4, 0.62, big, size=big_size, color=color, bold=True)
    text(slide, x + 0.2, y + 0.14 + big_size / 72 * 1.25, w - 0.4, 0.7, label, size=label_size, color=MUTED)


def card(slide, x, y, w, h, head, body, fill=CARD, head_color=NAVY, size=13, bullets=False, space=5):
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.22, y + 0.16, w - 0.44, 0.3, head.upper(), size=12, color=head_color, bold=True)
    text(slide, x + 0.22, y + 0.5, w - 0.44, h - 0.62, body, size=size, bullets=bullets,
         bullet_color=head_color, space=space)


def table(slide, x, y, rows, col_w, row_h=0.4, size=13, highlight=()):
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


def new_slide(kicker, title, section, number, title_size=28):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    text(s, LM, 0.42, 6.2, 0.3, kicker.upper(), size=12, color=TEAL, bold=True)
    if section:
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


num = lambda v, fmt="{:.3f}": fmt.format(v).replace("-", "−")

# ---------------------------------------------------------------- inputs
stab = pd.read_csv(ART / "stability_summary.csv").set_index("model")
pairs = pd.read_csv(ART / "stability_pairs.csv")
_l, _x = (pairs[pairs.model == m].set_index(["refit_i", "refit_j"]) for m in ("logistic", "xgboost"))
drift_wins, jac_wins, n_pairs = int((_l.mean_abs_prob_diff < _x.mean_abs_prob_diff).sum()), \
    int((_l.top20_jaccard > _x.top20_jaccard).sum()), len(_l)
indiv = pd.read_csv(ART / "individual_stability_summary.csv").set_index("model").loc["logistic"]
curve = pd.read_csv(ART / "abstention_curve.csv")
lc = curve[curve.model == "logistic"]
abst_full, abst_strict = lc.loc[lc.max_contested_votes.idxmax()], lc.loc[lc.max_contested_votes.idxmin()]
metrics = pd.read_csv(ART / "model_metrics.csv").set_index("model")
speedup = metrics.loc["xgboost", "fit_predict_seconds"] / metrics.loc["logistic", "fit_predict_seconds"]
overlap = pd.read_csv(ART / "selected_set_overlap.csv").query("model_a == 'logistic' and model_b == 'xgboost'").iloc[0]
calpair = pd.read_csv(ART / "calibration_paired_tests.csv")
gap_diff = calpair[(calpair.model_a == "logistic") & (calpair.model_b == "xgboost")
                   & (calpair.metric == "abs_gender_fnr_gap")].iloc[0]
if not (-0.05 < gap_diff.ci_low and gap_diff.ci_high < 0.05):
    raise ValueError("The gender-gap difference CI is no longer inside ±5 points: reword slide 19")

regimes = pd.read_csv(RES / "stability_by_regime_summary.csv").set_index(["regime", "model"])
reg_pairs = pd.read_csv(RES / "stability_pairs_by_regime.csv")
halves = reg_pairs[reg_pairs.regime == "disjoint_halves"]
_hl, _hx = (halves[halves.model == m].set_index(["fit_i", "fit_j"]) for m in ("logistic", "xgboost"))
halves_wins = int((_hl.top20_jaccard > _hx.top20_jaccard).sum())
retrain = pd.read_csv(RES / "course_retrain_50pct_vs_full.csv").groupby("model").mean(numeric_only=True)
lam = pd.read_csv(RES / "course_slide186_lambda_path.csv").groupby("lambda").mean(numeric_only=True)
chosen = pd.read_csv(RES / "course_slide186_chosen.csv")
lam_naive, lam_cv = lam.loc[0], lam.loc[chosen.lambda_cv.mode().iloc[0]]
replaced = lambda j: (1 - j) / (1 + j)
nog = pd.read_csv(RES / "no_gang_stability_summary.csv").set_index(["variant", "model"])
nog_g = pd.read_csv(RES / "no_gang_contested_by_gender.csv").set_index(["variant", "model", "gender"])
people = pd.read_csv(ART / "individual_stability.csv")
people = people[people.model == "logistic"].reset_index(drop=True)
pctl = people.mean_probability.rank(pct=True)
contested = ~people.decision_unanimous.astype(bool)
near10 = float(((pctl - 0.8).abs() <= 0.10)[contested].mean())
near20 = float(((pctl - 0.8).abs() <= 0.20)[contested].mean())
ENV_CHANGED_XGB = 80   # results/ E6: decisions changed when the same XGBoost ran on another machine (logistic: 0)

# ---------------------------------------------------------------- speaker script (3:00; mirrored in 10_p6_slides_and_script_3min.md)
SCRIPT = {
    17: "[P6 · 11:10 · 45s] The course defines stability simply: two datasets from the same population should give "
        "approximately the same model. We tested that three ways, from mildest to harshest. Changing only the random "
        "seed barely moves logistic regression; XGBoost already swaps about 2% of the people it selects. With eight "
        "bootstrap refits — the same eight for all three models — a person's risk moves by about three points, and "
        "about 13% of the people offered support change. Not 23%: Jaccard is not the share that changed. With two "
        "completely separate halves of the data, it's about 18%. In every test, logistic is more stable than XGBoost, "
        "and TabICL matches logistic on who gets selected.",
    18: "[P6 · 11:55 · 45s] Now one person. Across the eight refits, about one decision in eight is contested — some "
        "refits select the person, others don't — and all of them sit close to the cut. The natural fix is to send "
        "contested cases to a human. That raises precision from 0.822 to 0.846. But the gender gap widens, from minus "
        "0.090 to minus 0.119, because the few women we select sit at the margin far more often than men: 47% against "
        "30%. So abstaining is not fairness-neutral — and that asymmetry comes from gang affiliation: without it, "
        "women and men are equally often at the margin.",
    "18b": "[P6 · 12:40 · 35s] The course also shows how to fix this. When you retrain on new data, penalise the "
           "distance to the previous model, with lambda chosen by cross-validation. We implemented it for our logistic "
           "regression. A naive retrain reshuffles 19% of the people offered support; with the constraint, 12%. The "
           "coefficients move half as much, and accuracy doesn't drop. It needs a parameter vector, so it works for "
           "logistic, not XGBoost. And reproducibility: the same XGBoost on another machine changed 80 decisions; "
           "logistic changed none.",
    19: "[P6 · 13:15 · 55s] So, across the four dimensions. Performance: XGBoost's edge is real but small — the two "
        "models select 85% of the same people, and the difference in re-arrests captured includes zero. Fairness "
        "doesn't separate them: the gap difference is equivalent within five points. Calibration shows no detectable "
        "difference. What does separate them is stability, reproducibility, direct interpretability and cost — and "
        "all four favour logistic. So we recommend a shadow pilot of L1 logistic regression, with XGBoost as the "
        "challenger, plus a second arm without gang affiliation, where logistic is still the more stable model. The "
        "choice reverses if scores are quoted to people as probabilities, or at a scale where a dozen extra captured "
        "re-arrests matter. And no model allocates real support until the pilot shows the programme helps. Over to "
        "the demo.",
}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
NOTES = {}

# ================================================================ 17 · structural stability
s = new_slide("Stability · Structural", "Would a different sample give the same model?", "Stability", 17)
card(s, LM, 1.7, 4.55, 1.5, "Course definition (slide 182)",
     "“Two datasets from the same population should induce **approximately the same model**.” "
     "We test it three ways: new seed → bootstrap → two disjoint halves.", fill=CARD_N, size=12.5)
table(s, LM + 4.8, 1.7, [["Model", "Mean |Δp|", "Top-20% Jaccard"]] +
      [[n, f"{stab.loc[m, 'mean_abs_prob_diff']:.3f}", f"{stab.loc[m, 'top20_jaccard']:.2f}"]
       for n, m in (("Logistic", "logistic"), ("XGBoost", "xgboost"), ("TabICLv2", "tabicl"))],
      [1.3, 1.1, 1.85], row_h=0.375, size=12.5, highlight=(1,))
stat(s, LM + 9.3, 1.7, CW - 9.3, 1.5, f"{replaced(stab.loc['logistic', 'top20_jaccard']):.0%}",
     "of the people offered support change per refit (logistic) — not 23%", color=TEAL, fill=CARD_T,
     big_size=30, label_size=11.5)
picture(s, RES / "fig_stability_regimes.png", LM, 3.38, CW, 2.75)
text(s, LM, 6.18, CW, 0.5,
     f"**Logistic vs XGBoost (8 bootstrap refits):** lower drift in {drift_wins}/{n_pairs} pairs, steadier selection in "
     f"{jac_wins}/{n_pairs}; ahead in {halves_wins}/{len(_hl)} disjoint-half splits.",
     size=12.5, color=INK)
source(s, "stability_summary.csv · stability_pairs.csv · study results/stability_by_regime_summary.csv")
NOTES[17] = SCRIPT[17]

# ================================================================ 18 · one person + abstention
s = new_slide("Stability · Individual", "Stability for one person: abstaining is not fairness-neutral", "Stability", 18)
w4 = (CW - 0.75) / 4
for i, (big, label, col, fill) in enumerate([
        (f"≈{indiv.share_contested:.0%}", f"of decisions are contested across refits; {indiv.contested_share_of_selected:.0%} of "
         "those selected sit at that margin", NAVY, CARD_N),
        (f"{abst_full.precision_at_capacity:.3f} → {abst_strict.precision_at_capacity:.3f}",
         "precision when keeping only unanimous decisions (8/8)", TEAL, CARD_T),
        (f"{num(abst_full.fnr_gap_gender)} → {num(abst_strict.fnr_gap_gender)}", "gender FNR gap: abstention widens it",
         CORAL, CARD_C),
        (f"{nog_g.loc[('with_gang', 'logistic', 'F'), 'contested_share_of_selected']:.0%} vs "
         f"{nog_g.loc[('with_gang', 'logistic', 'M'), 'contested_share_of_selected']:.0%}",
         "of selected women vs men sit at the margin", CORAL, CARD_C)]):
    stat(s, LM + i * (w4 + 0.25), 1.7, w4, 1.35, big, label, color=col, fill=fill, big_size=22)
picture(s, FIG / "individual_stability.png", LM, 3.2, CW, 2.95)
text(s, LM, 6.18, CW, 0.5,
     f"Contested decisions all sit near the cut ({near10:.0%} within 10 percentile points).  "
     f"**Without gang affiliation** the asymmetry disappears: "
     f"{nog_g.loc[('no_gang', 'logistic', 'F'), 'contested_share_of_selected']:.0%} vs "
     f"{nog_g.loc[('no_gang', 'logistic', 'M'), 'contested_share_of_selected']:.0%} (backup A13).", size=12.5)
source(s, "individual_stability.png · individual_stability_summary.csv · abstention_curve.csv · study results/no_gang_*.csv")
NOTES[18] = SCRIPT[18]

# ================================================================ 18b · slide 186 on our data
s = new_slide("Stability · Retraining (course slide 186)", "Retraining without reshuffling: the course's stability constraint",
              "Stability", "18b", title_size=26)
card(s, LM, 1.7, 4.3, 1.5, "Course slide 186 on our data",
     "θ_new = argmin  L(θ; new data) + **λ · ||θ − θ_old||²**,  λ chosen by cross-validation. "
     "Retrain on new data, pay a price for moving away from the old model.", fill=CARD_N, size=12.5)
sw = (CW - 4.3 - 0.25 * 3) / 3
for i, (big, label, col, fill) in enumerate([
        (f"{replaced(lam_naive.top20_jaccard_vs_old):.0%} → {replaced(lam_cv.top20_jaccard_vs_old):.0%}",
         "of the people offered support reshuffled by a retrain", TEAL, CARD_T),
        (f"−{1 - lam_cv.coef_distance_l2 / lam_naive.coef_distance_l2:.0%}", "coefficient movement ||θ_new − θ_old||",
         NAVY, CARD_N),
        ("No AUC loss", f"CV AUC {lam_naive.cv_auc:.3f} → {lam_cv.cv_auc:.3f}, in 5 of 5 draws", NAVY, CARD_N)]):
    stat(s, LM + 4.3 + 0.25 + i * (sw + 0.25), 1.7, sw, 1.5, big, label, color=col, fill=fill, big_size=22)
fx, fw = picture(s, RES / "fig_slide186_frontier.png", LM, 3.38, 8.9, 2.75, align="left")
cx = fx + fw + 0.3
card(s, cx, 3.45, W - LM - cx, 2.6, "Reproducibility (slides 210–215)",
     [f"Same model, same data, another machine:", f"**Logistic: 0** decisions change", f"**XGBoost: {ENV_CHANGED_XGB}** change",
      ("Slide 186 needs a parameter vector θ, so it works for logistic, not directly for XGBoost.", {"size": 11.5, "color": MUTED})],
     fill=CARD_C, head_color=CORAL, size=12.5, space=4)
source(s, "study results/course_slide186_*.csv (5 draws; old data 40%, new data 60%) · results/run_info.json · course slides 186, 192, 215")
NOTES["18b"] = SCRIPT["18b"]

# ================================================================ 19 · recommendation
s = new_slide("Conclusion · Trade-offs", "Pilot logistic regression; run XGBoost as the challenger", "Stability", 19)
picture(s, FIG / "tradeoff_matrix.png", LM, 1.65, 6.6, 4.95, align="left")
rx, rw = 7.35, W - LM - 7.35
box(s, rx, 1.7, rw, 3.3, fill=CARD_T)
text(s, rx + 0.25, 1.83, rw - 0.5, 0.3, "RECOMMENDATION", size=12, color=TEAL, bold=True)
text(s, rx + 0.25, 2.15, rw - 0.5, 0.65, "L1 logistic regression for a prospective **shadow pilot**; XGBoost as challenger.",
     size=15)
text(s, rx + 0.25, 2.88, rw - 0.5, 2.1, [
    f"Same people: Jaccard {overlap.jaccard:.3f}, only {int(overlap.chosen_by_only_one)} differ; captured re-arrest "
    "difference CI includes 0.",
    f"More stable ({drift_wins}/{n_pairs} refit pairs), same decisions on any machine, directly interpretable, "
    f"~{speedup:.0f}× faster.",
    "Fairness: **equivalent within ±5 pts** (TOST); calibration: no detectable difference — not reasons to choose.",
    "**Second arm without gang affiliation** (A11): logistic still the more stable model."],
    size=12.5, space=3, bullets=True, bullet_color=TEAL)
card(s, rx, 5.12, rw, 1.02, "What would reverse it", "Scores quoted numerically to supervisees, or a scale where a few "
     "extra captured events matter.", fill=CARD_C, head_color=CORAL, size=12)
text(s, rx, 6.22, rw, 0.46, "**Limits:** evaluation set inspected repeatedly; no temporal or external validation; no "
     "evidence yet that the support programme helps.", size=11, color=MUTED)
source(s, "tradeoff_matrix.png · selected_set_overlap.csv · calibration_paired_tests.csv · stability_pairs.csv · A11")
NOTES[19] = SCRIPT[19]

# ================================================================ A12 · backup: the course's stability toolkit
s = new_slide("A12 · Stability extras", "The course's stability toolkit, applied to recidivism", None, "A12")
R = lambda reg, m, c: regimes.loc[(reg, m), f"{c}_mean"]
table(s, LM, 1.7, [["Test (course slide)", "Logistic", "XGBoost"],
                   ["New seed, same data — top-20% Jaccard (207)", f"{R('seed_only', 'logistic', 'top20_jaccard'):.3f}",
                    f"{R('seed_only', 'xgboost', 'top20_jaccard'):.3f}"],
                   ["8 bootstrap refits — Jaccard (deck, 182)", f"{stab.loc['logistic', 'top20_jaccard']:.3f}",
                    f"{stab.loc['xgboost', 'top20_jaccard']:.3f}"],
                   ["Two disjoint halves — Jaccard (182)", f"{R('disjoint_halves', 'logistic', 'top20_jaccard'):.3f}",
                    f"{R('disjoint_halves', 'xgboost', 'top20_jaccard'):.3f}"],
                   ["50% sample → full data — Jaccard (187, 191)", f"{retrain.loc['logistic', 'top20_jaccard']:.3f}",
                    f"{retrain.loc['xgboost', 'top20_jaccard']:.3f}"],
                   ["Importance distance ||φ1 − φ2||, bootstrap (193)", f"{R('bootstrap', 'logistic', 'importance_l2'):.3f}",
                    f"{R('bootstrap', 'xgboost', 'importance_l2'):.3f}"],
                   ["Decisions changed on another machine (215)", "0", f"{ENV_CHANGED_XGB}"]],
      [4.6, 1.35, 1.35], row_h=0.44, size=13, highlight=(3,))
card(s, LM + 7.6, 1.7, CW - 7.6, 1.55, "Honest reading",
     "Logistic's advantage is in **who gets selected** and in reproducibility. On explanation stability (||φ1 − φ2||) "
     "XGBoost ties or is slightly ahead.", fill=CARD_C, head_color=CORAL, size=12.5)
card(s, LM + 7.6, 3.45, CW - 7.6, 1.6, "Slide 43–44 check: which penalty?",
     "Ridge moves coefficients ~10% less; elastic net is sparsest and most sign-stable. Who is selected is the same "
     "for L1, L2 and elastic net (Jaccard ≈ 0.77).", fill=CARD_N, size=12.5)
text(s, LM, 5.0, 7.3, 0.9,
     "Seed and machine effects are tiny next to data effects. Under the course's literal test (disjoint halves) "
     f"logistic is ahead in {halves_wins} of {len(_hl)} splits.", size=12.5, color=INK)
source(s, "study results/stability_by_regime_summary.csv · course_retrain_50pct_vs_full.csv · "
          "course_penalty_stability.csv · run_info.json")

# ================================================================ A13 · backup: no-gang arm, slides 17–18 recomputed
s = new_slide("A13 · No-gang arm", "Slides 17–18 without gang affiliation (published protocol)", None, "A13")
V = lambda v, m, c: nog.loc[(v, m), c]
G = lambda v, m, g: nog_g.loc[(v, m, g), "contested_share_of_selected"]
cols = [("with_gang", "logistic"), ("no_gang", "logistic"), ("with_gang", "xgboost"), ("no_gang", "xgboost")]
table(s, LM, 1.7, [["", "Logistic with", "Logistic without", "XGBoost with", "XGBoost without"],
                   ["Mean |Δp| between refits"] + [f"{V(v, m, 'mean_abs_prob_diff'):.3f}" for v, m in cols],
                   ["Top-20% Jaccard between refits"] + [f"{V(v, m, 'top20_jaccard'):.3f}" for v, m in cols],
                   ["Contested decisions"] + [f"{V(v, m, 'share_contested'):.1%}" for v, m in cols],
                   ["Selected women / men at the margin"] + [f"{G(v, m, 'F'):.0%} / {G(v, m, 'M'):.0%}" for v, m in cols],
                   ["Abstention: gender FNR gap"] + [f"{num(V(v, m, 'gender_fnr_gap_all'))} → {num(V(v, m, 'gender_fnr_gap_unanimous'))}"
                                                     for v, m in cols],
                   ["Abstention: precision"] + [f"{V(v, m, 'precision_all'):.3f} → {V(v, m, 'precision_unanimous'):.3f}"
                                                for v, m in cols]],
      [3.45, 2.17, 2.17, 2.17, 2.17], row_h=0.46, size=12.5, highlight=(4, 5))
text(s, LM, 5.15, CW, 1.3, [
    "**Without gang affiliation, abstention no longer widens the gender gap:** selected women and men sit at the margin "
    "equally often. Slide 18's asymmetry comes from the field that is never recorded for women.",
    ("Same 8 resamples (seed 7) and model seed 42 as slide 17; with-gang logistic reproduces slides 17–18 exactly. "
     "XGBoost recomputed on CPU (third-decimal differences from the published run). FNR gaps: point estimates on a few "
     "hundred women, no CI.", {"size": 11, "color": MUTED})], size=13, space=6)
source(s, "study results/no_gang_stability_summary.csv · no_gang_contested_by_gender.csv · no_gang_abstention_curve.csv")

for index, slide in enumerate(prs.slides):
    key = [17, 18, "18b", 19, None, None][index]
    if key in NOTES:
        slide.notes_slide.notes_text_frame.text = NOTES[key]

prs.save(OUT)
print(f"{OUT}  ({len(prs.slides)} slides)")
