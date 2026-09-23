from pathlib import Path

import json
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
ART, FIG, REPORTS = ROOT / "artifacts", ROOT / "artifacts" / "figures", ROOT / "reports"
OUT = REPORTS / "ISAF_Recidivism_Presentation.pptx"
NAVY, CARD, WHITE = RGBColor(10, 31, 68), RGBColor(20, 48, 88), RGBColor(255, 255, 255)
ORANGE, PURPLE, PALE = RGBColor(251, 133, 0), RGBColor(123, 44, 191), RGBColor(240, 244, 248)
GREY, RED, TEAL = RGBColor(130, 143, 160), RGBColor(214, 67, 67), RGBColor(0, 180, 180)


def textbox(slide, text, x, y, w, h, size=20, color=WHITE, bold=False, align=PP_ALIGN.LEFT, font="Aptos", valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear(); frame.word_wrap = True; frame.vertical_anchor = valign
    p = frame.paragraphs[0]
    p.text = text; p.alignment = align
    p.font.name = font; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
    return box


def make_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
    return slide


def title(slide, text, kicker):
    textbox(slide, kicker.upper(), .65, .28, 12, .3, 10, ORANGE, True)
    textbox(slide, text, .65, .68, 12, .65, 26, WHITE, True)
    line = slide.shapes.add_shape(1, Inches(.65), Inches(1.42), Inches(1.1), Inches(.055))
    line.fill.solid(); line.fill.fore_color.rgb = ORANGE; line.line.fill.background()


def footer(slide, number):
    textbox(slide, "HEC Paris · ISAF · 2026", .65, 7.12, 4, .18, 8, GREY)
    textbox(slide, str(number), 12.15, 7.1, .45, .18, 8, GREY, align=PP_ALIGN.RIGHT)


def bullets(slide, items, x=.8, y=1.8, w=11.7, h=4.8, size=21):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame; frame.clear(); frame.word_wrap = True
    for i, item in enumerate(items):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = "•  " + item; p.font.name = "Aptos"; p.font.size = Pt(size); p.font.color.rgb = WHITE; p.space_after = Pt(12)


def card(slide, label, value, x, y, color=ORANGE, note=""):
    shape = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(2.65), Inches(1.38))
    shape.fill.solid(); shape.fill.fore_color.rgb = CARD; shape.line.color.rgb = color
    textbox(slide, value, x+.12, y+.15, 2.4, .48, 24, color, True, PP_ALIGN.CENTER)
    textbox(slide, label, x+.12, y+.72, 2.4, .3, 11, WHITE, True, PP_ALIGN.CENTER)
    if note: textbox(slide, note, x+.12, y+1.06, 2.4, .18, 8, PALE, align=PP_ALIGN.CENTER)


def picture(slide, path, x, y, w):
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))


def panel(slide, heading, body, x, y, w=3.75, h=3.6, color=ORANGE):
    shape = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = CARD; shape.line.color.rgb = color
    textbox(slide, heading, x+.25, y+.3, w-.5, .4, 20, color, True, PP_ALIGN.CENTER)
    textbox(slide, body, x+.35, y+1.08, w-.7, h-1.35, 15, PALE, align=PP_ALIGN.CENTER)


metrics = pd.read_csv(ART / "model_metrics.csv").set_index("model")
inc_auc = pd.read_csv(ART / "incumbent_discrimination.csv").set_index("ranker")["roc_auc"]
inc_econ = pd.read_csv(ART / "incumbent_economics.csv").set_index("ranker")["assumed_net_value"]
inf = pd.read_csv(ART / "fairness_inference.csv")
fpr20 = inf[(inf.rule == "top_20pct") & (inf.metric == "fpr")].pivot_table(index="model", columns="attribute", values="gap")
pred = pd.read_csv(ART / "test_predictions.csv")

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

# 1 — Title
s = make_slide(prs)
textbox(s, "TRUSTWORTHY AI / RECIDIVISM", .72, .58, 8, .3, 12, ORANGE, True)
textbox(s, "Better than the tool\nagencies already use", .72, 1.42, 8, 1.65, 36, WHITE, True)
textbox(s, "For a risk-assessment software vendor: which three-year re-arrest model to ship, judged on performance, interpretability, stability and fairness", .76, 3.35, 6.6, 1.4, 17, PALE)
textbox(s, "Team 11 · HEC Paris · Fall 2026", .76, 6.62, 7, .3, 11, GREY)
panel(s, "0.60 → 0.73", "Incumbent score vs our models (ROC AUC)\n\n3 model families\n\n1 deployment recommendation", 8.55, .9, 3.65, 5.75, PURPLE)

# 2 — Client need
s = make_slide(prs); title(s, "Our client, and the decision they sell", "01 · Engagement")
bullets(s, ["Client: a vendor selling risk-assessment tools to US state community-supervision agencies",
            "Deliverable: a three-year re-arrest score to allocate limited, voluntary re-entry support",
            "Used only to expand help — never sanctions, detention or surveillance",
            "Judged as a trustworthy AI system, not on accuracy alone"], y=1.85, size=22); footer(s, 2)

# 3 — Data design
s = make_slide(prs); title(s, "A clean test set and a strict time boundary", "02 · Data design")
card(s, "TRAIN", "18,028", .8, 1.9); card(s, "UNTOUCHED TEST", "7,807", 3.75, 1.9, PURPLE)
card(s, "BASELINE FIELDS", "29", 6.7, 1.9); card(s, "TEST TARGET RATE", f"{pred.actual.mean():.1%}", 9.65, 1.9, PURPLE)
bullets(s, ["Post-release violations, tests, programs and employment excluded to prevent leakage",
            "Race, gender and residence geography excluded from scoring; kept only in the audit layer",
            "Balanced outcome (57.8%), so no resampling; official NIJ split preserves an honest evaluation"], y=3.72, h=2.4, size=20); footer(s, 3)

# 4 — Three models
s = make_slide(prs); title(s, "Three model families, one comparison", "03 · Model design")
panel(s, "LOGISTIC", "White-box anchor\n\nRegularized linear score\n\nSigned, inspectable effects", .7, 1.9, color=ORANGE)
panel(s, "XGBOOST", "Nonlinear workhorse\n\nOrdinal counts + 5-fold CV tuning\n\nFast operational scoring", 4.8, 1.9, color=TEAL)
panel(s, "TABICLv2", "Foundation model\n\nPretrained in-context transformer\n\nNo training, no native explanation", 8.9, 1.9, color=PURPLE)
textbox(s, "Same eligible fields · Same held-out people · Same metrics", 2.5, 6.22, 8.3, .4, 17, ORANGE, True, PP_ALIGN.CENTER); footer(s, 4)

# 5 — Incumbent benchmark (the hook)
s = make_slide(prs); title(s, "Is any of this better than today's tool?", "04 · The client's real question")
picture(s, FIG / "incumbent_benchmark.png", .35, 1.7, 9.1)
card(s, "INCUMBENT AUC", f"{inc_auc['incumbent']:.2f}", 9.75, 1.9, RED)
card(s, "OUR MODELS AUC", "0.73", 9.75, 3.5, TEAL)
card(s, "NET VALUE GAIN", f"+${(inc_econ['xgboost']-inc_econ['incumbent'])/1e6:.1f}M", 9.75, 5.1, ORANGE)
textbox(s, "The existing 1–10 actuarial score is barely better than a coin flip at ranking. Any model nearly doubles net value.", 1.0, 6.85, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 5)

# 6 — Predictive performance
s = make_slide(prs); title(s, "Among the three: a near-tie, XGBoost calibrates best", "05 · Predictive performance")
picture(s, FIG / "performance_calibration.png", .45, 1.72, 8.05)
card(s, "BEST AUC · TABICL", f"{metrics.loc['tabicl','roc_auc']:.4f}", 9.25, 1.9, PURPLE)
card(s, "BEST BRIER · TABICL", f"{metrics.loc['tabicl','brier']:.4f}", 9.25, 3.5, PURPLE)
card(s, "BEST ECE · XGBOOST", f"{metrics.loc['xgboost','ece_10']:.4f}", 9.25, 5.1, ORANGE)
textbox(s, "Bootstrap 95% intervals overlap — the ranking among the three is not decisive.", 1.0, 6.85, 11, .3, 11, GREY, True, PP_ALIGN.CENTER); footer(s, 6)

# 7 — Learning curve
s = make_slide(prs); title(s, "Which model for which agency size?", "06 · Learning curve")
picture(s, FIG / "learning_curve.png", .8, 1.8, 11.7)
textbox(s, "TabICLv2 leads on small data (small county); XGBoost catches up as data grows (large state). No crossover — but the gap closes to +0.004 at full data.", 1.0, 6.7, 11, .5, 13, ORANGE, True, PP_ALIGN.CENTER); footer(s, 7)

# 8 — Economic performance
s = make_slide(prs); title(s, "Model value is a scenario, stress-tested", "07 · Economic performance")
card(s, "SERVICE CAPACITY", "20%", .8, 1.9); card(s, "XGBOOST NET", f"${inc_econ['xgboost']/1e6:.2f}M", 3.75, 1.9, TEAL)
card(s, "INCUMBENT NET", f"${inc_econ['incumbent']/1e6:.2f}M", 6.7, 1.9, RED); card(s, "RANDOM NET", f"${inc_econ.get('random', 1.46e6)/1e6:.2f}M", 9.65, 1.9, GREY)
bullets(s, ["$5,000 support cost · $50,000 event cost · 20% assumed effectiveness — all editable in the app",
            "Models beat the incumbent at every capacity from 5% to 50% (sensitivity sweep)",
            "A randomized or quasi-experimental pilot must estimate real intervention impact"], y=3.72, size=20); footer(s, 8)

# 9 — Interpretability
s = make_slide(prs); title(s, "Every prediction, and the performance, explained", "08 · Interpretability")
picture(s, FIG / "shap_individual.png", .4, 1.7, 8.3)
panel(s, "METHODS", "SHAP + LIME (local)\n\nXPER on AUC (Pérignon)\n\nGlobal surrogate R²=0.61\n\nPDP/ICE for all three", 9.0, 1.7, 3.9, 4.9, PURPLE)
textbox(s, "TabICLv2 has no native explanation path — a real deployment cost, covered only by model-agnostic PDP/ICE.", 1.0, 6.75, 11, .3, 11, RED, True, PP_ALIGN.CENTER); footer(s, 9)

# 10 — Stability
s = make_slide(prs); title(s, "Structural stability across refits on resampled data", "09 · Stability")
picture(s, FIG / "structural_stability.png", .5, 1.75, 8.3)
card(s, "SCORE DRIFT · TABICL", "0.045", 9.35, 1.9, RED)
card(s, "TOP-20% OVERLAP", "73–77%", 9.35, 3.5, ORANGE)
textbox(s, "~1 person in 4 changes priority status across refits — scores need governance. TabICLv2 is least stable (4 refits vs 8, a stated cost trade-off).", 1.0, 6.75, 11, .4, 11, GREY, True, PP_ALIGN.CENTER); footer(s, 10)

# 11 — Fairness at the deployed point
s = make_slide(prs); title(s, "Fairness, audited where we actually deploy", "10 · Subgroup audit")
picture(s, FIG / "fairness_operating_point.png", .4, 1.7, 8.5)
card(s, "GENDER FPR · XGB", f"{fpr20.loc['xgboost','Gender']:.3f}", 9.35, 1.9, TEAL)
card(s, "GENDER FPR · TABICL", f"{fpr20.loc['tabicl','Gender']:.3f}", 9.35, 3.5, RED)
card(s, "RACE FPR · XGB", f"{fpr20.loc['xgboost','Race']:.3f}", 9.35, 5.1, ORANGE)
textbox(s, "At the top-20% deployment point, gaps are 2–4× smaller than at 0.5 — audit the point you ship.", 1.0, 6.8, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 11)

# 12 — Impossibility result
s = make_slide(prs); title(s, "Our data shows both sides of the impossibility theorem", "11 · Fairness, sharpened")
panel(s, "RACE — FIXABLE", "Base rates near-equal\n(0.582 vs 0.564)\n\nCalibration + equal error jointly achievable\n\nFPR gap is a model property → group thresholds drive it to ~0", .7, 1.85, 5.75, 4.7, TEAL)
panel(s, "GENDER — BINDS", "Base rates differ 13.7 pts\n(0.591 vs 0.454)\n\nTheorem binds: cannot equalize both\n\nEqualizing gender FPR decalibrates women — a choice we surface, not hide", 6.85, 1.85, 5.75, 4.7, RED)
footer(s, 12)

# 13 — Improvement journey (process)
s = make_slide(prs); title(s, "How we got here: what we tried, where we landed", "12 · Process")
picture(s, FIG / "improvement_journey.png", .55, 1.75, 12.2)
textbox(s, "Full log in JOURNEY.md — every step measured, including the attempts we rejected.", 1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 13)

# 14 — Trade-off matrix (required)
s = make_slide(prs); title(s, "Which model should the client deploy?", "13 · Trade-offs across four dimensions")
picture(s, FIG / "tradeoff_matrix.png", 2.2, 1.62, 8.9); footer(s, 14)

# 15 — Recommendation
s = make_slide(prs); title(s, "Deploy XGBoost; challenger logistic; TabICL for small agencies", "14 · Recommendation")
panel(s, "WHY XGBOOST", "Best calibration + net value\n\nSmallest gender FPR gap\n\nSHAP-explainable\n\n35× faster than the TFM", .8, 1.8, 5.75, 4.65, TEAL)
panel(s, "CONDITIONS", "Benefit-only allocation\n\nLogistic as transparent challenger\n\nProspective shadow validation\n\nAppeal route, logs, quarterly audits, stop rules", 6.8, 1.8, 5.75, 4.65, PURPLE); footer(s, 15)

# 15 — App + roadmap
s = make_slide(prs); title(s, "The application makes every trade-off testable", "15 · Client experience")
panel(s, "ASSESS", "Score one person, all 3 models + SHAP", .75, 1.75, 5.7, 1.72, ORANGE)
panel(s, "AUDIT", "Race + gender, at the deployed point", 6.85, 1.75, 5.7, 1.72, PURPLE)
panel(s, "COMPARE", "Incumbent, learning curve, matrix", .75, 3.8, 5.7, 1.72, ORANGE)
panel(s, "SIMULATE", "Costs, capacity, sensitivity", 6.85, 3.8, 5.7, 1.72, PURPLE)
textbox(s, "streamlit run app.py", 4.4, 6.25, 4.5, .4, 17, PALE, True, PP_ALIGN.CENTER, "Consolas"); footer(s, 16)

REPORTS.mkdir(exist_ok=True)
prs.save(OUT)
print(OUT)
