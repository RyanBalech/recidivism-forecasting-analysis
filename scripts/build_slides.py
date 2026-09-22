from pathlib import Path

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
    textbox(slide, text, .65, .68, 12, .65, 27, WHITE, True)
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
        p.text = "•  " + item; p.font.name = "Aptos"; p.font.size = Pt(size); p.font.color.rgb = WHITE; p.space_after = Pt(14)


def card(slide, label, value, x, y, color=ORANGE, note=""):
    shape = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(2.65), Inches(1.38))
    shape.fill.solid(); shape.fill.fore_color.rgb = CARD; shape.line.color.rgb = color
    textbox(slide, value, x+.12, y+.15, 2.4, .48, 25, color, True, PP_ALIGN.CENTER)
    textbox(slide, label, x+.12, y+.7, 2.4, .3, 11, WHITE, True, PP_ALIGN.CENTER)
    if note: textbox(slide, note, x+.12, y+1.05, 2.4, .18, 8, PALE, align=PP_ALIGN.CENTER)


def picture(slide, path, x, y, w):
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))


def panel(slide, heading, body, x, y, w=3.75, h=3.6, color=ORANGE):
    shape = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = CARD; shape.line.color.rgb = color
    textbox(slide, heading, x+.25, y+.3, w-.5, .4, 20, color, True, PP_ALIGN.CENTER)
    textbox(slide, body, x+.35, y+1.08, w-.7, h-1.35, 15, PALE, align=PP_ALIGN.CENTER)


metrics = pd.read_csv(ART / "model_metrics.csv").set_index("model")
gaps = pd.read_csv(ART / "fairness_gaps.csv")
pred = pd.read_csv(ART / "test_predictions.csv")
race = gaps[gaps.attribute.eq("Race")].set_index("model")
gender = gaps[gaps.attribute.eq("Gender")].set_index("model")

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

s = make_slide(prs)
textbox(s, "TRUSTWORTHY AI / RECIDIVISM", .72, .58, 6, .3, 12, ORANGE, True)
textbox(s, "A useful score\nmust earn trust", .72, 1.42, 7, 1.65, 38, WHITE, True)
textbox(s, "Comparing logistic regression, XGBoost and TabICLv2 across performance, interpretability, stability and fairness", .76, 3.35, 6.4, 1.25, 18, PALE)
textbox(s, "Ryan Balech and team · HEC Paris · Fall 2026", .76, 6.62, 7, .3, 11, GREY)
panel(s, "25,835 PEOPLE", "Georgia parole cohort\n\n3 model families\n\n1 deployment recommendation", 8.55, .9, 3.65, 5.75, PURPLE)

s = make_slide(prs); title(s, "Start with the decision, not the algorithm", "01 · Client need")
bullets(s, ["Prioritize limited, voluntary re-entry support at supervision start", "Estimate a recorded three-year arrest outcome — not inherent criminality", "Use only to expand help; never to justify sanctions, detention or surveillance", "Success means useful allocation without unacceptable subgroup harm"], y=1.85, size=23); footer(s, 2)

s = make_slide(prs); title(s, "A clean test set and a strict time boundary", "02 · Data design")
card(s, "TRAIN", "18,028", .8, 1.9); card(s, "UNTOUCHED TEST", "7,807", 3.75, 1.9, PURPLE)
card(s, "BASELINE FIELDS", "29", 6.7, 1.9); card(s, "TEST TARGET RATE", f"{pred.actual.mean():.1%}", 9.65, 1.9, PURPLE)
bullets(s, ["Post-release violations, tests, programs and employment are excluded to prevent leakage", "Race, gender and geography are excluded from scoring; race and gender remain in the audit layer", "The official NIJ split preserves an honest final evaluation"], y=3.72, h=2.4, size=20); footer(s, 3)

s = make_slide(prs); title(s, "Three models expose different trade-offs", "03 · Model design")
panel(s, "LOGISTIC", "White-box anchor\n\nRegularized linear score\n\nSigned, inspectable effects", .7, 1.9, color=ORANGE)
panel(s, "XGBOOST", "Nonlinear workhorse\n\n550 shallow trees\n\nFast operational scoring", 4.8, 1.9, color=TEAL)
panel(s, "TABICLv2", "Foundation model\n\nPretrained in-context transformer\n\nTwo ensemble views", 8.9, 1.9, color=PURPLE)
textbox(s, "Same eligible fields · Same held-out people · Same metrics", 2.5, 6.22, 8.3, .4, 17, ORANGE, True, PP_ALIGN.CENTER); footer(s, 4)

s = make_slide(prs); title(s, "TabICLv2 wins narrowly; XGBoost calibrates best", "04 · Predictive performance")
picture(s, FIG / "performance_calibration.png", .45, 1.62, 8.05)
card(s, "BEST BRIER · TABICL", f"{metrics.loc['tabicl','brier']:.4f}", 9.25, 1.82, PURPLE)
card(s, "BEST AUC · TABICL", f"{metrics.loc['tabicl','roc_auc']:.4f}", 9.25, 3.43, PURPLE)
card(s, "BEST ECE · XGBOOST", f"{metrics.loc['xgboost','ece_10']:.4f}", 9.25, 5.04, ORANGE); footer(s, 5)

s = make_slide(prs); title(s, "Model value is a scenario, not a causal claim", "05 · Economic performance")
card(s, "SERVICE CAPACITY", "20%", .8, 1.85); card(s, "EVENTS CAPTURED", "1,282–1,292", 3.75, 1.85, PURPLE)
card(s, "RECALL", "≈ 28.7%", 6.7, 1.85); card(s, "ILLUSTRATIVE NET", "$5.0–5.1M", 9.65, 1.85, PURPLE)
bullets(s, ["$5,000 support cost · $50,000 event cost · 20% assumed effectiveness", "Every assumption is editable in the client application", "A randomized or strong quasi-experimental pilot must estimate actual intervention impact"], y=3.72, size=20); footer(s, 6)

s = make_slide(prs); title(s, "The main drivers recur across model families", "06 · Interpretability")
picture(s, FIG / "feature_importance.png", .35, 1.6, 12.45)
textbox(s, "Permutation importance shows model reliance, not causal levers.", 3.1, 6.76, 7.1, .25, 12, ORANGE, True, PP_ALIGN.CENTER); footer(s, 7)

s = make_slide(prs); title(s, "Fairness changes the preferred model", "07 · Subgroup audit")
picture(s, FIG / "race_fairness.png", .35, 1.62, 8.4)
card(s, "RACE FPR GAP · XGB", f"{race.loc['xgboost','fpr_gap']:.3f}", 9.45, 1.82, ORANGE)
card(s, "GENDER FPR GAP · XGB", f"{gender.loc['xgboost','fpr_gap']:.3f}", 9.45, 3.43, ORANGE)
textbox(s, "TabICLv2 gender FPR gap", 9.25, 5.25, 3, .3, 11, PALE, align=PP_ALIGN.CENTER)
textbox(s, f"{gender.loc['tabicl','fpr_gap']:.3f}", 9.25, 5.66, 3, .5, 27, RED, True, PP_ALIGN.CENTER); footer(s, 8)

s = make_slide(prs); title(s, "Sampling stability is similar; temporal stability is unknown", "08 · Stability")
card(s, "AUC CI WIDTH", "0.022–0.023", .9, 2.0); card(s, "BRIER CI WIDTH", "0.0079–0.0088", 3.9, 2.0, PURPLE)
card(s, "BOOTSTRAPS", "400", 6.9, 2.0); card(s, "TIME SPLIT", "Unavailable", 9.9, 2.0, RED)
bullets(s, ["Feature-destruction stress tests confirm material input reliance", "A later release cohort is a hard gate before operational launch", "Monitor calibration, input drift and subgroup gaps every quarter"], y=4.02, size=20); footer(s, 9)

s = make_slide(prs); title(s, "Recommend XGBoost for a controlled pilot", "09 · Decision")
panel(s, "WHY", "Near-tied accuracy\n\nBest calibration\n\nSmallest observed race and gender FPR gaps\n\nFast operations", .8, 1.8, 5.75, 4.65, ORANGE)
panel(s, "CONDITIONS", "Benefit-only allocation\n\nLogistic challenger\n\nProspective shadow validation\n\nAppeal, override logs and stop rules", 6.8, 1.8, 5.75, 4.65, PURPLE); footer(s, 10)

s = make_slide(prs); title(s, "The application makes trade-offs testable", "10 · Client experience")
panel(s, "COMPARE", "Performance and calibration", .75, 1.75, 5.7, 1.72, ORANGE)
panel(s, "AUDIT", "Race and gender metrics", 6.85, 1.75, 5.7, 1.72, PURPLE)
panel(s, "SIMULATE", "Costs, capacity and benefit", .75, 3.8, 5.7, 1.72, ORANGE)
panel(s, "TEST", "Held-out people and scenarios", 6.85, 3.8, 5.7, 1.72, PURPLE)
textbox(s, "streamlit run app.py", 4.4, 6.25, 4.5, .4, 17, PALE, True, PP_ALIGN.CENTER, "Consolas"); footer(s, 11)

s = make_slide(prs); title(s, "Deployment is a staged evidence program", "11 · Roadmap")
steps = [("1", "SHADOW", "Later cohort"), ("2", "VALIDATE", "Impact + fairness"), ("3", "LAUNCH", "Narrow benefit use"), ("4", "MONITOR", "Quarterly + stop rules")]
for i, (num, head, body) in enumerate(steps):
    x = .72 + i*3.12
    textbox(s, num, x, 2.05, .65, .65, 28, ORANGE if i < 2 else PURPLE, True, PP_ALIGN.CENTER)
    textbox(s, head, x+.75, 2.08, 2.1, .32, 15, WHITE, True)
    textbox(s, body, x+.75, 2.53, 2.1, .55, 13, PALE)
    if i < 3: textbox(s, "→", x+2.78, 2.15, .4, .4, 22, GREY, True)
textbox(s, "Launch only if the score demonstrates benefit without unacceptable subgroup harm.", 1.35, 4.5, 10.6, 1.1, 25, WHITE, True, PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
textbox(s, "Sources: NIJ challenge brief and data; TabICLv2 model card and papers. Full methodology in the repository.", 1.2, 6.55, 10.9, .3, 9, GREY, align=PP_ALIGN.CENTER); footer(s, 12)

REPORTS.mkdir(exist_ok=True)
prs.save(OUT)
print(OUT)

