from pathlib import Path

import json
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from PIL import Image


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
    # Keep tall figures inside the content area rather than clipping the footer.
    with Image.open(path) as img:
        ratio = img.height / img.width
    fitted_width = min(w, (6.85 - y) / ratio)
    x += (w - fitted_width) / 2
    w = fitted_width
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

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

# 1 — Title
s = make_slide(prs)
textbox(s, "TRUSTWORTHY AI / RECIDIVISM", .72, .58, 8, .3, 12, ORANGE, True)
textbox(s, "Forecasting risk,\nexamining trade-offs", .72, 1.42, 8, 1.65, 36, WHITE, True)
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
s = make_slide(prs); title(s, "Original split, with explicit validation limits", "02 · Data design")
card(s, "TRAIN", "18,028", .8, 1.9); card(s, "EVALUATION", "7,807", 3.75, 1.9, PURPLE)
card(s, "BASELINE FIELDS", "29", 6.7, 1.9); card(s, "TEST TARGET RATE", f"{pred.actual.mean():.1%}", 9.65, 1.9, PURPLE)
bullets(s, ["Post-release variables excluded; race, gender and geography retained only for audit",
            f"Found and fixed a representation leak: gang affiliation is missing for every woman and no man, so with missingness indicators gender is recoverable at AUC {leak_auc:.2f} — mode-fill plus a regression audit now blocks it",
            "Evaluation data were repeatedly inspected during development; fresh validation is still needed"], y=3.62, h=2.65, size=18); footer(s, 3)

# 4 — Three models
s = make_slide(prs); title(s, "Three model families, one comparison", "03 · Model design")
panel(s, "LOGISTIC", "White-box anchor\n\nCV-tuned L1, C=0.2154\n\nSigned, inspectable effects", .7, 1.9, color=ORANGE)
panel(s, "XGBOOST", "Nonlinear workhorse\n\nOrdinal counts + 5-fold CV tuning\n\nFast operational scoring", 4.8, 1.9, color=TEAL)
panel(s, "TABICLv2", "Foundation model\n\n16-member GPU ensemble\n\nTraining-only size sweep; no native attribution implemented", 8.9, 1.9, color=PURPLE)
textbox(s, "Same eligible fields · Same held-out people · Same metrics", 2.5, 6.22, 8.3, .4, 17, ORANGE, True, PP_ALIGN.CENTER); footer(s, 4)

# 5 — Incumbent benchmark (the hook)
s = make_slide(prs); title(s, "Comparison with the historical recorded score", "04 · The client's real question")
picture(s, FIG / "incumbent_benchmark.png", .35, 1.7, 9.1)
card(s, "INCUMBENT AUC", f"{inc_auc['incumbent']:.2f}", 9.75, 1.9, RED)
card(s, "OUR MODELS AUC", "0.73", 9.75, 3.5, TEAL)
card(s, "NET VALUE GAIN", f"+${(inc_econ['logistic']-inc_econ['incumbent'])/1e6:.1f}M", 9.75, 5.1, ORANGE, "recommended logistic model")
textbox(s, "This historical comparison does not establish superiority to current agency products. Dollar gains are scenarios.", 1.0, 6.85, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 5)

# 6 — Predictive performance
s = make_slide(prs); title(s, "Compare ranking, probability loss and calibration", "05 · Predictive performance")
picture(s, FIG / "performance_calibration.png", .45, 1.72, 8.05)
card(s, "AUC · TABICL", f"{metrics.loc['tabicl','roc_auc']:.4f}", 9.25, 1.9, PURPLE)
card(s, "BRIER · TABICL", f"{metrics.loc['tabicl','brier']:.4f}", 9.25, 3.5, PURPLE)
card(s, "AUC GAP · XGB − LOGIT", f"+{xgb_minus_logit:.4f}", 9.25, 5.1, ORANGE, "paired bootstrap: significant, small")
textbox(s, "All three models are within ~0.003 AUC; ECE differences are not significant. Performance alone cannot pick the model.", 1.0, 6.85, 11, .3, 11, GREY, True, PP_ALIGN.CENTER); footer(s, 6)

# 7 — Interpretability (brief: explain individual predictions and the main drivers)
s = make_slide(prs); title(s, "Can each model explain a decision?", "06 · Interpretability")
picture(s, FIG / "shap_individual.png", .4, 1.7, 8.3)
panel(s, "THREE ROUTES", "LOGISTIC\nCoefficients and marginal effects, read directly\n\n"
      f"XGBOOST\nNeeds SHAP; a depth-3 surrogate tree reproduces only R² = {surrogate_r2:.2f}\n\n"
      "TABICLv2\nNo native attribution: PDP/ICE only", 9.0, 1.7, 3.9, 4.9, PURPLE)
textbox(s, f"SHAP, permutation importance and XPER agree on the drivers (Spearman {agree_lo:.2f}–{agree_hi:.2f}), "
        "not on their order: quote the set, not the rank.", 1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 7)

# 8 — Structural stability (course §7: same population, approximately the same model)
s = make_slide(prs); title(s, "Would a different sample give the same model?", "07 · Stability")
picture(s, FIG / "structural_stability.png", .5, 1.75, 8.3)
card(s, "SCORE DRIFT · LOGISTIC", f"{stability.loc['logistic','mean_abs_prob_diff']:.3f}", 9.35, 1.9, ORANGE,
     f"lowest on all 28 pairs · XGB {stability.loc['xgboost','mean_abs_prob_diff']:.3f} · TabICL {stability.loc['tabicl','mean_abs_prob_diff']:.3f}")
card(s, "TOP-20% JACCARD", f"{stability.top20_jaccard.min():.0%}–{stability.top20_jaccard.max():.0%}", 9.35, 3.5, TEAL,
     f"logit {stability.loc['logistic','top20_jaccard']:.3f} · XGB {stability.loc['xgboost','top20_jaccard']:.3f} · TabICL {stability.loc['tabicl','top20_jaccard']:.3f}")
card(s, "DECISIONS THAT FLIP", f"{contested_share:.0%}", 9.35, 5.1, RED, "per person, across 8 refits (appendix A7)")
textbox(s, "Eight bootstrap refits per model, same resamples for all three. Jaccard is intersection / union, not the share switching.",
        1.0, 6.82, 11, .3, 11, GREY, True, PP_ALIGN.CENTER); footer(s, 8)

# 9 — Fairness at the deployed point
s = make_slide(prs); title(s, "Fairness at the proposed allocation rule", "08 · Subgroup audit")
picture(s, FIG / "fairness_support_access.png", .4, 1.7, 8.5)
card(s, "GENDER FNR GAP (M − F)", span(fnr20["Gender"]), 9.35, 1.9, RED, "women miss support more · all significant")
card(s, "RACE: EQUIVALENT ±5 PTS", f"{int(race_equiv.sum())} / {len(race_equiv)} models", 9.35, 3.5, TEAL,
     f"TOST on errors · AUC Black {min(race_auc[0::2]):.2f} vs White {min(race_auc[1::2]):.2f}")
card(s, "AGE FNR GAP (<33 − 33+)", span(fnr20["Age"]), 9.35, 5.1, ORANGE, "persists with age fixed: carried by prior record")
textbox(s, "Selection = support offered, so the harm is a missed offer: FNR is primary. Not significant ≠ fair; we test equivalence.", 1.0, 6.8, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 9)

# 10 — Impossibility result
s = make_slide(prs); title(s, "Fairness: probability scores and allocation rules", "09 · Fairness, sharpened")
panel(s, "SCORE CALIBRATION", f"Women are over-predicted by every model: mean score {calib.loc[('Gender', 'F'), 'mean_score_xgboost']:.2f} vs observed {calib.loc[('Gender', 'F'), 'base_rate']:.2f}\n\nYet they are selected less at the top 20%\n\nFor support, over-prediction helps women: recalibrating would widen the FNR gap", .7, 1.85, 5.75, 4.7, TEAL)
panel(s, "ALLOCATION ERRORS", "FPR, TPR and selection rates depend on policy\n\nChanging thresholds leaves probabilities unchanged\n\nOur threshold frontier uses evaluation labels: illustrative, not validated", 6.85, 1.85, 5.75, 4.7, RED)
footer(s, 10)

# 11 — Fairness interpretability: where the gender gap comes from
s = make_slide(prs); title(s, "Where does the gender gap come from?", "10 · Fairness interpretability (FPDP)")
picture(s, FIG / "fpdp_gender.png", .35, 1.65, 8.4)
panel(s, "CANDIDATE: GANG", "Never recorded for women; imputed as \"No\"\n\n"
      f"Drop + re-estimate (logistic): FNR gap {base_gender.loc['logistic', 'fnr_gap']:+.3f} → {drop_gang.loc['logistic', 'fnr_gap']:+.3f}\n\n"
      f"Women FNR {base_gender.loc['logistic', 'fnr_b']:.2f} → {drop_gang.loc['logistic', 'fnr_b']:.2f}; men {base_gender.loc['logistic', 'fnr_a']:.2f} → {drop_gang.loc['logistic', 'fnr_a']:.2f}\n\n"
      f"AUC −{gang_cost:.3f}; {base_gender.loc['logistic', 'captured_events'] - drop_gang.loc['logistic', 'captured_events']:.0f} fewer re-arrests captured", 9.0, 1.65, 3.95, 5.0, RED)
textbox(s, "Logistic FPDP is flat under a top-20% rule (a constant shifts every score equally), so it measures removing the feature, not a value.", 1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 11)

# 12 — Does the mitigation survive out-of-sample?
s = make_slide(prs); title(s, "Selecting and testing the fix on different data", "11 · Mitigation, validated")
panel(s, "IN-SAMPLE", "Candidate chosen AND scored on the same cohort\n\n"
      f"Equal-opportunity p 0.000 → {drop_gang.loc['logistic', 'p_equal_opportunity']:.2f}\n\n"
      "An upper bound, not evidence of generalisation", .7, 1.85, 5.75, 4.7, GREY)
panel(s, "OUT-OF-FOLD", "Candidate re-selected inside each training fold, scored on held-out folds\n\n"
      f"Same variable selected in {int(nested_sel * 100)}% of folds\n\n"
      f"Gender FNR gap {nested_base:+.3f} → {nested_mit:+.3f}  ·  AUC {nested_auc:+.3f}", 6.85, 1.85, 5.75, 4.7, TEAL)
textbox(s, "The evaluation cohort is never touched. About 70% of the gap closes out of fold, for roughly one AUC point.",
        1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 12)

# 13 — Trade-off matrix (required)
s = make_slide(prs); title(s, "Which model should the client deploy?", "12 · Trade-offs across four dimensions")
picture(s, FIG / "tradeoff_matrix.png", 2.2, 1.62, 8.9); footer(s, 13)

# 14 — Recommendation
s = make_slide(prs); title(s, "Pilot logistic regression; run XGBoost as the challenger", "13 · Recommendation")
panel(s, "WHY LOGISTIC", "Native, coefficient-level explanations — no second tool\n\n"
      f"Lower refit drift on all 28 pairs ({stability.loc['logistic','mean_abs_prob_diff']:.4f} vs {stability.loc['xgboost','mean_abs_prob_diff']:.4f})\n\n"
      f"More stable selected set (J {stability.loc['logistic','top20_jaccard']:.3f} vs {stability.loc['xgboost','top20_jaccard']:.3f})\n\n"
      "~9× faster", .7, 1.8, 3.85, 4.65, TEAL)
panel(s, "WHAT WE GIVE UP", f"XGBoost AUC edge +{xgb_minus_logit:.4f} and Brier −0.0010\n\n"
      "Calibration is a tie, not a loss: both indistinguishable from perfect\n\n"
      f"Captured re-arrests: interval {int(abs(cap_ci[0]))} to {int(abs(cap_ci[1]))} includes 0\n\n"
      "Significant, but not at the operating point", 4.75, 1.8, 3.85, 4.65, RED)
panel(s, "WHAT REVERSES IT", "The score is quoted numerically to supervisees\n\n"
      "Capacity large enough that +12 offers matters\n\nA feature set that widens the margin\n\nThen: promote the challenger",
      8.8, 1.8, 3.85, 4.65, PURPLE)
textbox(s, f"Both models select {overlap_share:.0%} of the same people at the top-20% rule. Fairness and calibration differences between them are not significant.",
        .8, 6.75, 11.7, .4, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 14)

# 15 — App
s = make_slide(prs); title(s, "The application makes every trade-off testable", "14 · Client experience")
panel(s, "ASSESS", "Score one person, all 3 models + SHAP", .75, 1.75, 5.7, 1.72, ORANGE)
panel(s, "AUDIT", "Race, gender, age at the deployed point + FPDP", 6.85, 1.75, 5.7, 1.72, PURPLE)
panel(s, "COMPARE", "Incumbent, learning curve, matrix", .75, 3.8, 5.7, 1.72, ORANGE)
panel(s, "SIMULATE", "Costs, capacity, sensitivity", 6.85, 3.8, 5.7, 1.72, PURPLE)
textbox(s, "streamlit run app.py", 4.4, 6.25, 4.5, .4, 17, PALE, True, PP_ALIGN.CENTER, "Consolas"); footer(s, 15)

# ---------------------------------------------------------------- APPENDIX
# Everything below is reference material for Q&A, not part of the 15-minute talk.
s = make_slide(prs)
textbox(s, "APPENDIX", .72, 2.6, 8, .5, 14, ORANGE, True)
textbox(s, "Supporting evidence\nfor questions", .72, 3.15, 9, 1.5, 34, WHITE, True)
textbox(s, "Learning curve · economic sensitivity · explanation disagreement · LIME fidelity · process log · proxy recovery · decision stability",
        .76, 4.95, 9.5, .9, 15, PALE)

# A1 — Learning curve
s = make_slide(prs); title(s, "Which model for which agency size?", "A1 · Learning curve")
picture(s, FIG / "learning_curve.png", .8, 1.8, 11.7)
textbox(s, "Subsamples of one Georgia cohort test sample-size sensitivity; they do not validate transfer to other agencies.", 1.0, 6.7, 11, .5, 13, ORANGE, True, PP_ALIGN.CENTER)

# A2 — Economic performance
s = make_slide(prs); title(s, "Model value is a scenario, stress-tested", "A2 · Economic performance")
card(s, "SERVICE CAPACITY", "20%", .8, 1.9)
card(s, "LOGISTIC NET", f"${inc_econ['logistic']/1e6:.2f}M", 3.75, 1.9, TEAL, f"XGBoost ${inc_econ['xgboost']/1e6:.2f}M")
card(s, "INCUMBENT NET", f"${inc_econ['incumbent']/1e6:.2f}M", 6.7, 1.9, RED); card(s, "RANDOM NET", f"${inc_econ.get('random', 1.46e6)/1e6:.2f}M", 9.65, 1.9, GREY)
bullets(s, ["$5,000 support cost · $50,000 event cost · 20% assumed effectiveness — all editable in the app",
            "Models beat the incumbent at every capacity from 5% to 50% (sensitivity sweep)",
            "A randomized or quasi-experimental pilot must estimate real intervention impact"], y=3.72, size=20)

# A3 — Explanations disagree
s = make_slide(prs); title(s, "SHAP, permutation importance and XPER rank differently", "A3 · Explanation disagreement")
picture(s, FIG / "explanation_agreement.png", .5, 1.7, 8.4)
panel(s, "WHY", "SHAP explains the prediction\n\nPermutation importance explains loss\n\nXPER decomposes AUC\n\n"
      "Different questions, so different rankings", 9.1, 1.7, 3.8, 4.9, PURPLE)
textbox(s, "Top-10 sets largely coincide; ordering is method-dependent. Quote the set, not the rank.", 1.0, 6.75, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER)

# A4 — LIME fidelity
s = make_slide(prs); title(s, "Is the local explanation faithful?", "A4 · LIME fidelity")
picture(s, FIG / "lime_individual.png", .4, 1.75, 12.5)
textbox(s, "Category-aware perturbation raised local fidelity from R² 0.25 to ≈0.41. Three fixed cases × three seeds; every condition keeps its sign.",
        1.0, 6.75, 11, .4, 12, ORANGE, True, PP_ALIGN.CENTER)

# A5 — Improvement journey
s = make_slide(prs); title(s, "How we got here: what we tried, where we landed", "A5 · Process")
picture(s, FIG / "improvement_journey.png", .55, 1.75, 12.2)
textbox(s, "Full log in docs/JOURNEY.md — every step measured, including the attempts we rejected.", 1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER)

# A6 — proxy recovery
s = make_slide(prs); title(s, "Does excluding an attribute remove it?", "A6 · Proxy recovery")
picture(s, FIG / "proxy_recovery.png", .4, 1.7, 8.4)
panel(s, "MEASURED", f"Predict the attribute from our own features\n\n"
      f"With missingness: gender AUC {leak_auc:.2f} — the leak, quantified\n\n"
      f"Shipped set: gender {gender_proxy_auc:.2f}, race {race_proxy_auc:.2f}\n\n"
      "Exclusion is not removal", 9.1, 1.7, 3.8, 4.9, RED)
textbox(s, "Proxies are gun charges, mental-health/substance conditions, violent arrests, age — the substance of the assessment, not incidental fields.",
        .8, 6.78, 11.7, .4, 11, ORANGE, True, PP_ALIGN.CENTER)

# A7 — per-person stability and abstention
s = make_slide(prs); title(s, "Would this person's offer survive a different sample?", "A7 · Decision stability")
picture(s, FIG / "individual_stability.png", .35, 1.7, 12.6)
textbox(s, f"{contested_share:.0%} of decisions flip between refits, and about {contested_of_selected:.0%} of those actually prioritised sit at that margin. "
        f"Abstaining on them raises precision and WIDENS the gender gap: women are concentrated at the margin.",
        .8, 6.72, 11.7, .45, 11, ORANGE, True, PP_ALIGN.CENTER)

# ---------------------------------------------------------------- SPEAKER NOTES
# Written into the notes pane so they appear in presenter view and travel with the
# file. Timings total 14 minutes across the 15 core slides, leaving a one-minute
# buffer in the 15-minute slot. Appendix slides carry the cue for when to jump to them.
NOTES = {
 1: """[0:00 · 30s] Opening.

Team 11. Our client is a software vendor selling risk-assessment tools to US
community-supervision agencies. The question they brought us: which three-year
re-arrest model should they ship?

Say the frame once, here: we judged four dimensions — performance,
interpretability, stability, fairness — not accuracy. The 0.60 to 0.73 on the
right is our models against the score agencies already have on file.""",

 2: """[0:30 · 45s] The decision being sold.

The score prioritises VOLUNTARY re-entry support at the start of parole:
employment, housing, treatment. Never sanctions, never detention, never
surveillance.

Land the ethical scoping here, because everything later depends on it. Our model
is wrong often — at the top 20%, about one in six people offered support are not
re-arrested. For OFFERING HELP that is an acceptable cost. For punishment it
would not be. If a juror later challenges the error rate, this is the answer.""",

 3: """[1:15 · 1m15s] Data design, and the leak.

NIJ's own split — we did not make a new one. 18,028 train, 7,807 evaluation,
29 fields available at supervision start.

Two things to say plainly rather than bury:

First, the evaluation set is NOT an untouched holdout. It is excluded from
fitting, but our development decisions inspected it. Our comparisons are
exploratory. Say it before anyone asks.

Second, the leak — this is our strongest slide. Gang affiliation is missing for
every woman and no man. The missingness pattern IS gender. TabICL encodes NaN as
a category, so it could read the attribute we thought we had excluded. We measured
the cost: with missingness indicators, gender is recoverable at AUC 1.00.
Perfectly. We fixed it with mode-filling and a regression test.""",

 4: """[2:30 · 45s] Three model families.

White-box, machine learning, foundation model — as the brief requires. Same 29
fields, same held-out people, same metrics.

Logistic is CV-tuned L1. XGBoost ordinal-encodes ordered counts, hyperparameters
from a 5-fold CV random search that is committed as code, not asserted. TabICL is
a 16-member GPU ensemble.

Flag the one that matters for later: TabICL has NO native attribution path. That
is a deployment cost, and it comes back on the recommendation slide.""",

 5: """[3:15 · 1m00s] The client's real question.

Not "which of your three models" — that is our question. Theirs is: is any of this
better than what we already run?

This dataset contains Georgia's recorded supervision score, which we use as one
input. On its own it reaches 0.60 AUC. Our models reach 0.73. At 20% capacity that
roughly doubles the net value of the scenario.

Give the caveat in the same breath: this is a HISTORICAL recorded score. It does
not tell us how today's commercial products perform. And the dollars are a
scenario — effectiveness is assumed, not measured.""",

 6: """[4:15 · 1m15s] FINDING ONE. Performance cannot pick the model.

All three are within 0.003 AUC. XGBoost beats logistic by +0.0025 with a paired
interval excluding zero — real, but tiny.

Two methodological points worth making, because they show the work:

We use PAIRED bootstrap differences. Comparing two separate intervals for overlap
is weaker and commonly misread.

ECE differences are not significant, and the ECE ranking is not even stable across
binning choices. We tested calibration bin-free instead: logistic and XGBoost are
both indistinguishable from perfect calibration. TabICL is the exception — slope
0.913, significantly below 1. Its probabilities are too extreme.

So: performance does not discriminate. Interpretability, stability and fairness do.""",

 7: """[5:30 · 1m00s] Interpretability: can we explain a decision to the person it affects?

The brief asks for two things: explain individual predictions, and the main
drivers of the model. Left: one person, explained by logistic and by XGBoost.
Red raises risk, green lowers it.

Then the three routes, because they differ in kind, not degree:
Logistic: the coefficients ARE the explanation, plus marginal effects in
probability points. No second tool.
XGBoost: needs SHAP. A depth-3 surrogate tree only reproduces part of it, so a
"simple summary" of XGBoost would mislead.
TabICL: no native attribution at all. Only model-agnostic PDP/ICE. That is a
deployment cost, and it comes back on the recommendation.

Honest line: SHAP, permutation importance and XPER agree on WHICH variables drive
the model, not on their order. Quote the set, not the rank. Appendix A3 and A4
have the disagreement and the LIME fidelity check.""",

 8: """[6:30 · 45s] Stability: would a different sample give the same model?

The course definition: two samples from the same population should give roughly
the same model. We refit every model on the same eight bootstrap resamples and
compare all 28 pairs.

Logistic drifts least on every one of the 28 pairs, and its selected top 20%
changes least. This is one of the two reasons we recommend it.

Then one person at a time: about one decision in seven flips between refits.
Appendix A7 shows why sending those borderline cases to a human would widen the
gender gap.

Caveat if asked: Jaccard is intersection over union, not the share of people who
switch, and refit sensitivity is not validation on a new cohort.""",

 9: """[7:15 · 1m45s] FINDING TWO. Unequal access to support.

Slow down here. This is the centre of the talk.

First the framing: selection means being OFFERED support. So the harm is a missed
offer, and FNR — equal opportunity — is our primary metric, not FPR.

Second, the operating point. We audit at the top-20% rule the product actually
ships, not at threshold 0.5. At 0.5 the tool would flag 68% of people. That is not
a threshold this product ever applies, and auditing there inflates every gap.

The results:
— Race: equivalent within ±5 points for all three models by TOST. No race test
  survives Holm correction across our 54 tests. Small, and unconfirmed.
— Gender: women who ARE re-arrested miss support 10 to 12 points more often.
  Significant for every model.
— Age: the largest disparity, 23 to 25 points. About 97% of re-arrested people
  aged 48+ are never selected. It persists even with the age field held fixed,
  because prior record carries age.

Say the honest line: not significant does NOT mean fair. That is why we test
equivalence, not just difference.""",

10: """[9:00 · 45s] The impossibility result, both sides.

Our data demonstrates both halves of the theorem within one project.

Race: base rates 0.582 and 0.564 — nearly equal. The theorem does not bind, so
those gaps are a property of our models and are fixable.

Gender: 0.591 against 0.454, fourteen points apart. Calibration and equal error
rates CANNOT both hold. We have to choose.

The mechanism is visible in our numbers: every model over-predicts women — mean
score 0.52 against an observed 0.454 — yet women are selected less at the top 20%.

And for THIS use, the direction matters. In a support programme over-prediction
works in women's favour. Recalibrating by gender would select even fewer women and
widen the gap. The harm here is the missed offer, not the miscalibration.""",

11: """[9:45 · 1m00s] Where the gender gap comes from.

We did not stop at measuring. The course sequence is test, identify the variable,
mitigate — and fairness partial dependence names the candidate.

Gang affiliation. Cramér's V of 1.0 with gender on the raw field, because it is
never recorded for women. It is a measurement artefact about record-keeping, not a
behavioural signal about women.

Dropping it and re-estimating shrinks the gender FNR gap from -0.10 to about zero
for logistic — partly by raising men's FNR, so say that too — at roughly 0.014 AUC
and 27 to 41 fewer captured re-arrests. Statistical parity is still rejected.""",

12: """[10:45 · 1m00s] Did the fix generalise, or did we just fit it?

This is the slide that separates us from a project that stops at the p-value.

The result on the previous slide chose the variable AND measured the improvement on
the same cohort. That is in-sample. It shows a mitigation can be FITTED, not that
it works.

So we separated them. Five outer folds, entirely inside the training partition. In
each fold we rerun the candidate search on that fold's training part, pick from
there alone, refit there, and measure on the held-out fold. The evaluation cohort
is never touched.

Gang affiliation is selected in all five folds, for both models. About 70% of the
gap closes out of fold, for roughly one AUC point. The equal-opportunity p-value
reaches 0.27, not the in-sample 0.99 — and two of five folds still reject.

We report the out-of-fold numbers. The in-sample figure is an upper bound.""",

13: """[11:45 · 45s] The four dimensions together.

Read across the rows, not down the columns.

Performance: separated by 0.003 AUC, and the only real calibration difference goes
AGAINST the model with the highest AUC.

Interpretability: unambiguous, and the gaps are large.

Stability: logistic drifts least on all 28 resample pairs.

Fairness: the differences BETWEEN models are smaller than they look. The gender and
age gaps are shared by all three.

That is the shape of the decision: accuracy does not discriminate here.""",

14: """[12:30 · 1m00s] FINDING THREE. The recommendation, with its counter-case.

Logistic regression for the shadow pilot. XGBoost as challenger, running in
parallel on the same cohort.

Be explicit about what we give up: XGBoost is ahead on AUC and Brier. Those are
real, significant differences.

They do not reach the decision. At the deployed rule the two models offer support
to 85% of the SAME people — 254 of 7,807 differ — and the difference in captured
re-arrests has an interval spanning zero.

State what we do NOT claim, because a juror will test it: logistic is not fairer
and not better calibrated. On both, the two are statistically tied. We chose it
for refit stability and for coefficients you can read without a second tool.

And what would reverse it: if the client quotes probabilities numerically to
supervisees, calibration starts to dominate. If their capacity is large enough that
twelve extra offers matters. That is why the challenger runs in parallel.""",

15: """[13:30 · 30s] The app, and close.

The application makes every trade-off testable: score one person with all three
models, see the explanation, audit by race, gender and age at the deployed point,
and edit the cost assumptions.

One thing to demo if there is time: pick a person and show the refit vote count.
About one decision in seven flips across refits, and a third of the people we
prioritise sit at that margin. The app shows that rather than hiding it behind a
point estimate.

Close on the frame: we are not trading accuracy for interpretability. The accuracy
difference does not change who gets help.

[APPENDIX CUES — jump to these on the matching question]
A1 learning curve · A2 economics and sensitivity
A3 why SHAP, permutation importance and XPER disagree · A4 LIME fidelity
A5 what we tried and rejected · A6 proxy recovery
A7 per-person stability and the abstention finding""",
}

for index, slide in enumerate(prs.slides, start=1):
    if index in NOTES:
        slide.notes_slide.notes_text_frame.text = NOTES[index]

REPORTS.mkdir(exist_ok=True)
prs.save(OUT)
print(f"{OUT}  ({len(prs.slides)} slides, speaker notes on {len(NOTES)})")
