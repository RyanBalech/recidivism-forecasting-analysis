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


def card(slide, label, value, x, y, color=ORANGE, note="", size=24):
    shape = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(2.65), Inches(1.38))
    shape.fill.solid(); shape.fill.fore_color.rgb = CARD; shape.line.color.rgb = color
    textbox(slide, value, x+.12, y+.15, 2.4, .48, size, color, True, PP_ALIGN.CENTER)
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


# ================================================================ P1 · INTRO + EDA (1.5 min)
# 1 — Title
s = make_slide(prs)
textbox(s, "TRUSTWORTHY AI / RECIDIVISM", .72, .58, 8, .3, 12, ORANGE, True)
textbox(s, "Forecasting risk,\nexamining trade-offs", .72, 1.42, 8, 1.65, 36, WHITE, True)
textbox(s, "For a risk-assessment software vendor: which three-year re-arrest model to ship, judged on performance, interpretability, stability and fairness", .76, 3.35, 6.6, 1.4, 17, PALE)
textbox(s, "Team 11 · HEC Paris · Fall 2026", .76, 6.62, 7, .3, 11, GREY)
panel(s, f"{inc_auc['incumbent']:.2f} → {metrics.roc_auc.max():.2f}", "Historical score vs our models (ROC AUC)\n\n3 model families\n\n1 recommendation", 8.55, .9, 3.65, 5.75, PURPLE)

# 2 — Client and decision
s = make_slide(prs); title(s, "Our client, and the decision they sell", "Intro · Client")
bullets(s, ["Client: a vendor selling risk tools to US community-supervision agencies",
            "Decision: rank people at supervision start; the top 20% are offered voluntary re-entry support",
            "Scope: support allocation only — never sanctions, detention or surveillance",
            "Target: recorded re-arrest within 3 years — a proxy for need, not for offending"], y=1.85, size=21); footer(s, 2)

# 3 — Data and EDA
s = make_slide(prs); title(s, "Georgia parolees, official split, training-only EDA", "Intro · Data")
picture(s, FIG / "eda_overview.png", .4, 1.7, 9.0)
_raw = pd.read_csv(ROOT / "data" / "nij-challenge2021_full_dataset.csv", usecols=["Training_Sample", "Recidivism_Within_3years"])
train_rate = _raw.loc[_raw.Training_Sample == 1, "Recidivism_Within_3years"].astype(str).eq("True").mean() if _raw.Recidivism_Within_3years.dtype == bool else _raw.loc[_raw.Training_Sample == 1, "Recidivism_Within_3years"].eq("Yes").mean()
card(s, "RE-ARRESTED IN 3 YEARS", f"{train_rate:.1%}", 9.95, 1.8, ORANGE, "training data, as in the chart")
card(s, "OFFICIAL SPLIT", "18,028 / 7,807", 9.95, 3.4, TEAL, "train / evaluation · NIJ flag", size=18)
card(s, "GANG FIELD MISSING", "100% of women", 9.95, 5.0, RED, "and 0% of men → next part", size=20)
textbox(s, "Cumulative 3-year target: not comparable with NIJ's annual leaderboard. The missingness is structured.",
        1.0, 6.85, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 3)

# ================================================================ P2 · FEATURES + LEAKAGE (1.5 min)
# 4 — Only information known when supervision starts
s = make_slide(prs); title(s, "Only information known when supervision starts", "Features · Eligibility")
panel(s, "IN THE MODEL", "29 baseline fields\n\nAge, prior arrests and convictions, offence, prison years, supervision level, Georgia's risk score",
      .7, 1.8, 3.85, 4.7, TEAL)
panel(s, "AUDIT ONLY", "Race, gender, residence (PUMA)\n\nNever model inputs — kept to measure who gets support",
      4.75, 1.8, 3.85, 4.7, PURPLE)
panel(s, "EXCLUDED", f"Everything after release: employment, drug tests, violations\n\nPositive control: adding them lifts AUC {posctrl_base:.3f} → {posctrl_leaky:.3f} — they leak the outcome",
      8.8, 1.8, 3.85, 4.7, RED)
footer(s, 4)

# 5 — The leak
s = make_slide(prs); title(s, "The leak we found: gender was encoded in missingness", "Features · Leakage")
picture(s, FIG / "proxy_recovery.png", .4, 1.7, 8.4)
panel(s, "WHAT HAPPENED", "Gang affiliation: missing for every woman, no man\n\n"
      f"Keep missingness → gender recovered at AUC {leak_auc:.3f}\n\n"
      "TabICL treated NaN as a category: it saw gender\n\n"
      f"Fix: mode-fill + regression test. After: AUC {gender_proxy_auc:.2f}", 9.1, 1.7, 3.8, 4.9, RED)
textbox(s, "Excluding an attribute is not removing it: part of it survives in legitimate features. Fairness picks this up.",
        1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 5)

# ================================================================ P3 · MODELS + PERFORMANCE (2.5 min)
# 6 — Three models and tuning
s = make_slide(prs); title(s, "Three model families, each tuned by cross-validation", "Models · Design")
panel(s, "LOGISTIC (L1)", "White box\n\nOne-hot encoding\n\n5-fold grid search\nC = 0.2154\n\nCurve flat for C in [0.05, 1]", .7, 1.8, 3.85, 4.5, TEAL)
panel(s, "XGBOOST", "Machine learning\n\nOrdinal counts\n\n5-fold random search, 60 draws\n\nDeeper trees overfit: rejected", 4.75, 1.8, 3.85, 4.5, ORANGE)
panel(s, "TABICLv2", "Foundation model\n\nMixed types, mode-filled\n\n16-member ensemble\n\nGains level off near 8", 8.8, 1.8, 3.85, 4.5, PURPLE)
textbox(s, f"{ml_n} other ML candidates (CatBoost, LightGBM, EBM, …) all land at test AUC {ml_lo:.3f}–{ml_hi:.3f}.",
        1.0, 6.62, 11, .3, 13, ORANGE, True, PP_ALIGN.CENTER); footer(s, 6)

# 7 — Performance
s = make_slide(prs); title(s, "Statistical performance: effectively a tie", "Models · Performance")
picture(s, FIG / "performance_calibration.png", .45, 1.72, 8.05)
card(s, "AUC", slash([metrics.loc[m, "roc_auc"] for m in m3], "{:.3f}"), 9.25, 1.9, ORANGE, "logistic / XGBoost / TabICL", size=15)
card(s, "AUC GAP · XGB − LOGIT", f"+{xgb_minus_logit:.4f}", 9.25, 3.5, TEAL, "paired bootstrap: real, but small")
card(s, "CALIBRATION SLOPE", slash([_cal.loc[m, "slope"] for m in m3], "{:.2f}"), 9.25, 5.1, PURPLE, "TabICL too extreme (significant)", size=16)
textbox(s, "All three within ~0.003 AUC. Performance alone cannot pick the model.", 1.0, 6.85, 11, .3, 11, GREY, True, PP_ALIGN.CENTER); footer(s, 7)

# 8 — Incumbent
s = make_slide(prs); title(s, "The client's real question: better than today's score?", "Models · Incumbent")
picture(s, FIG / "incumbent_benchmark.png", .45, 1.75, 8.3)
card(s, "HISTORICAL SCORE AUC", f"{inc_auc['incumbent']:.2f}", 9.25, 1.9, RED, "Georgia supervision score in the data")
card(s, "OUR MODELS AUC", f"≈ {metrics.roc_auc.mean():.2f}", 9.25, 3.5, TEAL, "all three models")
card(s, "NET VALUE @20%", f"${inc_econ['incumbent']/1e6:.2f}M → ${inc_econ['logistic']/1e6:.2f}M", 9.25, 5.1, ORANGE, "scenario: $5k cost, $50k event, 20% effect", size=18)
textbox(s, "Models beat the historical score at every capacity. Dollars are a scenario, not a causal estimate.",
        1.0, 6.85, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 8)

# ================================================================ P4 · INTERPRETABILITY (2 min)
# 9 — Global drivers
s = make_slide(prs); title(s, "What drives the score", "Interpretability · Global")
picture(s, FIG / "shap_global.png", .4, 1.7, 8.6)
card(s, "AGE 23–27 → 48+", f"{age_contrast*100:+.0f} pts", 9.35, 1.9, TEAL, "logistic, average probability change")
card(s, "GANG AFFILIATION", f"{gang_contrast*100:+.0f} pts", 9.35, 3.5, RED, "recorded 'Yes' vs 'No'")
card(s, "TOP DRIVERS", "Age · gang · priors", 9.35, 5.1, ORANGE, "same in logistic and XGBoost", size=17)
textbox(s, "Marginal effects in probability points, not log-odds: the unit a caseworker understands.",
        1.0, 6.85, 11, .3, 11, GREY, True, PP_ALIGN.CENTER); footer(s, 9)

# 10 — Model-agnostic view: PDP/ICE for all three models, and what each model costs to explain
s = make_slide(prs); title(s, "Looking from outside: PDP/ICE for all three models", "Interpretability · Model-agnostic")
picture(s, FIG / "pdp_ice.png", .4, 1.7, 8.3)
panel(s, "THREE ROUTES", "LOGISTIC\nCoefficients: read directly\n\n"
      f"XGBOOST\nNeeds SHAP; a depth-3 surrogate tree reproduces only R² = {surrogate_r2:.2f}\n\n"
      "TABICLv2\nNo native attribution: PDP/ICE is the only view", 9.0, 1.7, 3.9, 4.9, ORANGE)
textbox(s, "Black line = average effect (PDP); faint lines = individual people (ICE). All three models agree: risk falls with age.",
        .8, 6.85, 11.7, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 10)

# 11 — One person, and a faithfulness check
s = make_slide(prs); title(s, "Explaining one person, and checking the explanation", "Interpretability · Local")
picture(s, FIG / "shap_individual.png", .4, 1.7, 8.3)
panel(s, "IS IT FAITHFUL?", "SHAP: exact for these two models\n\n"
      f"LIME on raw features: local R² 0.25 → {lime_lo:.2f}–{lime_hi:.2f}\n\n"
      "3 cases × 3 seeds: all 63 conditions keep their sign\n\n"
      "Quote LIME for direction, SHAP for size", 9.0, 1.7, 3.9, 4.9, PURPLE)
textbox(s, "Red raises risk, green lowers it. The explanation is what an appeal would contest.",
        1.0, 6.85, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 11)

# 12 — Explaining performance: XPER (course method) and agreement with SHAP / permutation importance
s = make_slide(prs); title(s, "Explaining performance: where does the AUC come from?", "Interpretability · XPER")
picture(s, FIG / "xper.png", .4, 1.7, 8.4)
panel(s, "XPER", f"Splits the AUC itself into feature contributions (Shapley values)\n\n"
      f"Benchmark ≈ {xper_bench:.2f} (no information) + features = AUC\n\n"
      f"Age alone adds ≈ {xper_age.mean():.2f} AUC\n\n"
      f"Agrees with SHAP / permutation importance on the set (Spearman {agree_lo:.2f}–{agree_hi:.2f}), not the order", 9.1, 1.7, 3.8, 4.95, PURPLE)
textbox(s, "SHAP explains a prediction; permutation importance explains loss; XPER explains performance. Quote the set of drivers, not the rank.",
        .8, 6.85, 11.7, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 12)

# ================================================================ P5 · FAIRNESS (3 min)
# 12 — Race (1)
s = make_slide(prs); title(s, "Race (1): excluding race is not the same as being fair", "Fairness · Race")
panel(s, "BASE RATES", f"Black {calib.loc[('Race', 'BLACK'), 'base_rate']:.3f}\nWhite {calib.loc[('Race', 'WHITE'), 'base_rate']:.3f}\n\nNearly equal: the data does not force a gap",
      .7, 1.8, 3.85, 4.5, TEAL)
panel(s, "WHAT EXCLUSION GUARANTEES", f"Twins differing only in race get the same score\n\nAdding race back changes AUC by ≤ {race_ab_max:.3f}",
      4.75, 1.8, 3.85, 4.5, PURPLE)
panel(s, "WHAT IT DOES NOT", f"Race is recoverable from our features at AUC {race_proxy_auc:.2f}\n\nCriminal history carries part of it — and cannot be dropped",
      8.8, 1.8, 3.85, 4.5, RED)
textbox(s, "Selection = support offered, so the harm is missed support: FNR is our primary metric. We audit outcomes, not inputs.",
        .8, 6.62, 11.7, .3, 13, ORANGE, True, PP_ALIGN.CENTER); footer(s, 13)

# 13 — Race (2)
s = make_slide(prs); title(s, "Race (2): the outcome audit, with three caveats", "Fairness · Race")
picture(s, FIG / "fairness_race_gaps.png", .4, 1.7, 8.3)
panel(s, "RESULT + CAVEATS", f"All race gaps within ±5 pts (TOST); none survives Holm\n\nSelection ratio W/B {slash(race_ratio, '{:.2f}')} (> 4/5)\n\n"
      "1. Direction: slightly MORE Black people offered support\n2. Label: recorded arrest may carry policing bias\n"
      f"3. AUC Black {min(race_auc[0::2]):.2f} vs White {min(race_auc[1::2]):.2f}", 9.0, 1.7, 3.95, 4.95, TEAL)
footer(s, 14)

# 14 — Gender (1)
s = make_slide(prs); title(s, "Gender (1): women are over-predicted, yet selected less", "Fairness · Gender")
picture(s, FIG / "fairness_gender_gaps.png", .4, 1.7, 8.3)
panel(s, "WHY", f"FNR gap M − F {span(fnr20['Gender'])}, all models: the feature set, not one model\n\n"
      f"Women's mean score {calib.loc[('Gender', 'F'), 'mean_score_logistic']:.2f} vs observed {calib.loc[('Gender', 'F'), 'base_rate']:.3f}\n\n"
      f"Base rates {calib.loc[('Gender', 'M'), 'base_rate']:.2f} vs {calib.loc[('Gender', 'F'), 'base_rate']:.2f}: calibration and equal FNR cannot both hold",
      9.0, 1.7, 3.95, 4.95, RED)
textbox(s, f"Age: the largest gap ({span(fnr20['Age'])}), but by design: a validated risk factor. Risk vs need is the client's choice (A6).",
        .6, 6.82, 12.1, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 15)

# 15 — Gender (2)
s = make_slide(prs); title(s, "Gender (2): cause located, mitigated, validated out of sample", "Fairness · Gender")
picture(s, FIG / "fpdp_gender_focus.png", .35, 1.9, 8.4)
panel(s, "FPDP → GANG", "Never recorded for women: record-keeping, not behaviour\n\n"
      f"Drop it, refit inside {int(nested_sel * 100)}% of 5 training folds\n\n"
      f"Out of fold: FNR gap {nested_base:+.3f} → {nested_mit:+.3f} (~70% closed), AUC {nested_auc:+.3f}\n\n"
      "Cost: men's FNR +2 pts; selection rates still differ", 9.0, 1.65, 3.95, 5.0, TEAL)
textbox(s, "Evaluation cohort untouched. A mitigation option for the client, not applied to the recommended model.",
        1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 16)

# ================================================================ P6 · STABILITY + TRADE-OFFS + RECOMMENDATION (2.5 min)
# 16 — Structural stability
s = make_slide(prs); title(s, "Would a different sample give the same model?", "Stability · Structural")
picture(s, FIG / "structural_stability.png", .5, 1.75, 8.3)
card(s, "SCORE DRIFT (mean |Δp|)", slash([stability.loc[m, "mean_abs_prob_diff"] for m in m3]), 9.35, 1.9, ORANGE,
     "logistic below XGBoost on all 28 pairs", size=15)
card(s, "TOP-20% JACCARD", slash([stability.loc[m, "top20_jaccard"] for m in m3], "{:.2f}"), 9.35, 3.5, TEAL,
     "≈ 13% of the selected change per refit", size=16)
card(s, "PROTOCOL", "8 refits", 9.35, 5.1, PURPLE, "same bootstrap resamples for all 3")
textbox(s, "Course definition: two samples from the same population should give approximately the same model.",
        1.0, 6.82, 11, .3, 11, GREY, True, PP_ALIGN.CENTER); footer(s, 17)

# 17 — Per-person stability and abstention
s = make_slide(prs); title(s, "Stability for one person: abstaining is not fairness-neutral", "Stability · Individual")
picture(s, FIG / "individual_stability.png", .35, 1.7, 12.6)
textbox(s, f"{contested_share:.0%} of decisions flip. Unanimous only: precision {abst_full.precision_at_capacity:.3f} → {abst_strict.precision_at_capacity:.3f}, "
        f"gender FNR gap {abst_full.fnr_gap_gender:+.3f} → {abst_strict.fnr_gap_gender:+.3f}.",
        .8, 6.72, 11.7, .45, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 18)

# 18 — Trade-off matrix and recommendation
s = make_slide(prs); title(s, "Pilot logistic regression; run XGBoost as the challenger", "Conclusion · Trade-offs")
picture(s, FIG / "tradeoff_matrix.png", .35, 1.6, 7.7)
panel(s, "WHY LOGISTIC", f"Same people: {overlap_share:.0%} overlap with XGBoost; captured re-arrests CI includes 0\n\n"
      "More stable than XGBoost on all 28 refit pairs\n\nCoefficients read directly · ~9× faster\n\n"
      "Fairness and calibration: tied — not reasons\n\nReverses if: scores are quoted numerically, or scale makes +12 offers matter",
      8.4, 1.6, 4.5, 5.1, TEAL)
textbox(s, "Shadow pilot first: no temporal/external validation yet, and no evidence the support programme works.",
        .8, 6.85, 11.7, .3, 11, ORANGE, True, PP_ALIGN.CENTER); footer(s, 19)

# 19 — App
s = make_slide(prs); title(s, "The application makes every trade-off testable", "Demo · App")
panel(s, "1 · SCORE", "One person, all 3 models", .75, 1.75, 5.7, 1.72, ORANGE)
panel(s, "2 · EXPLAIN", "Why this score (SHAP)", 6.85, 1.75, 5.7, 1.72, PURPLE)
panel(s, "3 · STABILITY", "How many of 8 refits select them", .75, 3.8, 5.7, 1.72, ORANGE)
panel(s, "4 · WHAT IF", "Edit an input, watch the score move", 6.85, 3.8, 5.7, 1.72, PURPLE)
textbox(s, "streamlit run app.py   ·   backup: screenshots / recording", 3.2, 6.25, 7, .4, 15, PALE, True, PP_ALIGN.CENTER, "Consolas"); footer(s, 20)

# ---------------------------------------------------------------- APPENDIX (Q&A only)
s = make_slide(prs)
textbox(s, "APPENDIX", .72, 2.6, 8, .5, 14, ORANGE, True)
textbox(s, "Supporting evidence\nfor questions", .72, 3.15, 9, 1.5, 34, WHITE, True)
textbox(s, "Learning curve · economic sensitivity · explanation disagreement · global surrogate · threshold frontier · age · process log", .76, 4.95, 9.5, .9, 15, PALE)

s = make_slide(prs); title(s, "Which model for which agency size?", "A1 · Learning curve")
picture(s, FIG / "learning_curve.png", .8, 1.8, 11.7)
textbox(s, "Subsamples of one Georgia cohort test sample-size sensitivity; they do not validate transfer to other agencies.", 1.0, 6.7, 11, .5, 13, ORANGE, True, PP_ALIGN.CENTER)

s = make_slide(prs); title(s, "Model value is a scenario, stress-tested", "A2 · Economic sensitivity")
card(s, "SERVICE CAPACITY", "20%", .8, 1.9)
card(s, "LOGISTIC NET", f"${inc_econ['logistic']/1e6:.2f}M", 3.75, 1.9, TEAL, f"XGBoost ${inc_econ['xgboost']/1e6:.2f}M")
card(s, "HISTORICAL SCORE NET", f"${inc_econ['incumbent']/1e6:.2f}M", 6.7, 1.9, RED); card(s, "RANDOM NET", f"${inc_econ.get('random', 1.46e6)/1e6:.2f}M", 9.65, 1.9, GREY)
bullets(s, ["$5,000 support cost · $50,000 event cost · 20% assumed effectiveness — all editable in the app",
            "Models beat the historical score at every capacity from 5% to 50%",
            "A randomized or quasi-experimental pilot must estimate real intervention impact"], y=3.72, size=20)

s = make_slide(prs); title(s, "SHAP, permutation importance and XPER rank differently", "A3 · Explanation disagreement")
picture(s, FIG / "explanation_agreement.png", .6, 1.75, 12.0)
textbox(s, f"Spearman {agree_lo:.2f}–{agree_hi:.2f}; 8–10 of the top ten features shared. The set of drivers is robust, the order is method-dependent.",
        1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER)

s = make_slide(prs); title(s, "A depth-3 tree that mimics XGBoost", "A4 · Global surrogate")
picture(s, FIG / "global_surrogate.png", .5, 1.75, 12.3)
textbox(s, f"Test fidelity R² = {surrogate_r2:.2f}: readable, but it misses much of what XGBoost does. A surrogate can give an illusion of interpretability.",
        1.0, 6.82, 11, .3, 11, RED, True, PP_ALIGN.CENTER)

s = make_slide(prs); title(s, "Group thresholds: the fairness/utility frontier", "A5 · Threshold frontier")
picture(s, FIG / "fairness_frontier.png", .5, 1.75, 12.3)
textbox(s, "Optimised on the evaluation set, so optimistic. Group-specific thresholds by race or gender are disparate treatment.",
        1.0, 6.82, 11, .3, 11, RED, True, PP_ALIGN.CENTER)

s = make_slide(prs); title(s, "Age: the largest gap, and a policy choice", "A6 · Age")
picture(s, FIG / "fpdp_age_focus.png", .35, 1.9, 8.4)
panel(s, "WHY IT IS DIFFERENT", f"FNR gap <33 − 33+: {span(fnr20['Age'])}\n\n"
      f"Re-arrested 48+: {age_old_fnr:.0%} not selected, vs {age_young_fnr:.0%} at 18–22\n\n"
      "Age is a direct, validated risk factor: the gap is by design\n\n"
      "FPDP finds no candidate: age runs through criminal history", 9.0, 1.65, 3.95, 5.0, ORANGE)
textbox(s, "Ranking by risk or reserving places by age is the client's decision; the course's mitigation has nothing to act on.",
        .8, 6.82, 11.7, .3, 11, ORANGE, True, PP_ALIGN.CENTER)

s = make_slide(prs); title(s, "How we got here: what we tried, where we landed", "A7 · Process")
picture(s, FIG / "improvement_journey.png", .55, 1.75, 12.2)
textbox(s, "Full log in docs/JOURNEY.md — every step measured, including the attempts we rejected.", 1.0, 6.82, 11, .3, 11, ORANGE, True, PP_ALIGN.CENTER)

# ---------------------------------------------------------------- SPEAKER NOTES
# Timings follow reports/presentation_outline.md: 14:25 of talk across 20 core slides,
# leaving a buffer in the 15-minute slot. [P1]…[P6] marks who speaks.
NOTES = {
 1: """[P1 · 0:00 · 20s] Team 11. Our client sells risk-assessment tools to US community-supervision agencies. Their question: which three-year re-arrest model should they ship? We judged four dimensions — performance, interpretability, stability, fairness — not accuracy alone.""",
 2: """[P1 · 0:20 · 30s] The score prioritises VOLUNTARY re-entry support: employment, housing, treatment. Never sanctions or detention. Everything later depends on this: being selected means being offered help. Our target is recorded re-arrest, which is only a proxy for need.""",
 3: """[P1 · 0:50 · 40s] NIJ's own split: 18,028 training, 7,807 evaluation, Georgia 2013-2015. About 58% are re-arrested. Black and White men have almost the same rate; women are lower. On the right, missing values: the gang field is missing for EVERY woman and no man. The missingness is structured — over to [P2].""",
 4: """[P2 · 1:30 · 40s] We only use what is known when supervision starts: 29 fields. Race, gender and residence are never inputs; we keep them to audit who gets support. Everything recorded after release is excluded — and we proved why: adding those fields lifts AUC from 0.73 to 0.81, because they leak the outcome.""",
 5: """[P2 · 2:10 · 50s] The leak we found. Gang affiliation is missing for every woman. If missingness is kept, gender is recovered from our own inputs at AUC 1.00 — perfectly. TabICL treats a missing value as its own category, so it effectively saw gender. We fixed it with mode-filling and a regression test. But even after the fix, gender is still recoverable at 0.78: excluding an attribute is not removing it. [P5] picks this up.""",
 6: """[P3 · 3:00 · 50s] Three families, as the brief requires, each tuned by cross-validation on training data only. Logistic with L1, C chosen by grid search. XGBoost with a 60-draw random search; deeper trees overfit and were rejected. TabICL with 16 ensemble members; gains level off near 8. Six other ML models all land in the same narrow AUC band.""",
 7: """[P3 · 3:50 · 50s] Finding one: performance cannot pick the model. All three within 0.003 AUC. XGBoost beats logistic by 0.0025 on a PAIRED bootstrap — real, but small. Calibration: logistic and XGBoost are indistinguishable from perfect; TabICL's slope is 0.91, significantly below 1 — its probabilities are too extreme.""",
 8: """[P3 · 4:40 · 50s] The client's real question: is this better than the score agencies already have? The historical Georgia score in the data reaches 0.60 AUC; our models reach 0.73, and roughly double the scenario net value at 20% capacity. Caveats in the same breath: it is a historical score, and the dollars are a scenario. Over to [P4].""",
 9: """[P4 · 5:30 · 40s] What drives the score. Age at release, gang affiliation and prior record, in both models. In probability points: moving from age 23-27 to 48+ lowers predicted risk by about 27 points; a recorded gang affiliation raises it by about 17.""",
 10: """[P4 · 6:10 · 40s] Now from the outside, without opening the model. Partial dependence shows the average effect of one feature; the faint individual curves show each person. All three models agree that risk falls with age. This is the only way we can look inside TabICL at all: it has no native attribution. XGBoost can be summarised by a small surrogate tree, but that tree reproduces only 61% of it. Logistic needs none of this: its coefficients are the explanation.""",
 11: """[P4 · 6:50 · 40s] One person, explained by both models. Red raises risk, green lowers it — this is what an appeal would contest. We checked faithfulness: LIME on raw features fits locally about twice as well as before, and all 63 conditions keep their sign across seeds. We quote LIME for direction and SHAP for size.""",
 12: """[P4 · 7:30 · 40s] Last, explaining PERFORMANCE rather than predictions — XPER, from this course. It splits the AUC itself into contributions: an uninformative model gets about 0.47, and each feature adds its share; age alone adds about 0.09. Compared with SHAP and permutation importance, the three methods agree on WHICH features matter but not on their order, because they answer different questions: prediction, loss, performance. So we quote the set, not the rank. Over to [P5].""",
 13: """[P5 · 8:10 · 45s] Fairness. Our primary metric is FNR: people later re-arrested but not offered support — the real harm here. Race first, the question everyone expects after COMPAS. Base rates are almost equal, so the data does not force a gap. Excluding race guarantees twins get the same score, but race is still recoverable at 0.71 through criminal history. So we audit outcomes, not inputs.""",
 14: """[P5 · 8:55 · 45s] The outcome audit: every race gap is within five points by an equivalence test, and none survives Holm correction. Three caveats: the small gap runs toward MORE support for Black people; the label is recorded arrest, which may carry policing bias; and prediction quality is lower for Black people, AUC 0.72 vs 0.75.""",
 15: """[P5 · 9:40 · 45s] Gender is where we found a problem. Re-arrested women miss support 10 to 12 points more often, in all three models — so it comes from the features. Every model over-predicts women, yet selects them less: with different base rates, calibration and equal FNR cannot both hold. Age shows an even larger gap, but by design — age is a validated risk factor — so it is the client's policy choice; details in the appendix.""",
 16: """[P5 · 10:25 · 45s] We located the cause with the course's FPDP: gang affiliation, never recorded for women. Dropping it, selected in every one of five training folds, closes about 70% of the gap out of sample for about one AUC point — at the cost of men's FNR rising two points. It is an option for the client, not applied to our recommended model. Over to [P6].""",
 17: """[P6 · 11:10 · 40s] Stability, in the course's sense: two samples from the same population should give the same model. Eight bootstrap refits, same resamples for all three models. Logistic drifts less than XGBoost on all 28 pairs. Jaccard around 0.77 means about 13% of the selected people change per refit.""",
 18: """[P6 · 11:50 · 50s] At the level of one person, about one decision in seven flips across refits. Referring contested cases to a human raises precision — but WIDENS the gender gap, because women sit at the margin more often. Abstention is not fairness-neutral.""",
 19: """[P6 · 12:40 · 60s] Reading across the four dimensions: performance is a tie; interpretability and stability favour logistic; fairness gaps are shared by all three. So: logistic for a shadow pilot, XGBoost as challenger. The two select 85% of the same people and the difference in captured re-arrests includes zero. We do NOT claim logistic is fairer or better calibrated. It reverses if the client quotes scores numerically, or operates at a scale where a dozen extra offers matter.""",
 20: """[P1 · 13:40 · 45s] Demo one person: scores from all three models, the explanation, how many of eight refits select them, then edit an input and watch the score move. If the app fails, switch to the screenshots.

[APPENDIX CUES] A1 learning curve · A2 economics · A3 explanation disagreement · A4 global surrogate · A5 threshold frontier · A6 age · A7 process""",
}

for index, slide in enumerate(prs.slides, start=1):
    if index in NOTES:
        slide.notes_slide.notes_text_frame.text = NOTES[index]

REPORTS.mkdir(exist_ok=True)
prs.save(OUT)
print(f"{OUT}  ({len(prs.slides)} slides, speaker notes on {len(NOTES)})")
