# End-to-end review — 24 September 2026

The published models remain reproducible. A fresh, isolated fit of logistic regression, XGBoost and TabICLv2 produced exactly the same saved probabilities as the published run. This review found no direct evaluation-label leakage in fitting. It strengthened the leakage gate, GPU evidence, artifact checks and stability comparison; it does not establish new-cohort validity.

## Brief, data and target

The supplied `Project ISAF 2026_2027.pdf` requires three model families, statistical/economic performance, interpretability, stability, fairness, a client recommendation, slides, code/notebook and an interactive app. These are present. The root-level `Recidivism_Project_Submission.ipynb` is the executed professor-facing entry point; its current code-cell count is recorded in the [machine-readable audit](../artifacts/end_to_end/audit.json). The course permits a binary target, so cumulative three-year arrest is within the brief. It differs from the [NIJ challenge's annual conditional forecasts](https://nij.ojp.gov/funding/recidivism-forecasting-challenge), and leaderboard comparison would be invalid.

The audit checks all 18,028 training and 7,807 evaluation IDs against the original releases, verifies all 29 baseline feature values, and now also compares cumulative and annual **training outcomes** with the original training release. Annual outcomes in the full dataset are mutually exclusive and reconstruct the cumulative outcome. There is no ID overlap. Identical coarsened feature patterns span 20 evaluation rows; excluding them does not materially change model ranking.

## Leakage and fitting boundaries

All prediction-model fits inspected use training rows or subsets of training. Imputation, scaling and encoding are fitted within pipelines; calibration sees training out-of-fold predictions. Both saved nested-experiment files cover training IDs exactly once and contain no official evaluation IDs. Negative tests inject forbidden fields and alter evaluation outcomes to check that fitting inputs remain unchanged.

The previous leakage command depended on an existing prediction CSV because it also computed a duplicate-pattern performance sensitivity. That blocked checking data before the first training run. The data gate now runs without predictions, and `train_evaluate.py` invokes it before fitting. Post-fit sensitivity remains available separately.

Gang-affiliation missingness is gender-aligned. Mode imputation removes the explicit missing category but cannot remove every proxy. Public files do not establish precise measurement dates for all initial fields. Evaluation labels have also been repeatedly inspected during development. Those limitations prevent a claim that the project is entirely leakage-free or that reported improvements are confirmed on unseen cohorts. The deliberately invalid post-release positive control and shuffled-label controls from the [earlier review](deep_review.md) remain diagnostic evidence, not a certification.

## GPU use and accuracy

The original manifest records TabICL on CUDA. The fresh run independently confirmed CUDA on an NVIDIA GeForce RTX 4050 Laptop GPU, with a 16-member ensemble. Logistic regression and the published XGBoost model use CPU. Native categorical XGBoost in the earlier accuracy review used CUDA.

A new three-fold experiment uses only training data, identical folds and the same fixed XGBoost parameters with four CPU threads. Actual fitted-booster configuration is checked, so silently falling back to CPU fails the GPU check. CPU mean AUC is 0.733143 and Brier 0.204018; CUDA mean AUC is 0.733064 and Brier 0.204014. This is no meaningful predictive improvement. The latest audit measured approximately 1.90 seconds on CPU versus 3.33 seconds on CUDA per fit plus prediction; timings vary with load. CPU-backed input requires XGBoost's DMatrix prediction fallback when the booster is on CUDA. These timings describe this pipeline, not an optimized GPU-resident implementation. [XGBoost documents CUDA histogram training and device-related memory behavior](https://xgboost.readthedocs.io/en/stable/gpu/).

The earlier 144 inner-fold fits already tested encoding, depth, regularization and native categories; calibration and CatBoost did not establish a material gain. The equal-weight XGBoost–TabICL blend remains a research challenger: slightly better Brier/AUC point estimates but uncertain improvement, fewer captured events at 20% capacity, and greater inference cost. No new model or device was chosen using official evaluation performance. XGBoost's tiny gap to TabICL is not evidence that it is malfunctioning.

The checkpoint name is now explicit in shared configuration. Future training manifests also record its SHA-256; the new audit records the locally cached checkpoint hash. The fresh-run probability comparison is exact in this environment, not a promise of equality on other hardware.

## Analysis improvements

- **Stability:** all families now use the same eight bootstrap samples with model seeds fixed. Previously the models received different samples and some algorithm seeds varied, confounding comparisons. Sample hashes and pair identifiers make the new experiment auditable. Pairwise comparisons share refits and cannot be treated as independent observations or temporal validation.
- **Fairness:** all 90 published gap estimates are recomputed from predictions and ID-aligned attributes. A supplementary Holm correction covers the family of 54 course difference tests across models and attributes. Raw tests remain available for course comparability. This adjustment does not correct TOST, candidate selection in FPDP or historical development choices. The ±5-point equivalence tolerance was chosen after gap estimates had been seen; equivalence is exploratory and conditional on that tolerance.
- **XPER:** inspection of installed XPER 0.0.92 shows its kernel branch fits unconstrained weighted least squares. The sampled coalitions exclude the empty and full sets, and the regression imposes no endpoint efficiency constraint. Thus the sum is not guaranteed to equal observed AUC. The recorded residuals of approximately +0.018 and +0.023 remain visible; the outputs should not be presented as an exact decomposition. This identifies a structural explanation for lack of reconstruction, without quantifying every source of approximation error.
- **Artifact integrity:** the final audit recomputes all published probability, threshold and economic metrics, checks saved conventional model predictions, verifies nested prediction IDs/labels, and records input/source hashes. A regression test deliberately corrupts Brier loss and confirms the audit rejects it. CSV rounding is allowed only within a small numerical tolerance.

Economic values still assume intervention effectiveness; arrest risk is not estimated treatment benefit. SHAP reconstruction tests check the conventional explanations, while PDP/ICE and feature edits remain associative. The race A/B experiment deliberately includes race in an audit-only candidate; it does not alter the published scoring inputs. Fairness mitigation and group-threshold searches remain exploratory because they use evaluation outcomes.

Holm correction reduces the 54 difference-test rejections from 41 to 32. No race difference test survives this family-wise adjustment; the large gender and age equal-opportunity disparities do. This is not evidence of zero race disparity, and it does not convert the separately unadjusted equivalence results into confirmatory findings.

## Evidence and reproduction

[Machine-readable audit](../artifacts/end_to_end/audit.json), [CPU/CUDA fold results](../artifacts/end_to_end/xgboost_cpu_gpu.csv), [Holm-adjusted tests](../artifacts/end_to_end/fairness_tests_holm.csv), [stability protocol](../artifacts/stability_protocol.json).

```powershell
python -m pytest -q
python scripts/leakage_audit.py
python scripts/train_evaluate.py --output-dir artifacts/smoke/end_to_end
python scripts/validate_project.py
python scripts/stability_structural.py
python scripts/tradeoff_matrix.py
python scripts/build_standalone_notebook.py
python scripts/build_slides.py
python scripts/end_to_end_audit.py --gpu-check --refit-dir artifacts/smoke/end_to_end
```

`scripts/reproduce.py` runs the audit before notebook generation to refresh the adjusted fairness table, then again after rebuilding deliverables to record their final hashes. Optional hardware/refit checks are requested with the flags above. This review reran core training, validation and revised stability; it did not rerun the historical hyperparameter searches, learning curves or expensive XPER coalitions whose inputs were unchanged. The app passed both default and live TabICL inference checks; 33 regression tests passed.

The most useful next accuracy evidence would be a genuinely new cohort with verified measurement times. For this submission, the stronger result is a reproducible comparison with clearly measured trade-offs, rather than another tiny gain on the reused evaluation set.

**25 September submission update:** the root notebook now defines and runs the course analysis in its own cells from the original CSVs. The older artifact-backed notebook can be regenerated separately with `scripts/build_notebook.py`; it is no longer the submission entry point. Its broader exploratory tables remain available in `artifacts/` and `reports/`.

**25 September preservation update:** the earlier 94-cell review is checked in as `notebooks/extended_artifact_review.ipynb`, with all 47 code cells executed. The root notebook retains the reproducible course workflow and adds inline plots for local SHAP, LIME, refit stability and subgroup support access. The audit verifies both notebooks and records their hashes.

**25 September one-file submission update:** the root notebook now embeds the entire earlier review as an appendix: all narrative, 47 collapsed historical code listings, 50 HTML tables and 17 figure attachments. These historical outputs are snapshots and do not rerun as part of the self-contained core. The core modeling and course analysis cells remain executable from CSVs without project imports. The extended notebook is retained as the appendix source snapshot.

**25 September data packaging update:** the root notebook embeds the full NIJ CSV and the original first-release training/test CSVs as compressed, SHA-256-checked data. The `.ipynb` is therefore the only repository file needed to read and rerun the course workflow. Installed Python packages and TabICL's checkpoint remain external runtime dependencies.
