# Project plan and handoff

The current approach follows the supplied HEC ISAF brief: compare three model families across performance, interpretability, stability and fairness for a hypothetical support-allocation vendor.

## Implemented review

- Check NIJ protocol/codebook and distinguish annual competition forecasts from our cumulative baseline task.
- Preserve the official training/evaluation partition and document its repeated historical inspection.
- Rebuild inconsistent model artifacts and check saved-model/prediction agreement.
- Add training-prevalence and calibrated-score baselines, Brier skill, paired model differences and fixed-configuration training CV.
- Harden data/metric validation and unify exact-capacity allocation with deterministic ties.
- Add race-by-gender audit denominators and correct fairness/calibration/Jaccard interpretation.
- Regenerate dependent analyses, notebook and slides; improve app startup and missing-value handling.
- Record versions/hashes and provide an ordered reproduction runner.

See reports/research_review.md for evidence, reports/technical_report.md for methodological limits, and reports/presentation_notes.md for Q&A preparation.

## Team handoff

1. Review the pre-validation brief and submit by 24 September 2026 at 09:40.
2. Rehearse the 15-minute presentation and 10-minute Q&A; every member must understand every section.
3. Test the app on the presentation machine, including opt-in TabICLv2 checkpoint/inference.
4. Submit slides, code/notebook and application by 28 September 2026 at 09:40.

## Future research

Use fresh temporal/external data for confirmatory validation. Any new tuning should use training-only nested evaluation. Validate mitigation on independent data and estimate actual intervention benefit prospectively. Historical convergence of model scores does not prove a performance ceiling.
