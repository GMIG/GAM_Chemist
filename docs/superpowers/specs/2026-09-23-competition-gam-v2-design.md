# Competition GAM iteration 2: small interaction experiment

Date: 2026-09-23
Status: experiment design approved; written specification awaiting review.

## Objective

Test whether a few fixed pairwise terms improve transfer to withheld solvents. Run five GAM structures, select by the rule below, and produce one separate submission. Keep implementation and verification minimal.

This extends [the v1 specification](2026-09-23-competition-gam-v1-design.md). Its data contract, molecular descriptors, preprocessing, leakage boundary, target, metrics and validation assignments continue to apply.

## Inputs and baseline

Use only `data/task`: train directly on `log_k = log10(k in mS/cm)` and output `id,log_k`. Retain all labeled rows, including the low-conductivity tail. Do not retrieve DEC labels from the original dataset.

Keep the eight conceptual inputs and their encodings from v1. Fit continuous scaling and salt vocabulary separately within every fitting partition. Unseen validation salts receive zero salt adjustment. No new descriptors, concentration conversions, clipping or weighting.

Baseline is the existing descriptor GAM: cubic temperature and concentration smooths with five basis functions each and lambda 0.6; linear unit, salt and four descriptor terms with L2 lambda 1; unpenalized intercept.

## Five fixed candidates

Every candidate contains all baseline terms.

| ID | Additional terms | Hypothesis |
| --- | --- | --- |
| baseline | None | Reproduce iteration 1. |
| A | Concentration x temperature | The concentration response varies with temperature. |
| B | Concentration x mixture ring count | The concentration response varies with the structural ring descriptor. |
| C | Concentration x mixture logP | The concentration response varies with calculated hydrophobicity. |
| D | Both B and C terms | These two descriptor relationships provide complementary improvements. |

Each added term is a pyGAM tensor-product smooth over the already standardized inputs. Fix both marginal bases to cubic splines with five basis functions and smoothing lambda 0.6 on each axis. Each tensor therefore adds 25 coefficients. Keep every existing penalty unchanged; no grid search or adaptive candidate additions.

These standard tensor surfaces can also represent marginal effects. This is a predictive comparison of adding a surface, not a statistically isolated test of a pure interaction or proof of a chemical mechanism. Ring count and logP are structural proxies, not measured dielectric constant or viscosity.

With all 12 salts, baseline has 28 coefficients, A/B/C have 53, and D has 78. Record effective degrees of freedom as well as coefficient counts.

## Evaluation and selection

Fit every candidate on the same three checks used in v1:

- Supplied split: 6,070 fitting / 689 validation rows.
- DMC withheld from subset=train: 5,702 fitting / 368 validation rows.
- EMC withheld from subset=train: 4,905 fitting / 1,165 validation rows.

This is 15 validation fits and one final fit. Recompute baseline alongside the candidates. The two original reference models need not be refitted in this iteration.

Selection is deterministic:

1. A candidate is eligible only if its full DMC RMSE and full EMC RMSE are each no greater than the corresponding recomputed baseline score.
2. Select the eligible model with the lowest unrounded, equally weighted mean of those two RMSEs, including baseline among eligible models.
3. Break exact ties by fewer added tensors, then the fixed order baseline, A, B, C, D.
4. If no candidate strictly improves the baseline mean, retain baseline.

Use full RMSE, including unseen validation salts and all target values, for selection. Supplied-split scores, trimmed metrics, tail errors and per-salt metrics are diagnostics, not additional selection criteria. Explicitly flag any regressions in these diagnostics even when the selection rule chooses that candidate.

Report each check separately, mean chemical-check RMSE, sample SD across the two chemical checks and worst-check RMSE. Retain MAE, median absolute error, P90 absolute error, bias, trimmed RMSE and counts, per-salt/DOI metrics, and seen/unseen salt diagnostics from v1. Bias is measured minus predicted log_k. No new uncertainty estimator.

DMC and EMC are now tuning checks. Their selected-model scores are not untouched generalization estimates. Their shared rows, differing salt coverage and mismatch to DEC mixtures remain limitations. A leaderboard score, if provided, is external context and does not alter this fixed selection rule.

## Minimal implementation and outputs

Reuse v1 preprocessing, encoding, metrics and plotting helpers. Add a small dedicated runner with the five explicit candidate definitions and selection rule; extend shared fitting only as required for the added terms. No generic search framework or new dependencies.

Planned command:

```powershell
uv run python scripts/challenge_interactions.py --data-dir "data/task" --output-dir "runs/competition-gam-02"
```

Preserve iteration-1 outputs. Save in the new directory:

- `model_specs.json`: five complete resolved model definitions and fixed settings.
- `run_manifest.json`: input hashes, package versions, split sizes, scaling, salt vocabularies, coefficients, effective degrees of freedom and fit warnings.
- `solvent_descriptors.csv` and `splits.csv`: the descriptors and actual row assignments used.
- `metrics.json` and `validation_predictions.csv`: all 15 validation fits and their diagnostics.
- `selection.json`: eligible candidates, full-precision selection scores, selected ID and reason.
- `report.md`: comparison table, changes versus baseline, regressions and limitations.
- `diagnostics/`: selected-model parity/residual figures for all three checks and feature ranges, reusing the v1 plotting format.
- `submission.csv`: selected model refitted on all 6,759 labeled rows; 5,959 finite predictions in sample-submission order with exactly `id,log_k`.

Reject invalid inputs, nonconverged fits or non-finite results explicitly. If any planned fit fails, stop and report the failing candidate/check; do not silently choose among an incomplete comparison. Write the submission only after all checks and the final fit succeed. No automatic upload.

## Minimal verification

- Keep existing tests. Add one focused test for candidate tensor definitions and one table-driven selection test covering one-check regression, strict improvement and baseline fallback/ties.
- In one full real-data run, verify baseline reproduces iteration-1 validation predictions within absolute tolerance 1e-8 under the same inputs and environment.
- Independently recompute selection from saved predictions, check unchanged split membership, and verify the final submission header, exact ID order/count and finite predictions.
- Inspect selected-model diagnostic plots and summarize material weaknesses. No additional tuning in this milestone.

## Review checklist

- [x] Inspect existing model and evaluation contract.
- [x] Discuss alternatives and approve the five-model experiment.
- [x] Write the specification and check scope, exact settings and selection rules.
- [ ] User reviews the written specification.
- [ ] Write the minimal implementation plan, then implement and run.

The workspace has no Git repository; this specification is saved locally without a commit.
