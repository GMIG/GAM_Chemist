# LiPF6 GAM search: proposed experiment

Date: 2026-09-23

## Objective and boundary

Try up to 24 candidate GAM configurations to improve LiPF6 development performance over `s(T) + s(c)`. Change model terms, spline basis size, and smoothing strength. Use the prepared data and saved DOI split assignments in `runs/prep-01` throughout. Keep the final holdout sealed. This is an exploratory model comparison; improvement is an outcome to measure, not a guarantee.

The user selected LiPF6 and permitted both term changes and smoothing/flexibility changes. Following review, use an adaptive sequence: choose each next experiment from the results already observed. Fix the search boundary, comparison rules, and budget in advance; do not preassign a full sequence of model structures.

## Data available for this experiment

- 4,090 retained LiPF6 rows from seven DOIs.
- 3,440 development rows and 650 reserved holdout rows.
- Three fixed development validation folds: 1,627, 1,245, and 568 rows.
- Available varying predictors in every training fold: `T`, `c`, `DEC`, and `EC`.
- PC remains the implicit reference solvent. Its fraction is retained in the dataset.
- TFP has no variation in development. EMC has no variation in one training fold. Do not add either as a modeled effect in this experiment; retain their rows and fractions unchanged.

The existing baseline mean fold RMSE is 1.380871 mS/cm and mean fold MAE is 1.040699 mS/cm. Re-evaluate the baseline as candidate S01 so it is included in the same experiment record. Use development data only for diagnostics and ranking.

## Chemistry hypotheses

Conductivity depends on mobile charge carriers and their mobility. Changes in salt concentration and solvent composition can change viscosity, ionic association, and transport. This motivates curved concentration effects and interactions, without fixing their sign or assuming a universal conductivity maximum.

Primary sources informing these qualitative hypotheses:

- [Ionic association analysis of LiTDI, LiFSI and LiPF6 in EC/DMC](https://pubs.rsc.org/en/content/articlehtml/2019/ra/c8ra08430k): concentration, temperature, conductivity, viscosity, and ionic association.
- [Transport properties of LiPF6 in EC/DEC compared with EC/DMC](https://www.sciencedirect.com/science/article/pii/S0378775321009277): solvent composition, dissociation, and diffusivity.

The proposed GAM interactions are modeling hypotheses inferred from these studies. Do not transfer numerical concentration optima between the papers and this dataset; their units and solvent systems differ. Fitted solvent terms describe predictive associations in these data and do not establish isolated causal solvent effects.

## Adaptive experiment sequence

S01 reproduces `s(T) + s(c)` with cubic splines, five basis functions, and `lam = 0.6`. `l(x)` is a linear effect, `s(x)` is a smooth effect, and `te(x,y)` allows the effect of one variable to depend on the other. All models include the usual intercept, and linear terms remain unpenalized.

After each completed candidate:

1. Inspect mean errors, each fold's errors, and development residuals. For composition diagnostics, join saved predictions to solvent fractions by row ID. Examine whether patterns recur across publications or are confined to one fold.
2. Form a chemistry-informed hypothesis for a remaining error pattern or an informative alternative model. Residual patterns suggest tests; they do not prove a mechanism or uniquely diagnose overfitting.
3. Save the proposed parent model, observations, hypothesis, and one intended change before fitting the next candidate.
4. Fit the candidate on the same three saved folds. Compare it with its parent, the current best model, and S01. Record improvements and regressions even when the hypothesis fails.
5. Choose the next experiment from these results. Normally branch from the current best model; branching from an earlier model is allowed when the reason is recorded.

S02 is selected after inspecting S01. Adding `te(T,c)` is one plausible first hypothesis, not a required second model. Subsequent IDs identify actual chronological trials, not a predetermined experiment list.

| Observed pattern or question | Possible next experiment |
| --- | --- |
| Concentration-related errors differ across temperature ranges | Add `te(T,c)`. |
| Errors vary with DEC or EC fraction | Add a linear main effect for that solvent. |
| A linear solvent effect leaves systematic curvature | Replace that term with a smooth effect. |
| Concentration errors vary with solvent fraction | Add `te(c,DEC)` or `te(c,EC)` after including the corresponding main effects. |
| Temperature errors vary with solvent fraction | Add `te(T,DEC)` or `te(T,EC)` after including the corresponding main effects. |
| A smooth or interaction may be more complex than needed | Replace a smooth with a linear term, remove an optional term, or test stronger smoothing in separate trials. |
| Residual curvature persists with the current model | Test weaker smoothing or more basis functions in separate trials. |

These are candidate ideas, not automatic rules or claims about what the data will show. Do not repeat an identical specification and settings combination.

PC is the reference coordinate, but the solvent effects remain conditional on the other included variables and the observed mixtures. Preserve the existing main-term and tensor-term implementation for this comparison.

## Allowed changes and budget

Make exactly one deliberate change relative to the recorded parent:

- Add or remove one optional main term or pairwise interaction.
- Replace one main term between linear and smooth.
- Change the global basis size or global smoothing strength, one setting at a time.

Retain temperature and concentration main effects. DEC and EC main effects are optional. Include the main effects for both variables before adding their interaction; do not remove a main effect while keeping an interaction involving it. Keep the existing limit of three pairwise interactions.

Search settings:

- `n_splines`: 5 or 8.
- `lam`: 0.1, 0.6, 1, or 10.

Apply the selected settings to every smooth main term and every tensor margin in that candidate. Keep spline order at three and linear terms unpenalized. The GAM fits coefficients from each training fold; do not choose coefficients manually. Structural trials retain their parent's smoothing settings; smoothing trials retain their parent's structure.

There are at most 24 candidates including S01 and at most 72 fold fits. The division between term experiments and smoothing experiments depends on the observed results. Do not increase the budget after inspecting results. Failed candidates count toward the budget, retain their failure reason, and cannot win. Stop if the baseline fails; stop early if no further justified, valid, untried proposal remains and record the reason.

## Comparison and outputs

Use the existing arithmetic mean of fold RMSE as the primary score so it remains comparable with the baseline. Record mean MAE, every fold's RMSE/MAE and sample sizes, residual summaries, and development out-of-fold predictions. For exact RMSE ties, prefer fewer terms, then the earlier candidate ID.

Report more than average error. All quantities below are in mS/cm:

| Measure | Calculation and interpretation |
| --- | --- |
| Mean fold RMSE | Arithmetic mean of the three fold RMSE values; retain the existing ranking measure. |
| Standard deviation of fold RMSE | Sample standard deviation (`ddof=1`) of those three values; describes variation across the saved validation folds. |
| Worst-fold RMSE | Maximum of the three fold RMSE values; exposes a poorly predicted publication group. |
| Median absolute error | Median absolute residual across all development out-of-fold predictions; a typical row's error. |
| 90th-percentile absolute error | 90th percentile of absolute residuals, using linear interpolation; shows the upper error tail. |
| Mean signed residual (bias) | Mean of measured minus predicted conductivity across development predictions; positive means average underprediction. |

Also retain mean fold MAE and report pooled RMSE and pooled MAE with explicit labels. Pooled measures give each development row equal weight; mean fold measures give each fold equal weight. Do not silently switch the ranking between these weightings. Display every fold's score and size alongside the summaries.

For the saved LiPF6 baseline, mean fold RMSE is 1.381, its sample standard deviation is 0.566, and worst-fold RMSE is 2.023. Across 3,440 development predictions, median absolute error is 0.624, 90th-percentile absolute error is 2.103, and bias is +0.104. Recompute these values from S01 during the experiment.

Standard deviation across three folds is descriptive spread, not a confidence interval or uncertainty band on an individual prediction. Do not use the thousands of related measurement rows as independent repeats to claim statistical significance. If a candidate lowers mean RMSE but worsens worst-fold or tail error, explicitly report that tradeoff rather than describing it as uniformly better.

Report the best candidate across all trials and its absolute and percentage RMSE change versus S01. Show the change for every fold, including regressions in individual folds. Because results guide subsequent trials, the searched development score is not an independent estimate of final performance. Do not score the final holdout or claim guaranteed generalization.

Write one experiment directory under `runs/lipf6-search-01` containing:

- Candidate model JSON, a pre-fit `proposal.json` with observations and hypothesis, `evaluation.json`, and `oof_predictions.csv` in a directory per candidate.
- `summary.csv` with candidate ID, parent candidate ID, intended change, terms, settings, status, the error measures above, fold errors, and failure reason where relevant.
- `best_spec.json` for rerunning the selected candidate.
- `report.md` with the chronological decisions and hypotheses, ranking, baseline comparison, fold changes, and limitations.
- An experiment manifest identifying the prepared dataset hash, processing version, split-file hash, and search settings.

## Minimal implementation after design approval

Extend the evaluator's JSON specification with optional `smoothing: {"n_splines": 5, "lam": 0.6}`. Existing specifications retain their current defaults. Validate an integer basis size greater than cubic spline order and a finite positive smoothing strength; save resolved settings with each evaluation.

For this experiment, the assistant reviews each result, writes the next proposal and model JSON, and invokes the existing evaluator. Use a small Python helper only as needed to summarize development residuals and collect results. This does not implement the future LLM proposer or an autonomous experiment loop. No new service, optimization framework, or package dependency is needed.

Verification is limited to the existing tests plus a focused check that default and supplied smoothing settings reach both smooth and tensor terms, and that invalid settings are rejected. Inspect the actual experiment outputs for complete candidate records, consistent saved folds, finite scores for successful fits, and absence of holdout rows from predictions.
