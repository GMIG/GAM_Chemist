# First competition GAM: implementation specification

Date: 2026-09-23
Status: draft for user review. Implementation and model runs follow approval.

## Goal and scope

Build one small, runnable GAM that predicts the local competition target for every test row. Its solvent inputs must be computable for DEC even though no labeled training mixture contains DEC. Verify ordinary validation accuracy and test transfer to other withheld solvents.

Use one fixed descriptor GAM and two fixed reference models. No adaptive search, feature-selection loop, hyperparameter tuning, LLM client, or automated competition submission in this milestone. The references establish whether the GAM beats a constant and whether molecular inputs help.

The user confirmed that `data/task/metaData.csv` and `sample_submission.csv` govern this task. They supersede the pasted webpage's incompatible output format and conductivity floor.

## Authoritative data and target

| File under `data/task` | Role |
| --- | --- |
| `train.csv` | 6,759 labeled measurements, none containing DEC. |
| `test.csv` | 5,959 unlabeled measurements, all containing DEC. |
| `sample_submission.csv` | Required header and ID order: `id,log_k`. |
| `solvent_properties.csv` | Names and SMILES for 38 molecules; consistency checks and descriptor table. |
| `metaData.csv` | Definitions, units, target and task contract. |

Train directly on supplied `log_k = log10(k in mS/cm)` and submit predicted log_k directly. Do not transform the target again, exponentiate the submission, clip targets/predictions, or remove low-conductivity rows. Negative log_k predictions are valid.

Use supplied mole fractions. Do not rerun the earlier CALiSol filtering or solvent selection. Cover every test row and all three test salts: LiPF6, LiBF4 and LiBOB. Training contains 12 salts.

Use only these competition files. Do not recover DEC labels from the original full CALiSol dataset or reuse models/settings selected using those labels. The earlier publication-split, raw-conductivity experiment is separate.

## Exactly eight conceptual inputs

| # | Input | Definition | Initial term |
| --- | --- | --- | --- |
| 1 | `temperature_K` | Supplied temperature in K. | Smooth |
| 2 | `conc_native` | Supplied concentration, preserving its native unit. | Smooth |
| 3 | `unit_molL` | 1 for mol/L, 0 for mol/kg. | Linear |
| 4 | Salt identity | `salt_name`, encoded as salt-specific adjustments. | Categorical, ridge penalized |
| 5 | `mix_mw` | Mole-fraction-weighted molecular weight. | Linear |
| 6 | `mix_tpsa` | Mole-fraction-weighted topological polar surface area. | Linear |
| 7 | `mix_logp` | Mole-fraction-weighted RDKit calculated logP. | Linear |
| 8 | `mix_rings` | Mole-fraction-weighted ring count. | Linear |

Salt identity expands into one binary column per salt present in the current fitting partition. With all 12 salts available, the matrix has 19 columns: six continuous inputs, the unit indicator, and 12 salt indicators. Eight conceptual inputs therefore do not mean eight coefficients or eight encoded columns.

Only solvent SMILES are converted to molecular descriptors in v1. Salt SMILES, density, estimated molarity, original ratio fractions/basis and solvent classes are available source information but are not predictors. ID, DOI, subset, k and log_k are never predictors. DOI may label validation diagnostics.

### SMILES conversion

Parse distinct solvent structures with RDKit `Chem.MolFromSmiles`. Compute `Descriptors.MolWt`, `rdMolDescriptors.CalcTPSA`, `Crippen.MolLogP`, and `rdMolDescriptors.CalcNumRings`. Cache descriptors and record the RDKit version. Fail clearly on invalid SMILES.

For each descriptor d and supplied mole fractions x_i:

`mixture_d = sum(x_i * descriptor_d(solvent_i))`

Align semicolon-separated names, SMILES and mole fractions. Require equal list lengths, finite nonnegative fractions and a sum within 1e-8 of one. Validate names/structures against `solvent_properties.csv` using canonical SMILES. Use supplied fractions without conversion or silent renormalization.

DEC uses the same numerical descriptor coordinates as DMC, EMC and other solvents. No conductivity labels are required to calculate DEC descriptors. There is no solvent-name one-hot encoding or fitted DEC-specific coefficient.

Weighted averages lose structural information and can be similar for different mixtures. These are structural summaries, not measured mixture viscosity or dielectric constants. Transfer performance must be tested.

## Fixed model and fitting

Use pyGAM `LinearGAM` with:

```text
log_k = intercept
      + s(temperature_K) + s(conc_native)
      + l(unit_molL) + salt_adjustment
      + l(mix_mw) + l(mix_tpsa) + l(mix_logp) + l(mix_rings)
```

- Both smooth terms: cubic splines, five basis functions, `lam = 0.6`.
- All linear terms and salt indicators: L2 penalty with `lam = 1.0`.
- Intercept: unpenalized. Include all fitting-partition salt indicators; their shrinkage controls redundancy with the intercept.
- Standardize the six continuous inputs using only the current fitting rows: subtract means and divide by population standard deviations. Use scale 1 for a constant input. Leave unit/salt indicators unscaled.
- Fit salt vocabulary on the current fitting rows. An unseen validation salt receives all-zero salt indicators and hence no salt-specific adjustment. Report these cases. All final test salts occur in the full training data.
- No interactions, extra descriptors, weights, clipping, filtering or alternate transformations in this first model.
- The unit indicator tracks reported units but does not physically convert molarity and molality. A common concentration curve plus unit offset is an approximation to evaluate.
- Fail explicitly on non-finite inputs/predictions or failed/nonconverged fitting. Do not emit a partial submission.

With 12 fitted salts, there are 28 coefficients: 10 spline coefficients, five other linear coefficients, 12 salt adjustments and one intercept. Effective degrees of freedom will be lower because of regularization. Chemical-holdout fits have fewer salt coefficients when salts are absent from fitting.

### Fixed references

1. **Constant:** predict the mean log_k of the current fitting rows. Recompute it for each split.
2. **Conditions-and-salt GAM:** use identical settings and rows but omit the four mixture-descriptor terms. This isolates the value of molecular inputs.

The descriptor GAM is the specified first submission model. If a reference performs better, report that result clearly. Do not silently tune or switch the submission model; the outcome informs later experiments.

## Three validation checks

Fit each of the three fixed models afresh for each check:

| Check | Fitting rows | Validation rows | Interpretation |
| --- | --- | --- | --- |
| Supplied split | 6,070 rows with subset=train | 689 rows with subset=val | Withheld formulations; not unseen-solvent transfer. |
| DMC holdout | 5,702 subset=train rows without DMC | 368 subset=train rows containing DMC | Transfer to a related unseen solvent. |
| EMC holdout | 4,905 subset=train rows without EMC | 1,165 subset=train rows containing EMC | Transfer to a related unseen solvent. |

Use exact solvent-list membership. Every mixture containing the withheld solvent is excluded from its fit, irrespective of fraction, salt, temperature or concentration. The supplied subset=val rows never enter either chemistry check.

The DMC check also withholds all LiPDI and LiTDI training examples; the EMC check withholds all LiFSI examples. Retain them in full scores, list their counts/errors, and report a secondary score for salts seen in the respective fit. Full scores do not isolate solvent transfer alone.

Verify no exact formulation crosses the supplied split, using salt, normalized solvent composition, native concentration and unit; temperature is not part of the formulation key. Save all ID assignments. Record overlap between the DMC and EMC validation sets; they must not be portrayed as independent statistical repetitions.

Keep the model fixed through all checks. These checks are imperfect substitutes for the actual DEC task because solvent, mixture and salt distributions differ. They do not produce a true DEC test score. Never average the supplied-split score together with the chemistry-check scores.

## Metrics and diagnostics

For true y=log_k, prediction yhat and residual r=y-yhat, report for each model/check:

- Full RMSE: sqrt(mean(r**2)); primary metric, on the log10 conductivity scale.
- MAE, median absolute error and 90th-percentile absolute error (linear quantile interpolation).
- Mean signed residual (bias); positive means underprediction in log space.
- Trimmed RMSE on validation rows with true log_k >= -3, with included/excluded counts. Diagnostic only: keep all rows in fitting and the full score.
- Counts and errors by salt and publication, marking unseen salts.

For DMC and EMC, additionally show mean RMSE, sample standard deviation of the two RMSEs (`ddof=1`), and worst-check RMSE. Give the two checks equal weight and show both scores individually. Two-check standard deviation describes observed spread; it is not a confidence interval or individual prediction uncertainty.

Compare the descriptor GAM with both references for each check and flag regressions in worst-check or tail error. Do not present a mean improvement as uniform improvement. Do not claim to reproduce hidden test scores quoted in metadata.

Provide compact predicted-versus-observed and residual figures from validation predictions. Residual views cover temperature, concentration and four mixture descriptors, with checks/salts distinguished. Include fitting/validation descriptor ranges to flag extrapolation; range overlap does not establish chemical coverage.

## Final fit and submission

Refit the same descriptor GAM on all 6,759 training rows, including both supplied subsets. Refit scaler and salt vocabulary on those rows, then predict every test row.

Write `submission.csv` with exactly `id,log_k`, in sample_submission.csv order. Require unique matching test/sample IDs, 5,959 predictions for this input version, and finite values. Preserve IDs verbatim. Do not apply an inverse transform or conductivity floor. Create the file locally; uploading it is a separate action. Test labels are unavailable, so no observed test RMSE can be reported.

## Minimal implementation and artifacts

One command, with fixed explicit v1 settings:

```powershell
uv run python scripts/challenge_gam.py --data-dir "data/task" --output-dir "runs/competition-gam-01"
```

Use the existing uv environment with NumPy, pyGAM, RDKit and matplotlib. No service, database, generic experiment framework or further optimization dependency.

| Artifact | Content |
| --- | --- |
| `model_spec.json` | Exact eight inputs, descriptor definitions, terms and settings. |
| `run_manifest.json` | SHA-256 of all five input files, package versions, counts, target/split definitions, scaler and salt vocabulary for each fit. |
| `solvent_descriptors.csv` | Four descriptors for all 38 supplied solvent structures, including DEC. |
| `splits.csv` | Check name, row ID and fit/validation assignment. |
| `metrics.json` | All model/check scores, salt/publication breakdowns and chemistry-check summaries. |
| `validation_predictions.csv` | Check, model, ID, salt, DOI, actual log_k, predicted log_k and residual. IDs may recur across checks/models. |
| `diagnostics/` | Validation figures and feature-range summaries. |
| `report.md` | Input/target explanation, reference comparisons, dispersion/tail errors, coverage limits and submission description. |
| `submission.csv` | Final id,log_k predictions. |

Reject invalid schema, duplicate/mismatched IDs, unexpected units, non-finite values, invalid SMILES/fractions, DEC in training or test rows without DEC. Identify the problematic file/row. Do not silently drop or repair rows.

## Minimal verification and acceptance

Keep two focused unit tests: molecular transfer/mixture weighting, and log-space metrics/trimmed counts. Add a small unseen-salt encoding check if needed by the implementation. Run existing tests for regressions.

Use one complete real-data run to verify the fixed fits succeed, split membership is correct, constants use each fit's own target mean, metrics recompute from saved predictions, and the submission matches the required header, IDs and count with finite outputs. Verify that target, DOI, subset and ID never enter the predictor matrix.

This spec supersedes the earlier broader competition draft. Implement and run only after the user reviews it; consider extensions after the first model's results.
