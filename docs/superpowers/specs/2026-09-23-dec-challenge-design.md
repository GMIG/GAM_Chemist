# DEC challenge: local-file implementation

**Superseded draft:** the narrowed first milestone is specified in [First competition GAM](2026-09-23-competition-gam-v1-design.md). Its eight inputs and fixed checks replace the broader feature set and adaptive search below.

The user confirmed that `data/task/metaData.csv` and `sample_submission.csv` govern this task. Predict `log_k = log10(k)` directly and submit `id,log_k`. Preserve the low-conductivity tail; apply no target floor and no additional logarithm. The previous raw-conductivity/publication-split experiment is not a competition benchmark.

## Inputs and features

Use only the five CSV files in `data/task`. Do not recover DEC labels from the original full CALiSol dataset or reuse models fitted on those labels. All 6,759 labeled rows exclude DEC; all 5,959 test rows contain DEC. Test salts are LiPF6, LiBF4 and LiBOB, all observed in training.

Compute deterministic RDKit descriptors from supplied molecular SMILES and mixture mole fractions: weighted molecular weight, TPSA, logP, ring count, oxygen count, rotatable-bond count, carbonate fraction, and the weighted spread of molecular weight. These are structural descriptors, not measured viscosity or dielectric constants. Include supplied mixture density, native concentration and its unit, temperature and inverse temperature, and log1p(native concentration) as available inputs. Use only selected terms in each candidate.

Provide salt molecular descriptors (molecular weight, fluorine and oxygen counts) and optional ridge-penalized salt offsets. Vocabulary comes from the supplied training file; salts absent from a fold's fitting data receive no fitted offset and rely on the molecular descriptors. IDs, source DOI, subset, k, and log_k are never predictors.

## Validation and model search

Use only rows marked `subset=train` (6,070) for model selection. Make two fixed solvent holdouts: DMC (368 validation rows) and EMC (1,165 validation rows). Each fit excludes every mixture containing the respective held-out solvent. Report unseen salts in each fold rather than dropping them.

Reserve `subset=val` (689 rows) for one evaluation of the selected specification. This supplied split checks withheld formulations, not unseen solvents, and must be labeled accordingly. After that check, fit the frozen specification on all 6,759 training rows and generate the submission.

Start with additive inverse-temperature/concentration smooths and molecular composition effects. Choose subsequent model terms and smoothing changes from observed validation residuals, recording the rationale before each fit. Use at most 24 candidate configurations. Keep descriptors, target, training rows and splits fixed. Scale numeric predictors using fitting rows only. Ridge-penalize linear descriptors and salt offsets; use cubic smooths and pairwise tensor terms.

Rank by mean full log-RMSE across the two chemical holdouts. Report individual fold errors, sample standard deviation, worst-fold RMSE, MAE, median absolute error, 90th-percentile absolute error and signed bias. Also report trimmed RMSE on labeled validation rows with true log_k >= -3, explicitly including retained/excluded counts; the full score remains primary. Descriptive fold spread is not a confidence interval. No true test score is available locally.

## Minimal deliverables

- One standalone Python command for candidate evaluation and final model/submission generation.
- A small recording helper for adaptive proposals and a summary CSV.
- Saved candidate specifications, hypotheses, validation predictions and full metrics.
- Final `submission.csv`, frozen model specification, supplied-validation metrics and a concise report with diagnostic plots.
- Input hashes and explicit feature/target/split settings for reproducibility.

RDKit supplies molecular descriptors; matplotlib supplies report figures. Keep automated testing to descriptor transfer/mixture weighting and metric semantics, plus existing tests. Check exact submission header, ID order, row count and finite predictions.

## Execution checklist

- [ ] Implement features, chemical validation, metrics and final submission command.
- [ ] Run focused tests, then perform and record sequential model experiments.
- [ ] Freeze the selected specification, evaluate supplied validation once, and fit all training rows.
- [ ] Validate the submission and write the result report.
