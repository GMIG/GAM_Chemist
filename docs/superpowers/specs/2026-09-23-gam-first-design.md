# LLM–GAM Electrolyte Experimenter: GAM-first implementation spec

Date: 2026-09-23

## Decision and scope

Build an independently runnable CALiSol-23 preparation and GAM evaluation pipeline first. Define the future LLM proposal interface now, but do not implement an LLM client or experiment loop in this milestone. The repository is currently empty, so there is no existing code or API to preserve.

The first runnable result is a fixed `s(T) + s(c)` GAM for LiPF6, evaluated by publication-grouped development cross-validation. Preserve the final holdout wherever the split policy provides one. The pipeline must also report why other salts are ineligible, rather than silently omitting them.

The research question remains whether later chemically motivated LLM edits improve an interpretable GAM. This milestone establishes the data and evaluation machinery needed to test that question; it makes no claim about LLM improvement.

## Source and preprocessing

Use the [CALiSol-23 dataset](https://data.dtu.dk/articles/dataset/CALiSol-23/24559960) CSV as a local input file. Record its SHA-256, source URL, row count, and preprocessing settings in a run manifest. The [authors' repository](https://github.com/Pele0599/CALiSol-23) contains the CSV and solvent physical constants/conversion example. The source reports `T` in K and `k` in mS/cm; retained `c` is in mol/kg. Do not require a network connection during fitting.

1. Require `doi`, `k`, `T`, `c`, `salt`, `c units`, `solvent ratio type`, and the 38 named solvent columns. Fail with a named missing-column error if the schema differs.
2. Keep only `c units == "mol/kg"`. Parse `k`, `T`, `c`, and solvent values as finite numbers; require `k > 0`, `c >= 0`, a nonempty DOI, nonnegative solvent ratios, and a raw solvent-ratio sum within 0.03 of 1. Record counts for each exclusion reason. Preserve the original CSV row number as a traceable row ID.
3. Convert all 38 solvent ratios to mole fractions before solvent selection. For ratio `r_i`, use `r_i / M_i` for mass (`w`), `r_i * rho_i / M_i` for volume (`v`), and `r_i` for mole (`mol`), then divide each converted amount by the row total. `M_i` and `rho_i` are explicit, versioned constants taken from the authors' notebook. Stop on an unknown ratio type or missing constant for a positive fraction. Check that converted fractions are finite, nonnegative, and sum to 1 within numerical tolerance.
4. For each salt independently, count rows where each converted solvent fraction is positive. Select up to five most frequent solvents, breaking equal counts by solvent name; if only one to five solvents occur for that salt, select all of them. Keep a row only if **every** positive solvent belongs to that salt's selected set. A retained mixture can contain any number of selected solvents from one through five; the other selected fractions may be zero. Never zero out an unselected solvent. Record the selected set and its size in the audit.
5. Produce a per-salt audit with rows before and after each filter, solvent occurrence frequencies, selected solvents, surviving DOI counts, and eligibility. A salt is eligible for modeling with at least two distinct surviving DOIs and at least 200 surviving rows. Under the current CSV, LiPF6, LiBF4, and LiBOB are eligible. These counts are checked by the preparation command, not hardcoded into modeling logic.

For LiPF6 in the current source CSV, 4,330 rows pass the initial `mol/kg` and data-quality filters; its selected solvents are `PC`, `DEC`, `EC`, `EMC`, and `TFP`. All valid `mol/kg` rows observed in this CSV use mass ratios (`w`), although the converter supports all three declared ratio types. Provisional retained counts for all salts appear in the dataset-card example below; the script computes them for each run.

## Split policy

Create splits after deterministic preprocessing and solvent selection. With exactly two DOIs, use both as two DOI-grouped development folds and reserve no final holdout. With three or more DOIs, reserve one entire DOI as the final holdout: choose the DOI whose row count is closest to 20% of that salt's retained rows, with lexicographic DOI tie-breaking. This uses group size only, never conductivity values. Use two-fold `GroupKFold` on the remaining development rows for three-DOI salts and three-fold `GroupKFold` for salts with four or more DOIs. Keep the same folds for every candidate GAM of a salt. Save each row ID, DOI, and split/fold assignment in a manifest. Record per-salt fold count and whether a holdout exists. Assert that no DOI crosses the holdout/development boundary or two validation folds.

The normal development command must not calculate or print holdout metrics. A separate final-evaluation command may fit the frozen selected specification on all development rows and score the holdout once in the later LLM experiment milestone where a holdout exists. A two-DOI salt has no reserved final test, so its cross-validation scores alone cannot serve as an untouched final assessment. With few publications, fold variance and publication shift are material; retain every fold's score and size.

## Standalone GAM evaluator

Input is the prepared per-salt table, split manifest, and a declarative JSON model specification. Its feature namespace is `T`, `c`, and the selected solvent fractions except one implicit reference. If a salt has `n` selected solvents (`1 <= n <= 5`), preserve all `n` mole fractions in the prepared table and expose `n - 1` independent solvent-fraction coordinates to the GAM. Designate the most frequent solvent as the reference because the selected fractions sum to one; with only one selected solvent, there are no solvent coordinates. Report the reference solvent in the manifest and in future LLM context. `salt`, DOI, ratio type, concentration unit, and `k` are never predictors.

The JSON specification has `main_terms: [{"feature": variable, "type": "s" | "l"}, ...]` and `interactions: [[variable_a, variable_b], ...]`. Map these only to [pyGAM's](https://pygam.readthedocs.io/en/latest/reference/_autosummary/pygam.terms.html) smooth `s()`, linear `l()`, and pairwise tensor `te()` terms; do not accept arbitrary Python, formula text, coefficients, or high-order interactions. Allow at most one main term per variable, either linear or smooth, so replacing `s(x)` with `l(x)` is one structural change. Validate known variable names, unique interactions, distinct variables within an interaction, and at most three interactions. Canonicalize interaction pair order so duplicate pairs are unambiguous. The baseline specification is `main_terms: [{"feature": "T", "type": "s"}, {"feature": "c", "type": "s"}]`, `interactions: []`.

Use `LinearGAM` for raw conductivity `k` with fixed spline settings across candidates: five basis functions per univariate smooth and per tensor margin, cubic spline order, and fixed `lam = 0.6` for smooth terms. Linear terms have no spline basis or smoothing penalty. Do not tune hyperparameters during model-structure comparisons. Fit a fresh model on each development training fold. Return per-fold RMSE and MAE, fold sizes, mean RMSE and MAE, and out-of-fold predictions/residuals. Provide compact residual summaries by temperature and concentration bins; these bins are diagnostics, not predictive features. Mark nonconvergence, unsupported terms, or insufficient training variation as explicit failed evaluations with a reason. A failed evaluation cannot replace an incumbent model.

Only the evaluator imports the GAM library. Data preparation does not fit models, and the evaluator has no LLM dependency.

## Future LLM boundary (contract only)

The future proposer receives variable names/descriptions (including units and reference solvent), current GAM specification, development fold metrics, residual summaries, and complete prior experiment summaries for **one salt**. It never receives final-holdout outcomes. Its structured output has `hypothesis`, `operation`, `term` (or old/new terms for a replacement), and a short chemical/statistical `rationale`. Operations are add/remove a linear or smooth main term, add/remove a pairwise interaction, or replace one term (including `s(x)` ↔ `l(x)`). One proposal changes one structure; it contains no coefficients or executable code.

A deterministic proposal validator, not the LLM, will enforce the feature namespace, one-change rule, and three-interaction limit. A later orchestrator will apply valid proposals, call the same standalone evaluator, log all attempts and fold results, and compare candidates on the fixed development folds. This milestone documents the contract; it does not call an LLM or implement the loop.

## Minimal project shape and outputs

Provide `scripts/prepare_calisol.py` as the runnable data-preparation deliverable. It accepts a local CALiSol-23 CSV and an output directory, performs every validation, conversion, solvent-selection, eligibility, and split step specified above, and creates the output directory if needed. Running this script must be sufficient to reproduce the prepared dataset; no Codex session, notebook execution, manual data editing, or LLM call is part of preparation. Keep the GAM evaluator separate, with its own command that consumes the script's outputs and a JSON term spec. Keep fixed preprocessing and split settings in the Python script and write their values to the run manifest. No database, service, chemistry descriptors, model registry, or generic workflow framework is needed.

The preparation script is invoked as `python scripts/prepare_calisol.py --input "data/raw/CALiSol-23 Dataset.csv" --output-dir "runs/prep-01"`. It writes `calisol23_molkg_top5_molefrac_v2_<sha12>.csv`, `dataset_card.md`, `audit.json`, `splits.csv`, and `run_manifest.json` in that directory. `<sha12>` is the first 12 hexadecimal characters of the input CSV's SHA-256; the manifest records the full hash, processing version `v2`, and exact filtered CSV filename. This names the filtered dataset distinctly from the source and ties it to the input version. The filtered CSV contains retained rows for every salt, their original row IDs and DOIs, and all normalized solvent fractions. `splits.csv` contains row ID, salt, DOI, holdout/development assignment, and development fold for eligible salts only. The evaluation command reads the filtered CSV filename from the manifest and a JSON term spec, then writes `evaluation.json` and development out-of-fold predictions to its own caller-supplied output directory. Re-running preparation on the same input bytes must produce the same data, card, and split assignments.

### Filtered dataset CSV schema

Write one row per measurement that survives the data-quality and solvent-selection filters, including rows from salts ineligible for modeling. Keep rows in source CSV order. The columns, in order, are:

| Column | Meaning |
| --- | --- |
| `row_id` | Unique 1-based data-row number in the input CSV, excluding its header. |
| `doi` | Original publication DOI, retained for grouping and traceability. |
| `salt` | Original salt identity. |
| `k` | Positive conductivity in mS/cm; target, never a predictor. |
| `T` | Temperature in K. |
| `c` | Salt concentration in mol/kg. |
| `source_ratio_type` | Original `w`, `v`, or `mol` ratio type, for provenance only. |
| 38 original solvent column names | Float mole fractions after conversion and normalization; absent solvents are `0`. |

Every retained row has all 38 solvent columns, no missing values, no positive fraction outside its salt's selected set, and a solvent-fraction sum of 1 within floating-point tolerance. The original `c units` column is omitted because every retained row is `mol/kg`; the manifest records this rule. `splits.csv` is separate: it has `row_id`, `salt`, `doi`, `partition` (`development` or `holdout`), and `fold` (0 or 1 for two-fold development; 0, 1, or 2 for three-fold development; empty for holdout rows). It contains only rows from eligible salts.

### `dataset_card.md` info card

Generate the card from the same audit and manifest data as the CSV; do not copy provisional counts from this spec. Include the filtered dataset's exact filename, the CALiSol-23 source link and full input SHA-256, processing version, total retained rows, units (`k` in mS/cm, `T` in K, `c` in mol/kg, solvents as mole fractions), the `mol/kg` and positive-conductivity filters, per-salt selection of up to five solvents, and the eligibility rule (at least two DOIs and 200 rows). Include this table with values calculated from the run. For the current source CSV, the provisional table is:

| Salt | Retained rows | Distinct DOIs | Eligible |
| --- | ---: | ---: | --- |
| LiPF6 | 4,090 | 7 | Yes |
| LiBF4 | 2,881 | 4 | Yes |
| LiBOB | 3,251 | 2 | Yes |
| LiAsF6 | 45 | 1 | No |

List each salt's selected solvents and reference solvent below the table. State that DOI is used for grouped splitting, `splits.csv` holds split assignments, and any final holdout is not scored during development. State which salts have no final holdout. Keep the card short enough to inspect before modeling.

## Verification and acceptance

Keep automated testing narrow:

- One parameterized conversion test checks mass, volume, and mole ratios, normalization, and rejection of a positive solvent outside a selected set.
- Focused split tests check deterministic DOI isolation for two-DOI salts without a holdout, three-DOI salts with two development folds, and larger salts with three folds.
- One artifact test checks per-salt exclusion reasons and the manifest's source and constants provenance.
- Small evaluator tests check finite metrics for two and three development folds and confirm holdout rows never appear in out-of-fold predictions.

Run `scripts/prepare_calisol.py` on the actual CALiSol-23 CSV as the data-prep smoke check, then run the GAM evaluator on its outputs. The evaluator must fit the baseline for LiPF6, LiBF4, and LiBOB, emit finite fold RMSE/MAE, and report whether a final holdout exists. Inspect the generated audit for the expected order of magnitude and DOI counts. Do not add tests that merely repeat implementation details.

Completion of this milestone means another developer can run the preparation and baseline evaluation from a local CSV, inspect exactly which rows and sources were used, submit a different valid GAM term spec without changing code, and get comparable development metrics. It does **not** mean the LLM loop or final scientific comparison is complete.
