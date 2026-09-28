# CALiSol-23 GAM experiment

## Competition GAM

Place the competition's `train.csv`, `test.csv`, `sample_submission.csv`, `solvent_properties.csv` and `metaData.csv` in `data/task/`. Datasets, generated runs and the virtual environment are excluded from Git.

Run the fixed molecular-descriptor GAM and its two reference models:

```powershell
uv run python scripts/challenge_gam.py --data-dir "data/task" --output-dir "runs/competition-gam-01"
```

The command derives four solvent descriptors from SMILES, evaluates the supplied split and separate DMC/EMC holdouts, then fits all labeled competition rows and predicts DEC mixtures. It reads only the supplied competition files. The local target and submission format are **log_k = log10(k in mS/cm)** and **id,log_k**, as specified by the local metadata and sample submission.

Open `runs/competition-gam-01/report.md` for scores and plots; `submission.csv` contains the predictions. Metrics, split assignments, descriptors, input hashes and fitted coefficients are saved alongside them. See the [approved competition spec](docs/superpowers/specs/2026-09-23-competition-gam-v1-design.md). There is no parameter search in this first version.

## Earlier publication-split experiment

This small experiment has two independent commands: data preparation and GAM evaluation. The first reads a local [CALiSol-23](https://data.dtu.dk/articles/dataset/CALiSol-23/24559960) CSV and writes a filtered CSV, dataset card, audit, and DOI-grouped split manifest. The second fits a supplied GAM structure on the saved development folds. The dataset is published under CC BY 4.0. Solvent molar masses and densities in `scripts/solvent_constants.json` come from the [authors' notebook](https://github.com/Pele0599/CALiSol-23/blob/main/CALiSol-23.ipynb).

## Run data preparation

With [uv](https://docs.astral.sh/uv/) and Python 3.13 (`uv run` creates the environment when needed):

```powershell
uv run python scripts/prepare_calisol.py --input "data/raw/CALiSol-23 Dataset.csv" --output-dir "runs/prep-01"
```

Download the CSV from the dataset page or the authors' repository to the indicated input path first. The script takes any local input/output paths via its two flags. Open `dataset_card.md` in the output directory for the generated salt counts and selected solvents. The filtered CSV has a name containing the source file's SHA-256 prefix; `run_manifest.json` records its exact name.

## Evaluate a GAM

```powershell
uv run python scripts/evaluate_gam.py --prep-dir "runs/prep-01" --salt LiPF6 --spec "model_specs/baseline.json" --output-dir "runs/gam-baseline-LiPF6"
uv run python scripts/evaluate_gam.py --prep-dir "runs/prep-01" --salt LiBF4 --spec "model_specs/baseline.json" --output-dir "runs/gam-baseline-LiBF4"
uv run python scripts/evaluate_gam.py --prep-dir "runs/prep-01" --salt LiBOB --spec "model_specs/baseline.json" --output-dir "runs/gam-baseline-LiBOB"
```

The baseline is `s(T) + s(c)`; `model_specs/linear_c.json` replaces `s(c)` with `l(c)`. Each run writes `evaluation.json` with fold RMSE/MAE and temperature/concentration residual summaries, plus `oof_predictions.csv` with one development prediction per row. LiPF6 and LiBF4 each use three DOI-grouped development folds and one untouched DOI holdout. LiBOB has only two DOIs, so it uses two DOI-grouped folds and has no final holdout. A model spec may contain linear or smooth main terms and up to three pairwise tensor interactions; see the [implementation spec](docs/superpowers/specs/2026-09-23-gam-first-design.md) for the JSON grammar.

## Checks

```powershell
uv run python -m unittest discover -s tests -v
```

The LLM proposer and iterative experiment loop are specified but are not implemented here.
