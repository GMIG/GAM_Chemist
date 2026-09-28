# CALiSol-23 Data Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Provide a standalone Python command that reproducibly filters CALiSol-23, converts solvent ratios to mole fractions, selects up to five solvents per salt, and writes a named dataset, audit, split manifest, and dataset card.

**Architecture:** One Python script owns preparation and file output. A checked-in JSON file holds the 38 solvent molar masses and densities from the authors' notebook. Three small standard-library `unittest` cases verify conversion/mixture exclusion, DOI split isolation, and artifact provenance; one real-data smoke run verifies generated artifacts.

**Tech Stack:** Python 3.13 standard library, scikit-learn for `GroupKFold`; no pandas, notebook runtime, or LLM dependency.

## Global Constraints

- Follow `docs/superpowers/specs/2026-09-23-gam-first-design.md` for filters, units, schema, filename, and eligibility.
- A salt needs at least 200 retained rows and five distinct DOIs for splits.
- The script must accept `--input` and `--output-dir` and must derive the output name from the input SHA-256.
- Keep automated verification to three focused data-prep tests and one real-data smoke run.
- This workspace is not a Git repository; no commit or worktree steps apply.

---

### Task 1: Conversion and per-salt filtering

**Files:**
- Create: `scripts/prepare_calisol.py`
- Create: `scripts/solvent_constants.json`
- Create: `tests/test_prepare_calisol.py`

**Interfaces:**
- `convert_fractions(ratios: dict[str, float], ratio_type: str, constants: dict) -> dict[str, float]` returns all 38 normalized fractions.
- `select_solvents(rows: list[dict], solvent_names: list[str]) -> tuple[list[dict], dict]` returns converted rows with no outside solvents and the per-salt selected sets.

- [x] Write a conversion test using EC/DEC 50:50 by mass, volume, and mole, checking normalized fractions against the formulas in the spec. In the same test, provide a row with a positive sixth solvent and assert it is excluded after top-five selection.
- [x] Run `python -m unittest discover -s tests -v`; confirm the test fails because the script functions do not exist yet.
- [x] Add the 38 constants from the authors' notebook and implement required-column validation, row-quality filtering, ratio conversion, per-salt frequency selection, and a count of every exclusion reason.
- [x] Run the same test command and confirm the conversion test passes.

### Task 2: Reproducible splits and output artifacts

**Files:**
- Modify: `scripts/prepare_calisol.py`
- Modify: `tests/test_prepare_calisol.py`

**Interfaces:**
- `make_splits(rows: list[dict], audit: dict, min_rows: int = 200, min_dois: int = 5) -> list[dict]` chooses the holdout DOI by size and uses `GroupKFold(n_splits=3)` on development rows.
- `prepare(input_path: Path, output_dir: Path) -> Path` writes the named CSV, `audit.json`, `splits.csv`, `run_manifest.json`, and `dataset_card.md`; it returns the CSV path.

- [x] Write a test with five DOIs that checks deterministic holdout choice, three development folds, and DOI isolation. Set a small synthetic eligibility threshold only through a function argument so the production rule remains fixed at 200 rows/five DOIs.
- [x] Run `python -m unittest discover -s tests -v`; confirm the new test fails because split generation is absent.
- [x] Implement the split manifest and the exact CSV, JSON, and Markdown schemas in the spec. Add an `argparse` entry point for `--input` and `--output-dir`.
- [x] Check per-salt exclusion reasons and source/constant hashes in the generated audit and manifest.
- [x] Run `python -m unittest discover -s tests -v` and `python -m compileall -q scripts tests`; confirm both succeed.
- [x] Download the published CSV to `data/raw/CALiSol-23 Dataset.csv` and run `python scripts/prepare_calisol.py --input "data/raw/CALiSol-23 Dataset.csv" --output-dir "runs/prep-01"`. Inspect the generated card, row counts, file hash, fraction sums, and split DOI isolation. The provisional reference is LiPF6 4,090 rows/seven DOIs, LiBF4 2,881/four, LiBOB 3,251/two, LiAsF6 45/one.

The GAM evaluator is a separate implementation plan and must consume these artifacts without re-running preprocessing.

