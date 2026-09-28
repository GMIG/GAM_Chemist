# Standalone GAM Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Evaluate a declarative GAM term specification on the fixed LiPF6 DOI-grouped development folds produced by the data-prep script.

**Architecture:** A separate Python command reads the preparation manifest, filtered CSV, audit, and split file. It validates a small JSON term grammar, maps it to pyGAM terms, fits one model per development fold, and writes metrics plus out-of-fold predictions. It never scores the final holdout.

**Tech Stack:** Python 3.13, NumPy, pyGAM, standard library CSV/JSON; the prep script's scikit-learn dependency remains separate.

## Global Constraints

- Follow `docs/superpowers/specs/2026-09-23-gam-first-design.md` for the term grammar, baseline, feature namespace, fixed smoothing settings, and metrics.
- Accept only `s`, `l`, and pairwise `te` terms, with at most three interactions and one main term per variable.
- Use fixed folds from `splits.csv`; never tune on or score holdout rows.
- Keep verification to one focused evaluator test and one actual-data smoke sequence.
- This workspace is not a Git repository; no commit or worktree steps apply.

---

### Task 1: Term validation and grouped development evaluation

**Files:**
- Create: `scripts/evaluate_gam.py`
- Create: `tests/test_evaluate_gam.py`
- Create: `model_specs/baseline.json`
- Create: `model_specs/linear_c.json`
- Modify: `pyproject.toml` and `uv.lock`

**Interfaces:**
- `validate_spec(spec: dict, feature_names: list[str]) -> dict` rejects unknown, duplicate, overlimit, or malformed terms.
- `evaluate(prep_dir: Path, salt: str, spec: dict, output_dir: Path) -> dict` reads fixed folds, fits each model, writes `evaluation.json` and `oof_predictions.csv`, and returns metrics.

- [x] Write one synthetic test that creates a tiny prepared run with three DOI development folds, evaluates `s(T) + l(c)`, checks finite fold RMSE/MAE, and verifies no holdout row appears in predictions.
- [x] Run the test and confirm it fails because the evaluator module is absent.
- [x] Pin pyGAM in `pyproject.toml` and `uv.lock`; sync the environment with uv.
- [x] Implement the JSON validator and the fold evaluator with pyGAM `LinearGAM`, `s`, `l`, and `te`. Use five splines/cubic order/`lam=0.6` for smooths and no penalty for linear terms. Record explicit failure reason on fit failure.
- [x] Run the test; confirm it passes.

### Task 2: Real-data smoke and documentation

**Files:**
- Modify: `README.md`

- [x] Run the baseline and `s(c)` to `l(c)` variant on `runs/prep-01` LiPF6 using the same three development folds.
- [x] Check three finite fold scores and predictions only for development rows; inspect residual summaries by T and c bins.
- [x] Document the command and result-file meanings in `README.md`.
- [x] Run the full minimal test suite, compile both scripts, and compare the fold assignments used by the two candidate runs.

