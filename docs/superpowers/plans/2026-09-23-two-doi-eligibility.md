# Two-DOI Salt Eligibility Implementation Plan

> **For agentic workers:** Implement the two tasks in order with focused test-first checks.

**Goal:** Evaluate salts with at least 200 retained rows and two distinct DOIs.

**Architecture:** Data preparation chooses two or three DOI-grouped development folds based on publication count and reserves a holdout only when at least three DOIs exist. The evaluator reads the assigned fold IDs from `splits.csv` and fits each fold independently.

**Tech Stack:** Python 3.13, uv, scikit-learn GroupKFold, pyGAM, unittest.

## Global Constraints

- Keep solvent conversion and selection unchanged.
- Never use conductivity values to choose splits.
- Preserve DOI isolation and deterministic assignments.
- Keep the implementation and tests small.

### Task 1: Preparation and split artifacts

**Files:** `scripts/prepare_calisol.py`, `tests/test_prepare_calisol.py`, `docs/superpowers/specs/2026-09-23-gam-first-design.md`.

- [x] Add failing tests for two-DOI and three-DOI split behavior and the manifest threshold.
- [x] Run those tests and confirm the expected failures.
- [x] Implement the 2/3/4+ DOI split rule, lower the threshold to two, and record actual per-salt split settings.
- [x] Run the preparation tests.

### Task 2: Evaluator and real-data check

**Files:** `scripts/evaluate_gam.py`, `tests/test_evaluate_gam.py`, `README.md`.

- [x] Add a failing evaluator test for two development folds with no holdout.
- [x] Run that test and confirm the expected failure.
- [x] Read the fold count from `splits.csv`, validate consecutive fold IDs and DOI isolation, and evaluate each fold.
- [x] Run all tests, prepare the CALiSol-23 CSV, and evaluate baseline LiPF6, LiBF4, and LiBOB.
- [x] Inspect generated card, split assignments, fold scores, and document commands.
