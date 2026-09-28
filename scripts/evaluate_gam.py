"""Evaluate a declarative GAM on fixed CALiSol-23 development folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import warnings
from pathlib import Path

import numpy as np
from pygam import LinearGAM, l, s, te  # type: ignore[reportMissingImports]


def validate_spec(spec: dict, feature_names: list[str]) -> dict:
    """Validate and canonicalize the small, non-executable model grammar."""
    if set(spec) not in ({"main_terms", "interactions"}, {"main_terms", "interactions", "smoothing"}):
        raise ValueError("Model spec requires main_terms and interactions, with optional smoothing")
    smoothing = spec.get("smoothing", {"n_splines": 5, "lam": 0.6})
    if not isinstance(smoothing, dict) or set(smoothing) != {"n_splines", "lam"}:
        raise ValueError("smoothing requires n_splines and lam")
    if type(smoothing["n_splines"]) is not int or smoothing["n_splines"] <= 3:
        raise ValueError("n_splines must be an integer greater than cubic spline order 3")
    lam = smoothing["lam"]
    if type(lam) not in (int, float) or not math.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive")
    if not isinstance(spec["main_terms"], list) or not isinstance(spec["interactions"], list):
        raise ValueError("Model terms must be lists")
    main_terms = []
    seen_features = set()
    for term in spec["main_terms"]:
        if not isinstance(term, dict) or set(term) != {"feature", "type"}:
            raise ValueError("Each main term needs feature and type")
        feature, kind = term["feature"], term["type"]
        if feature not in feature_names or kind not in {"s", "l"}:
            raise ValueError(f"Unsupported main term: {term}")
        if feature in seen_features:
            raise ValueError(f"Duplicate main term for {feature}")
        seen_features.add(feature)
        main_terms.append({"feature": feature, "type": kind})
    if len(spec["interactions"]) > 3:
        raise ValueError("At most three pairwise interactions are allowed")
    interactions = []
    seen_pairs = set()
    for pair in spec["interactions"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each interaction must be a pair of features")
        if any(feature not in feature_names for feature in pair) or pair[0] == pair[1]:
            raise ValueError(f"Unsupported interaction: {pair}")
        ordered = tuple(sorted(pair))
        if ordered in seen_pairs:
            raise ValueError(f"Duplicate interaction: {pair}")
        seen_pairs.add(ordered)
        interactions.append(list(ordered))
    if not main_terms and not interactions:
        raise ValueError("Model must contain at least one term")
    return {"main_terms": main_terms, "interactions": interactions, "smoothing": dict(smoothing)}


def _pygam_terms(spec: dict, feature_names: list[str]):
    indices = {feature: index for index, feature in enumerate(feature_names)}
    smoothing = spec.get("smoothing", {"n_splines": 5, "lam": 0.6})
    terms = []
    for term in spec["main_terms"]:
        index = indices[term["feature"]]
        terms.append(
            s(index, spline_order=3, **smoothing)
            if term["type"] == "s"
            else l(index, penalties=None)
        )
    for first, second in spec["interactions"]:
        terms.append(
            te(
                indices[first], indices[second],
                spline_order=3, **smoothing,
            )
        )
    combined = terms[0]
    for term in terms[1:]:
        combined += term
    return combined


def _residual_bins(predictions: list[dict], feature: str) -> list[dict]:
    values = [row[feature] for row in predictions]
    low, high = min(values), max(values)
    width = (high - low) / 3 if high > low else 0
    groups = [[] for _ in range(3 if width else 1)]
    for row in predictions:
        index = min(int((row[feature] - low) / width), 2) if width else 0
        groups[index].append(row["residual"])
    summaries = []
    for index, residuals in enumerate(groups):
        if not residuals:
            continue
        summaries.append({
            "lower": low + index * width,
            "upper": low + (index + 1) * width if width else high,
            "count": len(residuals),
            "mae": statistics.mean(abs(value) for value in residuals),
            "mean_residual": statistics.mean(residuals),
        })
    return summaries


def evaluate(prep_dir: Path, salt: str, spec: dict, output_dir: Path) -> dict:
    """Fit each fixed development fold and write metrics plus OOF predictions."""
    manifest = json.loads((prep_dir / "run_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((prep_dir / "audit.json").read_text(encoding="utf-8"))
    salt_info = audit["salts"].get(salt)
    if not salt_info or not salt_info["eligible"]:
        raise ValueError(f"Salt {salt!r} is not eligible for modeling")
    feature_names = ["T", "c"] + [
        name for name in salt_info["selected_solvents"]
        if name != salt_info["reference_solvent"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        clean_spec = validate_spec(spec, feature_names)
    except ValueError as error:
        result = {"status": "failed", "salt": salt, "reason": str(error), "spec": spec}
        (output_dir / "evaluation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result

    with (prep_dir / manifest["dataset_file"]).open(newline="", encoding="utf-8") as file:
        data = {int(row["row_id"]): row for row in csv.DictReader(file) if row["salt"] == salt}
    with (prep_dir / "splits.csv").open(newline="", encoding="utf-8") as file:
        split_rows = [row for row in csv.DictReader(file) if row["salt"] == salt]
    development = [row for row in split_rows if row["partition"] == "development"]
    fold_ids = {int(row["fold"]) for row in development}
    if fold_ids not in ({0, 1}, {0, 1, 2}):
        raise ValueError("Expected two or three consecutive development folds")
    if any(int(row["row_id"]) not in data for row in split_rows):
        raise ValueError("Split manifest references a missing dataset row")
    doi_assignments = {}
    for row in split_rows:
        assignment = (row["partition"], row["fold"])
        if row["doi"] in doi_assignments and doi_assignments[row["doi"]] != assignment:
            raise ValueError("DOI crosses split assignments")
        doi_assignments[row["doi"]] = assignment

    fold_results = []
    predictions = []
    for fold in sorted(fold_ids):
        train = [data[int(row["row_id"])] for row in development if int(row["fold"]) != fold]
        validation = [data[int(row["row_id"])] for row in development if int(row["fold"]) == fold]
        for feature in {term["feature"] for term in clean_spec["main_terms"]}.union(
            feature for pair in clean_spec["interactions"] for feature in pair
        ):
            needs_spline = any(
                term == {"feature": feature, "type": "s"}
                for term in clean_spec["main_terms"]
            ) or any(feature in pair for pair in clean_spec["interactions"])
            required_unique = 4 if needs_spline else 2
            if len({float(row[feature]) for row in train}) < required_unique:
                reason = f"Fold {fold}: insufficient variation in {feature}"
                result = {"status": "failed", "salt": salt, "reason": reason, "spec": clean_spec}
                (output_dir / "evaluation.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                return result
        x_train = np.array([[float(row[name]) for name in feature_names] for row in train])
        y_train = np.array([float(row["k"]) for row in train])
        x_validation = np.array(
            [[float(row[name]) for name in feature_names] for row in validation]
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gam = LinearGAM(_pygam_terms(clean_spec, feature_names)).fit(x_train, y_train)
                predicted = gam.predict(x_validation)
            if any("did not converge" in str(warning.message).lower() for warning in caught):
                raise RuntimeError("GAM did not converge")
            if not np.isfinite(predicted).all():
                raise RuntimeError("GAM produced non-finite predictions")
        except Exception as error:
            result = {
                "status": "failed", "salt": salt,
                "reason": f"Fold {fold} fit failed: {error}", "spec": clean_spec,
            }
            (output_dir / "evaluation.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            return result
        residuals = []
        for row, estimate in zip(validation, predicted):
            actual = float(row["k"])
            residual = actual - float(estimate)
            residuals.append(residual)
            predictions.append({
                "row_id": int(row["row_id"]), "doi": row["doi"], "salt": salt,
                "fold": fold, "k": actual, "prediction": float(estimate),
                "residual": residual, "T": float(row["T"]), "c": float(row["c"]),
            })
        fold_results.append({
            "fold": fold,
            "n_train": len(train),
            "n_validation": len(validation),
            "rmse": math.sqrt(statistics.mean(value * value for value in residuals)),
            "mae": statistics.mean(abs(value) for value in residuals),
        })

    predictions.sort(key=lambda row: row["row_id"])
    errors = np.array([row["residual"] for row in predictions])
    result = {
        "status": "ok", "salt": salt, "spec": clean_spec,
        "features": feature_names,
        "reference_solvent": salt_info["reference_solvent"],
        "has_final_holdout": any(row["partition"] == "holdout" for row in split_rows),
        "folds": fold_results,
        "mean_rmse": statistics.mean(fold["rmse"] for fold in fold_results),
        "mean_mae": statistics.mean(fold["mae"] for fold in fold_results),
        "std_fold_rmse": statistics.stdev(fold["rmse"] for fold in fold_results),
        "worst_fold_rmse": max(fold["rmse"] for fold in fold_results),
        "pooled_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "pooled_mae": float(np.mean(np.abs(errors))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "p90_absolute_error": float(np.quantile(np.abs(errors), 0.9)),
        "bias": float(np.mean(errors)),
        "residual_diagnostics": {
            "T_bins": _residual_bins(predictions, "T"),
            "c_bins": _residual_bins(predictions, "c"),
        },
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["row_id", "doi", "salt", "fold", "k", "prediction", "residual", "T", "c"],
        )
        writer.writeheader()
        writer.writerows(predictions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = evaluate(args.prep_dir, args.salt, spec, args.output_dir)
    print(f"{result['status']}: {args.output_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
