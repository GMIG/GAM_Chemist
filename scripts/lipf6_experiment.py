"""Small recording helper for the assistant's sequential LiPF6 experiment."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_gam import evaluate, validate_spec

PREP = Path("runs/prep-01")
ROOT = Path("runs/lipf6-search-01")
FEATURES = ["T", "c", "DEC", "EC"]
METRICS = ["mean_rmse", "std_fold_rmse", "worst_fold_rmse", "mean_mae",
           "pooled_rmse", "pooled_mae", "median_absolute_error", "p90_absolute_error", "bias"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def signature(spec):
    return json.dumps({**spec, "main_terms": sorted(spec["main_terms"], key=lambda t: t["feature"]),
                       "interactions": sorted(spec["interactions"])}, sort_keys=True)


def refresh_summary():
    records = []
    for directory in sorted(ROOT.glob("S[0-9][0-9]")):
        result = json.loads((directory / "evaluation.json").read_text())
        proposal = json.loads((directory / "proposal.json").read_text())
        record = {"id": directory.name, "parent": proposal["parent"],
                  "change": json.dumps(proposal["change"]), "status": result["status"],
                  "spec": json.dumps(result["spec"]), "reason": result.get("reason", "")}
        record.update({key: result.get(key, "") for key in METRICS})
        record.update({f"fold_{f['fold']}_rmse": f["rmse"] for f in result.get("folds", [])})
        records.append(record)
    fields = ["id", "parent", "change", "status", "spec", *METRICS,
              "fold_0_rmse", "fold_1_rmse", "fold_2_rmse", "reason"]
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    successful = [r for r in records if r["status"] == "ok"]
    if successful:
        def key(r):
            spec = json.loads(r["spec"])
            return r["mean_rmse"], len(spec["main_terms"]) + len(spec["interactions"]), r["id"]
        best = min(successful, key=key)
        write_json(ROOT / "best_spec.json", json.loads(best["spec"]))
        return best["id"]


def run_trial(identifier, parent, observation, hypothesis, change):
    ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(ROOT.glob("S[0-9][0-9]"))
    if len(existing) >= 24 or identifier != f"S{len(existing) + 1:02d}":
        raise ValueError("Expected next sequential ID within the 24-model budget")
    prep_manifest = json.loads((PREP / "run_manifest.json").read_text())
    provenance = {"salt": "LiPF6", "source_sha256": prep_manifest["source_sha256"],
                  "processing_version": prep_manifest["processing_version"], "budget": 24,
                  "input_hashes": {name: hashlib.sha256((PREP / name).read_bytes()).hexdigest()
                                   for name in ("splits.csv", "run_manifest.json", "audit.json", prep_manifest["dataset_file"])}}
    manifest_path = ROOT / "experiment_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != provenance:
            raise ValueError("Prepared inputs changed during experiment")
    else:
        write_json(manifest_path, provenance)
    if parent is None:
        if existing or change != {"baseline": True}:
            raise ValueError("Only S01 can initialize the baseline")
        spec = {"main_terms": [{"feature": "T", "type": "s"}, {"feature": "c", "type": "s"}],
                "interactions": [], "smoothing": {"n_splines": 5, "lam": 0.6}}
    else:
        spec = json.loads((ROOT / parent / "model.json").read_text())
        if len(change) != 1:
            raise ValueError("Exactly one model change required")
        operation, value = next(iter(change.items()))
        if operation == "add_main":
            spec["main_terms"].append(value)
        elif operation == "replace_main":
            assert any(t["feature"] == value["feature"] for t in spec["main_terms"])
            spec["main_terms"] = [value if t["feature"] == value["feature"] else t for t in spec["main_terms"]]
        elif operation == "remove_main":
            spec["main_terms"] = [t for t in spec["main_terms"] if t["feature"] != value]
        elif operation == "add_interaction":
            spec["interactions"].append(value)
        elif operation == "remove_interaction":
            spec["interactions"] = [p for p in spec["interactions"] if sorted(p) != sorted(value)]
        elif operation in ("lam", "n_splines"):
            spec["smoothing"][operation] = value
        else:
            raise ValueError("Unsupported experiment operation")
    spec = validate_spec(spec, FEATURES)
    mains = {t["feature"] for t in spec["main_terms"]}
    assert {"T", "c"} <= mains
    assert all(set(pair) <= mains for pair in spec["interactions"])
    assert spec["smoothing"]["n_splines"] in (5, 8)
    assert spec["smoothing"]["lam"] in (0.1, 0.6, 1, 10)
    if any(signature(json.loads((d / "model.json").read_text())) == signature(spec) for d in existing):
        raise ValueError("Duplicate candidate")
    directory = ROOT / identifier
    directory.mkdir()
    write_json(directory / "proposal.json", {"parent": parent, "observation": observation,
                                            "hypothesis": hypothesis, "change": change})
    write_json(directory / "model.json", spec)
    result = evaluate(PREP, "LiPF6", spec, directory)
    if result["status"] == "ok":
        with (directory / "oof_predictions.csv").open(newline="", encoding="utf-8") as file:
            predictions = list(csv.DictReader(file))
        with (PREP / prep_manifest["dataset_file"]).open(newline="", encoding="utf-8") as file:
            data = {r["row_id"]: r for r in csv.DictReader(file) if r["salt"] == "LiPF6"}
        with (PREP / "splits.csv").open(newline="", encoding="utf-8") as file:
            development_ids = {r["row_id"] for r in csv.DictReader(file)
                               if r["salt"] == "LiPF6" and r["partition"] == "development"}
        assert len(predictions) == len(development_ids) == len({r["row_id"] for r in predictions})
        assert {r["row_id"] for r in predictions} == development_ids
        residual = np.array([float(r["residual"]) for r in predictions])
        diagnostics = {}
        for feature in FEATURES:
            values = np.array([float(data[r["row_id"]][feature]) for r in predictions])
            diagnostics[feature] = {"residual_correlation": float(np.corrcoef(values, residual)[0, 1]), "bins": []}
            edges = np.linspace(values.min(), values.max(), 4)
            for index in range(3):
                mask = (values >= edges[index]) & ((values <= edges[index + 1]) if index == 2 else (values < edges[index + 1]))
                if mask.any():
                    diagnostics[feature]["bins"].append({"lower": float(edges[index]), "upper": float(edges[index+1]),
                        "n": int(mask.sum()), "bias": float(residual[mask].mean()), "mae": float(np.abs(residual[mask]).mean())})
        diagnostics["by_doi"] = []
        for doi in sorted({r["doi"] for r in predictions}):
            errors = np.array([float(r["residual"]) for r in predictions if r["doi"] == doi])
            diagnostics["by_doi"].append({"doi": doi, "n": len(errors), "rmse": float(np.sqrt(np.mean(errors**2))),
                                          "mae": float(np.mean(np.abs(errors))), "bias": float(errors.mean())})
        write_json(directory / "diagnostics.json", diagnostics)
    best = refresh_summary()
    print(json.dumps({"id": identifier, "status": result["status"], "best": best,
                      **{k: round(result[k], 5) for k in METRICS if k in result},
                      "fold_rmse": [round(f["rmse"], 5) for f in result.get("folds", [])],
                      "reason": result.get("reason")}))
