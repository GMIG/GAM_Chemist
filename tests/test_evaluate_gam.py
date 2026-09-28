import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_gam import evaluate, validate_spec, _pygam_terms


class EvaluatorTests(unittest.TestCase):
    def test_smoothing_settings_reach_smooth_and_tensor_terms(self):
        spec = {"main_terms": [{"feature": "T", "type": "s"}], "interactions": [["T", "c"]]}
        default = validate_spec(spec, ["T", "c"])
        self.assertEqual(default["smoothing"], {"n_splines": 5, "lam": 0.6})
        tuned = validate_spec({**spec, "smoothing": {"n_splines": 8, "lam": 10}}, ["T", "c"])
        terms = _pygam_terms(tuned, ["T", "c"])
        self.assertEqual(terms[0].n_splines, 8)
        self.assertEqual(terms[0].lam, [10])
        self.assertEqual(terms[1].n_splines, [8, 8])
        self.assertEqual(terms[1].lam, [[10], [10]])
        for setting in ({"n_splines": 3, "lam": 1}, {"n_splines": 5, "lam": -1}, {"n_splines": 5, "lam": float("nan")}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                validate_spec({**spec, "smoothing": setting}, ["T", "c"])

    def test_two_doi_salt_uses_two_folds_without_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prep = root / "prep"
            prep.mkdir()
            (prep / "run_manifest.json").write_text(
                json.dumps({"dataset_file": "data.csv"}), encoding="utf-8"
            )
            (prep / "audit.json").write_text(
                json.dumps({"salts": {"LiBOB": {
                    "eligible": True, "selected_solvents": ["PC"],
                    "reference_solvent": "PC",
                }}}), encoding="utf-8"
            )
            with (prep / "data.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["row_id", "doi", "salt", "k", "T", "c", "PC"])
                writer.writeheader()
                for index in range(40):
                    writer.writerow({
                        "row_id": index + 1, "doi": "A" if index < 20 else "B",
                        "salt": "LiBOB", "k": 2 + 0.03 * index,
                        "T": 260 + index, "c": 0.5 + 0.1 * (index % 4), "PC": 1,
                    })
            with (prep / "splits.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["row_id", "salt", "doi", "partition", "fold"])
                writer.writeheader()
                for index in range(40):
                    writer.writerow({
                        "row_id": index + 1, "salt": "LiBOB",
                        "doi": "A" if index < 20 else "B",
                        "partition": "development", "fold": index // 20,
                    })
            result = evaluate(
                prep, "LiBOB", {"main_terms": [{"feature": "T", "type": "l"}], "interactions": []},
                root / "result",
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["folds"]), 2)
            self.assertTrue(all(math.isfinite(fold["rmse"]) for fold in result["folds"]))
            with (root / "result" / "oof_predictions.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 40)

    def test_scores_development_folds_without_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prep = root / "prep"
            prep.mkdir()
            (prep / "run_manifest.json").write_text(
                json.dumps({"dataset_file": "data.csv"}), encoding="utf-8"
            )
            (prep / "audit.json").write_text(
                json.dumps(
                    {"salts": {"LiPF6": {
                        "eligible": True,
                        "selected_solvents": ["PC", "EC"],
                        "reference_solvent": "PC",
                    }}}
                ),
                encoding="utf-8",
            )
            with (prep / "data.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["row_id", "doi", "salt", "k", "T", "c", "PC", "EC"],
                )
                writer.writeheader()
                for index in range(28):
                    writer.writerow({
                        "row_id": index + 1,
                        "doi": "ABCD"[index // 7],
                        "salt": "LiPF6",
                        "k": 2 + 0.02 * index + 0.3 * (index % 5),
                        "T": 260 + index,
                        "c": 0.5 + 0.1 * (index % 2),
                        "PC": 0.7,
                        "EC": 0.3,
                    })
            with (prep / "splits.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file, fieldnames=["row_id", "salt", "doi", "partition", "fold"]
                )
                writer.writeheader()
                for index in range(28):
                    writer.writerow({
                        "row_id": index + 1,
                        "salt": "LiPF6",
                        "doi": "ABCD"[index // 7],
                        "partition": "holdout" if index >= 21 else "development",
                        "fold": "" if index >= 21 else index // 7,
                    })
            spec = {
                "main_terms": [
                    {"feature": "T", "type": "s"},
                    {"feature": "c", "type": "l"},
                ],
                "interactions": [],
            }
            result = evaluate(prep, "LiPF6", spec, root / "result")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["folds"]), 3)
            self.assertTrue(all(math.isfinite(fold["rmse"]) for fold in result["folds"]))
            with (root / "result" / "oof_predictions.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                predictions = list(csv.DictReader(file))
            self.assertEqual(len(predictions), 21)
            self.assertTrue(all(row["doi"] != "D" for row in predictions))

            interaction_spec = {
                **spec,
                "interactions": [["T", "c"]],
            }
            interaction_result = evaluate(
                prep, "LiPF6", interaction_spec, root / "interaction"
            )
            self.assertEqual(interaction_result["status"], "failed")
            self.assertIn("insufficient variation in c", interaction_result["reason"])


if __name__ == "__main__":
    unittest.main()
