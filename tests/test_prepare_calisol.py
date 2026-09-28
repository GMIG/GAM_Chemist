import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_calisol import convert_fractions, make_splits, prepare, select_solvents


class PreparationTests(unittest.TestCase):
    def test_conversion_and_outside_solvent_exclusion(self):
        constants = {
            "EC": {"molar_mass": 88.06, "density": 1.321},
            "DEC": {"molar_mass": 118.132, "density": 0.975},
        }
        for ratio_type, ec_amount, dec_amount in (
            ("w", 0.5 / 88.06, 0.5 / 118.132),
            ("v", 0.5 * 1.321 / 88.06, 0.5 * 0.975 / 118.132),
            ("mol", 0.5, 0.5),
        ):
            with self.subTest(ratio_type=ratio_type):
                fractions = convert_fractions(
                    {"EC": 0.5, "DEC": 0.5}, ratio_type, constants
                )
                self.assertTrue(math.isclose(sum(fractions.values()), 1.0))
                self.assertAlmostEqual(
                    fractions["EC"], ec_amount / (ec_amount + dec_amount)
                )
        with self.assertRaisesRegex(ValueError, "Missing solvent constants"):
            convert_fractions({"EC": 1.0}, "mol", {})

        solvents = list("ABCDEF")
        rows = [
            {"salt": "S", **{name: float(name == present) for name in solvents}}
            for present in list("AABBCCDDEEF")
        ]
        retained, selected = select_solvents(rows, solvents)
        self.assertEqual(selected["S"], list("ABCDE"))
        self.assertEqual(len(retained), 10)
        self.assertTrue(all(row["F"] == 0 for row in retained))

    def test_split_keeps_publications_together(self):
        rows = []
        for doi, size in (("A", 8), ("B", 6), ("C", 4), ("D", 3), ("E", 2)):
            rows.extend(
                {"row_id": len(rows) + 1, "salt": "S", "doi": doi}
                for _ in range(size)
            )
        splits = make_splits(rows, {}, min_rows=1, min_dois=5)
        self.assertEqual(splits, make_splits(rows, {}, min_rows=1, min_dois=5))
        by_doi = {}
        for split in splits:
            by_doi.setdefault(split["doi"], set()).add(
                (split["partition"], split["fold"])
            )
        self.assertEqual(by_doi["C"], {("holdout", "")})
        self.assertTrue(all(len(assignments) == 1 for assignments in by_doi.values()))
        self.assertEqual(
            {split["fold"] for split in splits if split["partition"] == "development"},
            {0, 1, 2},
        )

    def test_two_doi_salt_has_two_folds_without_holdout(self):
        rows = [
            {"row_id": index + 1, "salt": "S", "doi": "A" if index < 4 else "B"}
            for index in range(8)
        ]
        splits = make_splits(rows, {}, min_rows=1)
        self.assertEqual(len(splits), 8)
        self.assertEqual({row["partition"] for row in splits}, {"development"})
        self.assertEqual({row["fold"] for row in splits}, {0, 1})
        self.assertEqual(len({row["fold"] for row in splits if row["doi"] == "A"}), 1)
        self.assertEqual(len({row["fold"] for row in splits if row["doi"] == "B"}), 1)

    def test_three_doi_salt_has_holdout_and_two_folds(self):
        rows = [
            {"row_id": index + 1, "salt": "S", "doi": "ABC"[index // 4]}
            for index in range(12)
        ]
        splits = make_splits(rows, {}, min_rows=1)
        self.assertEqual({row["doi"] for row in splits if row["partition"] == "holdout"}, {"A"})
        self.assertEqual(
            {row["fold"] for row in splits if row["partition"] == "development"},
            {0, 1},
        )

    def test_artifacts_record_filter_reasons_and_settings(self):
        constants_path = Path("scripts/solvent_constants.json")
        solvents = list(json.loads(constants_path.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            with source.open("w", newline="", encoding="utf-8") as file:
                fields = ["doi", "k", "T", "c", "salt", "c units", "solvent ratio type", *solvents]
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for k, units in ((1, "mol/kg"), (0, "mol/kg"), (1, "mol/L")):
                    writer.writerow({
                        "doi": "source-1", "k": k, "T": 298, "c": 1,
                        "salt": "S", "c units": units, "solvent ratio type": "w",
                        **{name: int(name == "EC") for name in solvents},
                    })
            prepare(source, root / "out")
            audit = json.loads((root / "out" / "audit.json").read_text())
            manifest = json.loads((root / "out" / "run_manifest.json").read_text())
            self.assertEqual(audit["salts"]["S"]["source_rows"], 3)
            self.assertEqual(
                audit["salts"]["S"]["exclusions"],
                {"concentration_unit": 1, "non_positive_conductivity": 1},
            )
            self.assertEqual(manifest["source_rows"], 3)
            self.assertEqual(manifest["preprocessing"]["eligibility_min_dois"], 2)
            self.assertEqual(manifest["reference_solvents"]["S"], "EC")
            self.assertEqual(
                manifest["constants_sha256"],
                hashlib.sha256(constants_path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
