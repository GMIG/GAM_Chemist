"""Prepare CALiSol-23 measurements for per-salt GAM experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path


SOURCE_URL = "https://data.dtu.dk/articles/dataset/CALiSol-23/24559960"
CONSTANTS_URL = "https://github.com/Pele0599/CALiSol-23/blob/main/CALiSol-23.ipynb"
VERSION = "v2"


def make_splits(
    rows: list[dict], _audit: dict, min_rows: int = 200, min_dois: int = 2
) -> list[dict]:
    """Assign whole publications to development folds and an optional holdout."""
    from sklearn.model_selection import GroupKFold

    by_salt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_salt[row["salt"]].append(row)

    assignments: dict[int, dict] = {}
    for salt, salt_rows in by_salt.items():
        by_doi: dict[str, list[dict]] = defaultdict(list)
        for row in salt_rows:
            by_doi[row["doi"]].append(row)
        if len(salt_rows) < min_rows or len(by_doi) < min_dois:
            continue
        holdout_doi = (
            min(
                by_doi,
                key=lambda doi: (abs(len(by_doi[doi]) - 0.2 * len(salt_rows)), doi),
            )
            if len(by_doi) >= 3 else None
        )
        development = [row for row in salt_rows if row["doi"] != holdout_doi]
        groups = [row["doi"] for row in development]
        n_folds = min(3, len(set(groups)))
        folds = {}
        for fold, (_, validation_indices) in enumerate(
            GroupKFold(n_splits=n_folds).split(development, groups=groups)
        ):
            for index in validation_indices:
                folds[development[index]["row_id"]] = fold
        for row in salt_rows:
            holdout = row["doi"] == holdout_doi
            assignments[row["row_id"]] = {
                "row_id": row["row_id"],
                "salt": salt,
                "doi": row["doi"],
                "partition": "holdout" if holdout else "development",
                "fold": "" if holdout else folds[row["row_id"]],
            }
        doi_assignments: dict[str, set[tuple[str, int | str]]] = defaultdict(set)
        for row in salt_rows:
            split = assignments[row["row_id"]]
            doi_assignments[row["doi"]].add((split["partition"], split["fold"]))
        if any(len(group_assignments) != 1 for group_assignments in doi_assignments.values()):
            raise AssertionError(f"DOI crosses splits for {salt}")
    return [assignments[row["row_id"]] for row in rows if row["row_id"] in assignments]


def convert_fractions(
    ratios: dict[str, float], ratio_type: str, constants: dict
) -> dict[str, float]:
    """Convert one complete solvent mixture to mole fractions."""
    if ratio_type not in {"w", "v", "mol"}:
        raise ValueError(f"Unknown solvent ratio type: {ratio_type!r}")
    amounts = {}
    for name, ratio in ratios.items():
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError(f"Invalid ratio for {name}: {ratio}")
        if ratio > 0 and name not in constants:
            raise ValueError(f"Missing solvent constants for {name}")
        if ratio == 0:
            amounts[name] = 0.0
        elif ratio_type == "mol":
            amounts[name] = ratio
        else:
            mass = constants[name]["molar_mass"]
            density = constants[name]["density"] if ratio_type == "v" else 1.0
            amounts[name] = ratio * density / mass
    total = sum(amounts.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Solvent mixture has no finite positive amount")
    return {name: amount / total for name, amount in amounts.items()}


def select_solvents(
    rows: list[dict], solvent_names: list[str]
) -> tuple[list[dict], dict[str, list[str]]]:
    """Keep mixtures composed only of each salt's five most common solvents."""
    frequencies: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        frequencies[row["salt"]].update(
            name for name in solvent_names if row[name] > 0
        )
    selected = {
        salt: [name for name, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:5]]
        for salt, counts in frequencies.items()
    }
    retained = [
        row
        for row in rows
        if all(row[name] == 0 for name in solvent_names if name not in selected[row["salt"]])
    ]
    return retained, selected


def prepare(input_path: Path, output_dir: Path) -> Path:
    """Write the reproducible filtered CSV, audit, grouped splits, and card."""
    input_bytes = input_path.read_bytes()
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    constants_path = Path(__file__).parent / "solvent_constants.json"
    constants_bytes = constants_path.read_bytes()
    constants = json.loads(constants_bytes.decode("utf-8"))
    solvent_names = list(constants)
    reader = csv.DictReader(io.StringIO(input_bytes.decode("utf-8-sig"), newline=""))
    required = {
        "doi", "k", "T", "c", "salt", "c units", "solvent ratio type", *solvent_names
    }
    missing = sorted(required - set(reader.fieldnames or []))
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

    exclusions = Counter()
    exclusions_by_salt: dict[str, Counter] = defaultdict(Counter)
    source_by_salt = Counter()
    raw_by_salt = Counter()
    quality_by_salt = Counter()
    valid_rows = []
    source_rows = 0

    def exclude(reason: str, salt: str) -> None:
        exclusions[reason] += 1
        if salt:
            exclusions_by_salt[salt][reason] += 1

    for row_id, source in enumerate(reader, start=1):
        source_rows += 1
        salt = source["salt"].strip()
        if salt:
            source_by_salt[salt] += 1
        if source["c units"] != "mol/kg":
            exclude("concentration_unit", salt)
            continue
        if not salt:
            exclude("missing_salt", salt)
            continue
        raw_by_salt[salt] += 1
        doi = source["doi"].strip()
        if not doi:
            exclude("missing_doi", salt)
            continue
        try:
            k, temperature, concentration = (
                float(source[name]) for name in ("k", "T", "c")
            )
            ratios = {name: float(source[name]) for name in solvent_names}
        except (TypeError, ValueError):
            exclude("non_numeric", salt)
            continue
        if not all(math.isfinite(value) for value in (k, temperature, concentration)):
            exclude("non_finite_measurement", salt)
            continue
        if k <= 0:
            exclude("non_positive_conductivity", salt)
            continue
        if concentration < 0:
            exclude("negative_concentration", salt)
            continue
        if any(not math.isfinite(value) or value < 0 for value in ratios.values()):
            exclude("invalid_solvent_ratio", salt)
            continue
        if abs(sum(ratios.values()) - 1.0) > 0.03:
            exclude("solvent_ratio_sum", salt)
            continue
        ratio_type = source["solvent ratio type"].strip()
        fractions = convert_fractions(ratios, ratio_type, constants)
        if abs(sum(fractions.values()) - 1.0) > 1e-9:
            raise ValueError(f"Converted fractions do not sum to one at row {row_id}")
        valid_rows.append(
            {
                "row_id": row_id,
                "doi": doi,
                "salt": salt,
                "k": k,
                "T": temperature,
                "c": concentration,
                "source_ratio_type": ratio_type,
                **fractions,
            }
        )
        quality_by_salt[salt] += 1

    retained, selected = select_solvents(valid_rows, solvent_names)
    solvent_frequencies: dict[str, Counter] = defaultdict(Counter)
    for row in valid_rows:
        solvent_frequencies[row["salt"]].update(
            name for name in solvent_names if row[name] > 0
        )
    retained_by_salt = Counter(row["salt"] for row in retained)
    dois = defaultdict(set)
    for row in retained:
        dois[row["salt"]].add(row["doi"])
    salt_audit = {}
    for salt in source_by_salt:
        outside_count = quality_by_salt[salt] - retained_by_salt[salt]
        if outside_count:
            exclusions["outside_selected_solvents"] += outside_count
            exclusions_by_salt[salt]["outside_selected_solvents"] += outside_count
    for salt in sorted(source_by_salt):
        solvents = selected.get(salt, [])
        count = retained_by_salt[salt]
        doi_count = len(dois[salt])
        eligible = count >= 200 and doi_count >= 2
        salt_audit[salt] = {
            "source_rows": source_by_salt[salt],
            "molkg_rows": raw_by_salt[salt],
            "quality_rows": quality_by_salt[salt],
            "retained_rows": count,
            "excluded_outside_selected_solvents": quality_by_salt[salt] - count,
            "exclusions": dict(sorted(exclusions_by_salt[salt].items())),
            "distinct_dois": doi_count,
            "selected_solvents": solvents,
            "solvent_frequencies": dict(sorted(solvent_frequencies[salt].items())),
            "reference_solvent": solvents[0] if solvents else None,
            "eligible": eligible,
            "development_folds": (2 if doi_count == 2 else min(3, doi_count - 1)) if eligible else 0,
            "has_final_holdout": eligible and doi_count >= 3,
        }
    audit = {
        "source_rows": source_rows,
        "quality_rows": len(valid_rows),
        "retained_rows": len(retained),
        "exclusions": dict(sorted(exclusions.items())),
        "salts": salt_audit,
    }
    splits = make_splits(retained, audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = f"calisol23_molkg_top5_molefrac_{VERSION}_{input_hash[:12]}.csv"
    dataset_path = output_dir / dataset_name
    with dataset_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["row_id", "doi", "salt", "k", "T", "c", "source_ratio_type", *solvent_names],
        )
        writer.writeheader()
        writer.writerows(retained)
    with (output_dir / "splits.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["row_id", "salt", "doi", "partition", "fold"]
        )
        writer.writeheader()
        writer.writerows(splits)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_url": SOURCE_URL,
        "source_sha256": input_hash,
        "source_rows": source_rows,
        "constants_source_url": CONSTANTS_URL,
        "constants_sha256": hashlib.sha256(constants_bytes).hexdigest(),
        "processing_version": VERSION,
        "dataset_file": dataset_name,
        "reference_solvents": {
            salt: details["reference_solvent"]
            for salt, details in salt_audit.items()
            if details["retained_rows"]
        },
        "split_by_salt": {
            salt: {
                "development_folds": details["development_folds"],
                "has_final_holdout": details["has_final_holdout"],
            }
            for salt, details in salt_audit.items() if details["eligible"]
        },
        "preprocessing": {
            "concentration_unit": "mol/kg",
            "minimum_conductivity_exclusive": 0,
            "minimum_concentration_inclusive": 0,
            "raw_ratio_sum_tolerance": 0.03,
            "mole_fraction_sum_tolerance": 1e-9,
            "max_solvents_per_salt": 5,
            "eligibility_min_rows": 200,
            "eligibility_min_dois": 2,
            "holdout_target_fraction": 0.2,
            "max_development_folds": 3,
        },
        "concentration_unit": "mol/kg",
        "conductivity_unit": "mS/cm",
        "temperature_unit": "K",
        "solvent_unit": "mole fraction",
        "split_rule": "2 DOIs: 2 development folds, no holdout; 3 DOIs: 1 holdout, 2 folds; 4+ DOIs: 1 holdout, 3 folds",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    card = [
        "# CALiSol-23 filtered dataset",
        "",
        f"Dataset file: `{dataset_name}`",
        f"Source: {SOURCE_URL}",
        f"Source SHA-256: `{input_hash}`",
        f"Processing version: `{VERSION}`",
        f"Retained measurements: {len(retained):,}",
        "",
        "## Preparation",
        "",
        "Keep mol/kg rows with finite T and c, positive k, and a valid solvent mixture. "
        "Convert all solvent ratios to mole fractions; for each salt retain mixtures "
        "composed only of its up to five most frequent solvents.",
        "Units: k in mS/cm; T in K; c in mol/kg; solvents as mole fractions.",
        "Eligibility: at least 200 retained rows and two distinct DOIs.",
        "",
        "## Salt summary",
        "",
        "| Salt | Retained rows | Distinct DOIs | Eligible |",
        "| --- | ---: | ---: | --- |",
    ]
    for salt, details in salt_audit.items():
        if not details["retained_rows"]:
            continue
        card.append(
            f"| {salt} | {details['retained_rows']:,} | {details['distinct_dois']} | "
            f"{'Yes' if details['eligible'] else 'No'} |"
        )
    card.extend(["", "## Solvents by salt", ""])
    for salt, details in salt_audit.items():
        if not details["retained_rows"]:
            continue
        solvents = ", ".join(details["selected_solvents"]) or "none"
        reference = details["reference_solvent"] or "none"
        card.append(f"- {salt}: {solvents}. Reference: {reference}.")
    card.extend(["", "## Evaluation splits", ""])
    for salt, details in salt_audit.items():
        if details["eligible"]:
            holdout = "one DOI holdout" if details["has_final_holdout"] else "no final holdout"
            card.append(f"- {salt}: {details['development_folds']} DOI-grouped folds; {holdout}.")
    card.extend(
        [
            "",
            "DOI is retained for grouped splitting, not prediction. `splits.csv` holds "
            "development folds and any final holdout. A final holdout is not scored "
            "during development.",
            "",
        ]
    )
    (output_dir / "dataset_card.md").write_text("\n".join(card), encoding="utf-8")
    return dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local CALiSol-23 CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset_path = prepare(args.input, args.output_dir)
    print(f"Wrote {dataset_path}")


if __name__ == "__main__":
    main()
