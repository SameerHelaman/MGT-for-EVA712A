# =============================================================================
# MODULE: utils/generate_scaffold_cv.py
# PURPOSE: Freeze five outer scaffold folds and fold-specific inner validation sets.
# LIBRARIES: Python standard-library modules provide CSV, hashing, JSON and paths;
#            NumPy supports deterministic numerical balancing of fold labels.
# OUTPUT: One compound assignment CSV, five structure-level split manifests and
#         one checksummed metadata JSON under experiment_a_ligand_mgt/cv/.
# CALCULATIONS: Whole Bemis-Murcko scaffold groups are greedily balanced by
#               compound count and mean pKD without ever splitting a scaffold.
# =============================================================================
"""Generate deterministic, leakage-free five-fold scaffold cross-validation manifests."""

from __future__ import annotations  # Postpone type-annotation evaluation for compatibility.

import argparse  # Parse reproducible command-line options.
import csv  # Read and write human-auditable manifest tables.
import hashlib  # Calculate SHA-256 checksums for frozen artifacts.
import json  # Persist machine-readable metadata and validation results.
import random  # Resolve equal-sized scaffold ordering deterministically from the seed.
from collections import defaultdict  # Group compounds and rows without manual key initialization.
from pathlib import Path  # Manipulate platform-independent input and output paths.

import numpy as np  # Calculate label means, scales and fold-balancing objectives.
from rdkit import Chem  # Parse canonical SMILES and remove structure-specific stereochemical assignments.
from rdkit.Chem.Scaffolds import MurckoScaffold  # Calculate standard achiral Bemis-Murcko frameworks.

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Locate the MGT repository from this utility file.
DEFAULT_DATA_ROOT = PROJECT_ROOT / "OpenBind_EV-A71_2A" / "experiment_a_ligand_mgt"  # Point to the curated experiment.
DEFAULT_SOURCE = DEFAULT_DATA_ROOT / "splits" / "scaffold_seed_123.csv"  # Reuse the audited 621-structure manifest as source data.


def sha256_file(path: Path) -> str:  # Define a helper that fingerprints one frozen artifact.
    """Return the SHA-256 checksum of one file without loading it all into memory."""
    digest = hashlib.sha256()  # Create a fresh SHA-256 accumulator.
    with path.open("rb") as handle:  # Open the artifact in binary mode for stable cross-platform bytes.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Stream one-megabyte blocks until end-of-file.
            digest.update(block)  # Add this block to the running cryptographic digest.
    return digest.hexdigest()  # Return the final lowercase hexadecimal checksum.


def read_csv(path: Path) -> list[dict[str, str]]:  # Define a typed CSV reader used for the source manifest.
    """Read a complete CSV manifest as dictionaries keyed by its header."""
    with path.open("r", encoding="utf-8", newline="") as handle:  # Open text with explicit reproducible encoding/newline handling.
        return list(csv.DictReader(handle))  # Materialize rows because validation repeatedly traverses them.


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:  # Define one stable manifest writer.
    """Write rows with a fixed column order and Unix-compatible newline handling."""
    path.parent.mkdir(parents=True, exist_ok=True)  # Create the CV directory before writing its first artifact.
    with path.open("w", encoding="utf-8", newline="") as handle:  # Open the destination without platform-added blank rows.
        writer = csv.DictWriter(handle, fieldnames=fields)  # Freeze the public schema to the supplied field order.
        writer.writeheader()  # Write descriptive column names as the first record.
        writer.writerows(rows)  # Write every structure or compound record in deterministic order.


def compound_table(rows: list[dict[str, str]]) -> list[dict[str, object]]:  # Collapse repeated structures to unique compounds.
    """Validate compound metadata consistency and return one record per compound."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)  # Collect all crystallographic rows sharing a compound identity.
    for row in rows:  # Traverse every curated structure record exactly once.
        grouped[row["official_compound_group_id"]].append(row)  # Attach the structure to its official compound group.
    compounds: list[dict[str, object]] = []  # Prepare the unique-compound result table.
    for compound_id, members in sorted(grouped.items()):  # Process compounds deterministically by identifier.
        smiles = {member["canonical_smiles"] for member in members}  # Collect structure-derived stereochemical identities across repeats.
        labels = {float(member["experimental_pKD"]) for member in members}  # Collect affinity labels across repeats.
        molecules = [Chem.MolFromSmiles(smiles_value) for smiles_value in sorted(smiles)]  # Parse every observed stereochemical representation.
        if any(molecule is None for molecule in molecules):  # Require every curated identity to remain RDKit-readable.
            raise ValueError(f"Compound {compound_id} contains an invalid canonical SMILES")  # Reject an unscaffoldable compound.
        achiral_identities = {Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True) for molecule in molecules}  # Remove pose-specific stereochemistry.
        scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False) for molecule in molecules}  # Recompute achiral frameworks.
        if len(achiral_identities) != 1 or len(scaffolds) != 1 or "" in scaffolds:  # Require one constitution and one non-empty framework.
            raise ValueError(f"Compound {compound_id} has inconsistent achiral identity or scaffold")  # Stop before ambiguous fold allocation.
        if len(labels) != 1:  # Require one frozen affinity label across repeated structures.
            raise ValueError(f"Compound {compound_id} has inconsistent pKD")  # Reject scientifically inconsistent grouping.
        compounds.append({  # Store the single validated compound-level record.
            "official_compound_group_id": compound_id,  # Preserve the official grouping key used by evaluation.
            "canonical_smiles": sorted(smiles)[0],  # Retain one deterministic isomeric display identity for reporting.
            "scaffold": next(iter(scaffolds)),  # Store the recomputed achiral indivisible Bemis-Murcko framework.
            "experimental_pKD": next(iter(labels)),  # Store the unique experimental regression target.
            "structure_count": len(members),  # Record how many crystal structures represent this compound.
        })  # Finish this compound record.
    return compounds  # Return all unique compounds for fold allocation.


def assign_scaffold_groups(compounds: list[dict[str, object]], folds: int, seed: int) -> dict[str, int]:  # Balance indivisible scaffolds.
    """Assign whole scaffold groups to folds using count and pKD-distribution objectives."""
    if folds < 2:  # Cross-validation requires at least two non-overlapping test partitions.
        raise ValueError("folds must be at least two")  # Reject an invalid cross-validation request.
    by_scaffold: dict[str, list[dict[str, object]]] = defaultdict(list)  # Collect compounds by indivisible scaffold.
    for compound in compounds:  # Traverse each unique compound.
        by_scaffold[str(compound["scaffold"])].append(compound)  # Attach it to exactly one scaffold family.
    if len(by_scaffold) < folds:  # Every fold requires at least one independent scaffold family.
        raise ValueError("Fewer scaffold groups than requested folds")  # Stop instead of producing empty test folds.
    rng = random.Random(seed)  # Create an isolated deterministic random-number generator.
    ordering = list(by_scaffold)  # Copy scaffold identifiers before deterministic shuffling.
    rng.shuffle(ordering)  # Randomize ties reproducibly rather than depending on CSV order.
    ordering.sort(key=lambda scaffold: len(by_scaffold[scaffold]), reverse=True)  # Place large scaffold families first for better balance.
    target_count = len(compounds) / folds  # Define the ideal number of compounds per fold.
    labels = np.asarray([float(item["experimental_pKD"]) for item in compounds], dtype=float)  # Form the global target vector.
    global_mean = float(labels.mean())  # Calculate the overall affinity mean used by the balancing penalty.
    global_scale = max(float(labels.std(ddof=0)), 1e-8)  # Calculate a nonzero scale for dimensionless label imbalance.
    fold_members: list[list[dict[str, object]]] = [[] for _ in range(folds)]  # Hold compounds already allocated to each fold.
    assignment: dict[str, int] = {}  # Map every scaffold identifier to its selected fold.
    for scaffold in ordering:  # Allocate one complete scaffold family at a time.
        group = by_scaffold[scaffold]  # Retrieve all compounds that must move together.
        best_fold = None  # Defer the decision until all candidate folds are scored.
        best_score = None  # Track the smallest count-plus-label imbalance objective.
        for fold in range(folds):  # Evaluate placement into each possible fold.
            candidate = fold_members[fold] + group  # Simulate adding the complete scaffold to this fold.
            candidate_labels = np.asarray([float(item["experimental_pKD"]) for item in candidate])  # Extract candidate pKD values.
            count_penalty = len(fold_members[fold]) / target_count  # Measure current occupancy before adding the indivisible scaffold.
            mean_penalty = ((float(candidate_labels.mean()) - global_mean) / global_scale) ** 2  # Measure the candidate fold's pKD shift.
            candidate_key = (count_penalty, mean_penalty, fold)  # Prioritize the least-populated fold, then label balance and fold number.
            if best_score is None or candidate_key < best_score:  # Retain the best placement observed so far.
                best_score = candidate_key  # Store this fold's objective for later comparisons.
                best_fold = fold  # Store this fold as the current allocation choice.
        assert best_fold is not None  # Confirm that at least one valid candidate fold was evaluated.
        fold_members[best_fold].extend(group)  # Commit the complete scaffold family to its chosen fold.
        assignment[scaffold] = best_fold  # Freeze the scaffold-to-fold mapping.
    return assignment  # Return mappings used to construct structure-level manifests.


def validate_manifest(rows: list[dict[str, object]]) -> dict[str, object]:  # Audit one outer-fold train/validation/test manifest.
    """Assert compound and scaffold disjointness and return partition counts."""
    partitions = ("train", "validation", "test")  # Fix the only permitted supervised partitions.
    compound_sets = {part: {str(row["official_compound_group_id"]) for row in rows if row["split"] == part} for part in partitions}  # Collect compound identities.
    scaffold_sets = {part: {str(row["scaffold"]) for row in rows if row["split"] == part} for part in partitions}  # Collect scaffold identities.
    if any(not compound_sets[part] for part in partitions):  # Every outer experiment requires non-empty fitting, selection and test sets.
        raise AssertionError("Train, validation and test partitions must all be non-empty")  # Reject unusable manifests immediately.
    for left_index, left in enumerate(partitions):  # Compare each partition with those following it.
        for right in partitions[left_index + 1 :]:  # Avoid duplicate symmetric comparisons.
            if compound_sets[left] & compound_sets[right]:  # Detect any repeated compound crossing partitions.
                raise AssertionError(f"Compound leakage between {left} and {right}")  # Reject the invalid manifest.
            if scaffold_sets[left] & scaffold_sets[right]:  # Detect any Bemis-Murcko scaffold crossing partitions.
                raise AssertionError(f"Scaffold leakage between {left} and {right}")  # Reject the invalid manifest.
    return {  # Report both structure- and compound-level partition sizes.
        part: {  # Create one nested count record for this partition.
            "structures": sum(row["split"] == part for row in rows),  # Count crystallographic structure records.
            "compounds": len(compound_sets[part]),  # Count unique official compound groups.
            "scaffolds": len(scaffold_sets[part]),  # Count unique Bemis-Murcko scaffold families.
        }  # Finish the partition count record.
        for part in partitions  # Repeat the count calculation for train, validation and test.
    }  # Return the audit summary.


def generate(data_root: Path, source: Path, output_dir: Path, folds: int, seed: int) -> dict[str, object]:  # Run the complete freeze operation.
    """Generate outer and inner scaffold assignments, validate them and freeze checksums."""
    source_rows = read_csv(source)  # Load all 621 curated crystallographic records.
    compounds = compound_table(source_rows)  # Collapse repeated structures for leakage-safe allocation.
    outer_assignment = assign_scaffold_groups(compounds, folds, seed)  # Allocate every scaffold to one outer test fold.
    compound_outer = {str(item["official_compound_group_id"]): outer_assignment[str(item["scaffold"])] for item in compounds}  # Map compounds to outer folds.
    scaffold_by_compound = {str(item["official_compound_group_id"]): str(item["scaffold"]) for item in compounds}  # Map every official group to its achiral framework.
    assignment_rows = [  # Build a concise compound-level outer-fold table.
        {**item, "outer_fold": compound_outer[str(item["official_compound_group_id"])]}  # Add the frozen test-fold index.
        for item in sorted(compounds, key=lambda row: str(row["official_compound_group_id"]))  # Preserve deterministic row ordering.
    ]  # Finish the compound assignment table.
    assignment_path = output_dir / f"scaffold_grouped_{folds}fold_seed_{seed}.csv"  # Define the master assignment artifact.
    write_csv(assignment_path, assignment_rows, list(assignment_rows[0]))  # Persist the master outer-fold mapping.
    fold_records: list[dict[str, object]] = []  # Collect metadata for all outer-fold experiments.
    expected_test_compounds: set[str] = set()  # Track that each compound becomes test data exactly once.
    for outer_fold in range(folds):  # Construct one independent train/validation/test experiment per outer fold.
        development = [item for item in compounds if compound_outer[str(item["official_compound_group_id"])] != outer_fold]  # Exclude the locked outer test fold.
        inner_assignment = assign_scaffold_groups(development, folds, seed + 1000 + outer_fold)  # Split development scaffolds into five inner groups.
        validation_inner_fold = outer_fold % folds  # Select one balanced inner group as validation deterministically.
        split_by_compound: dict[str, str] = {}  # Map every compound to its role in this outer experiment.
        for item in compounds:  # Assign train, validation or test without inspecting test performance.
            compound_id = str(item["official_compound_group_id"])  # Normalize the official identifier to text.
            scaffold = str(item["scaffold"])  # Read the compound's indivisible scaffold family.
            if compound_outer[compound_id] == outer_fold:  # Identify compounds held out by the outer loop.
                split_by_compound[compound_id] = "test"  # Lock this compound for final fold evaluation.
                expected_test_compounds.add(compound_id)  # Record its single out-of-fold evaluation membership.
            elif inner_assignment[scaffold] == validation_inner_fold:  # Identify the selected inner validation scaffold group.
                split_by_compound[compound_id] = "validation"  # Use it only for checkpoint selection and early stopping.
            else:  # Remaining development scaffolds are eligible for fitting.
                split_by_compound[compound_id] = "train"  # Use this compound for optimization and target normalization.
        fold_rows: list[dict[str, object]] = []  # Prepare the structure-level manifest expected by existing trainers.
        for row in sorted(source_rows, key=lambda item: item["complex_name"]):  # Preserve all repeated structures in deterministic order.
            compound_id = row["official_compound_group_id"]  # Read the structure's official compound group.
            fold_rows.append({  # Copy the established split schema and replace only membership.
                "complex_name": row["complex_name"],  # Preserve the structure identifier used by datasets.
                "official_compound_group_id": compound_id,  # Preserve compound grouping for evaluation.
                "canonical_smiles": row["canonical_smiles"],  # Preserve audited molecular identity.
                "scaffold": scaffold_by_compound[compound_id],  # Use the recomputed achiral scaffold enforced by CV.
                "experimental_pKD": row["experimental_pKD"],  # Preserve the frozen regression target.
                "split": split_by_compound[compound_id],  # Assign this structure according to its compound/scaffold group.
                "split_method": "scaffold",  # Retain compatibility with existing trainer validation.
                "seed": seed,  # Retain the shared experiment seed expected by trainers.
                "outer_fold": outer_fold,  # Record which fold acts as untouched test data.
                "inner_validation_fold": validation_inner_fold,  # Record the deterministic validation group.
            })  # Finish this structure-level membership row.
        counts = validate_manifest(fold_rows)  # Assert zero compound and scaffold leakage before writing.
        manifest_path = output_dir / f"scaffold_cv{folds}_fold_{outer_fold}_seed_{seed}.csv"  # Define this fold's trainer-compatible manifest.
        write_csv(manifest_path, fold_rows, list(fold_rows[0]))  # Freeze exact structure membership for this experiment.
        fold_records.append({  # Add this fold to the metadata report.
            "outer_fold": outer_fold,  # Identify the held-out fold.
            "inner_validation_fold": validation_inner_fold,  # Identify the inner group used for early stopping.
            "manifest": str(manifest_path.resolve()),  # Record the absolute manifest location.
            "sha256": sha256_file(manifest_path),  # Fingerprint exact row membership and ordering.
            "counts": counts,  # Store structure, compound and scaffold partition counts.
            "compound_leakage_zero": True,  # Record the successfully asserted compound-separation invariant.
            "scaffold_leakage_zero": True,  # Record the successfully asserted scaffold-separation invariant.
        })  # Finish this fold metadata record.
    if expected_test_compounds != {str(item["official_compound_group_id"]) for item in compounds}:  # Verify complete out-of-fold coverage.
        raise AssertionError("Each compound must occur in exactly one outer test fold")  # Reject incomplete outer assignments.
    metadata = {  # Build the complete reproducibility record.
        "method": "five-fold grouped Bemis-Murcko scaffold cross-validation",  # Name the scientific split design.
        "folds": folds,  # Record the number of outer test partitions.
        "seed": seed,  # Record the deterministic assignment seed.
        "source_manifest": str(source.resolve()),  # Link the audited source dataset membership.
        "source_sha256": sha256_file(source),  # Freeze the exact source manifest bytes.
        "assignment_manifest": str(assignment_path.resolve()),  # Link the compound-level outer-fold table.
        "assignment_sha256": sha256_file(assignment_path),  # Freeze the exact compound assignment bytes.
        "scaffold_method": "RDKit Bemis-Murcko scaffold recomputed with includeChirality=False from canonical SMILES",  # Document scaffold provenance.
        "outer_test_rule": "each complete scaffold family appears in one outer fold",  # Document held-out grouping.
        "inner_validation_rule": "one of five balanced scaffold groups from the remaining development compounds",  # Document early-stopping membership.
        "training_only_operations": ["target normalization", "atom masking pretraining", "gradient optimization"],  # State leakage-sensitive operations.
        "total_structures": len(source_rows),  # Record total curated crystallographic observations.
        "total_compounds": len(compounds),  # Record total independent compound groups.
        "total_scaffolds": len({str(item["scaffold"]) for item in compounds}),  # Record total indivisible scaffold families.
        "fold_records": fold_records,  # Embed checksums and partition counts for every outer run.
    }  # Finish the metadata object.
    metadata_path = output_dir / "cv_metadata.json"  # Define the canonical CV metadata file.
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")  # Persist readable deterministic JSON.
    return metadata  # Return metadata for tests, notebooks or command-line reporting.


def main() -> None:  # Define the command-line entry point for fold generation.
    """Parse paths and generate the frozen five-fold scaffold CV artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)  # Create a CLI whose help text states the scientific purpose.
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)  # Accept the curated experiment root.
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)  # Accept an alternative audited source manifest.
    parser.add_argument("--output_dir", type=Path, default=None)  # Allow explicit placement of frozen CV artifacts.
    parser.add_argument("--folds", type=int, default=5)  # Default to the requested five outer folds.
    parser.add_argument("--seed", type=int, default=123)  # Default to the established project seed.
    args = parser.parse_args()  # Parse and validate command-line values.
    data_root = args.data_root.resolve()  # Convert the experiment root to an absolute path.
    source = args.source.resolve()  # Convert the source manifest to an absolute path.
    output_dir = (args.output_dir or data_root / "cv").resolve()  # Default outputs to the experiment's CV directory.
    metadata = generate(data_root, source, output_dir, args.folds, args.seed)  # Generate, validate and checksum all fold artifacts.
    print(json.dumps(metadata, indent=2, sort_keys=True))  # Report exact counts and artifact paths to the user.


if __name__ == "__main__":  # Run the CLI only when this module is executed directly.
    main()  # Generate the requested frozen scaffold cross-validation manifests.
