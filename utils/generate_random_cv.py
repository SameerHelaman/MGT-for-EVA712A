# =============================================================================
# MODULE: utils/generate_random_cv.py
# PURPOSE: Freeze five compound-grouped random outer folds and inner validation sets.
# LIBRARIES: Standard-library CSV/JSON/path modules persist reproducible manifests;
#            shared scaffold-CV helpers validate compound identity and calculate checksums.
# OUTPUT: One compound assignment CSV, five trainer-compatible split manifests and
#         one checksummed metadata JSON under experiment_a_ligand_mgt/cv_random/.
# CALCULATIONS: Compounds sorted by pKD are allocated in shuffled blocks across
#               five folds, balancing sample counts and target distributions.
# =============================================================================
"""Generate deterministic compound-grouped random five-fold cross-validation manifests."""

from __future__ import annotations  # Postpone type-annotation evaluation for compatibility.

import argparse  # Parse fold count, seed and artifact path options.
import json  # Persist machine-readable fold metadata and audit results.
import random  # Shuffle compounds and fold order reproducibly from fixed seeds.
from pathlib import Path  # Resolve dataset and output paths without shell-specific syntax.

try:  # Prefer package-style imports when called by the generic repository runner.
    from utils.generate_scaffold_cv import (  # Reuse audited identity, CSV and checksum helpers.
        DEFAULT_DATA_ROOT,  # Share the curated OpenBind experiment root.
        compound_table,  # Collapse repeated structures and validate compound identities/labels.
        read_csv,  # Read the existing frozen source manifest.
        sha256_file,  # Fingerprint every generated artifact.
        write_csv,  # Write deterministic trainer-compatible CSV files.
    )  # Finish the package-style shared-helper import list.
except ModuleNotFoundError:  # Support direct execution as `python utils/generate_random_cv.py`.
    from generate_scaffold_cv import (  # Import the sibling module when `utils` is the script search root.
        DEFAULT_DATA_ROOT,  # Share the curated OpenBind experiment root.
        compound_table,  # Collapse repeated structures and validate compound identities/labels.
        read_csv,  # Read the existing frozen source manifest.
        sha256_file,  # Fingerprint every generated artifact.
        write_csv,  # Write deterministic trainer-compatible CSV files.
    )  # Finish the direct-script shared-helper import list.

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Locate the MGT repository root from this utility.
DEFAULT_SOURCE = DEFAULT_DATA_ROOT / "splits" / "random_seed_123.csv"  # Use the existing audited random manifest as source membership.


def assign_compounds(compounds: list[dict[str, object]], folds: int, seed: int) -> dict[str, int]:  # Allocate compound groups without splitting repeats.
    """Assign compounds to count- and pKD-balanced folds using stratified shuffled blocks."""
    if folds < 2:  # Cross-validation requires at least two independent outer partitions.
        raise ValueError("folds must be at least two")  # Reject invalid requests before artifact creation.
    if len(compounds) < folds:  # Every fold needs at least one compound.
        raise ValueError("Fewer compounds than requested folds")  # Prevent empty test partitions.
    ordered = sorted(compounds, key=lambda item: (float(item["experimental_pKD"]), str(item["official_compound_group_id"])))  # Order targets for local stratification.
    rng = random.Random(seed)  # Create an isolated deterministic random-number generator.
    assignment: dict[str, int] = {}  # Map each official compound group to exactly one fold.
    for block_start in range(0, len(ordered), folds):  # Process adjacent pKD-ranked compounds in blocks of five.
        block = ordered[block_start : block_start + folds]  # Select compounds with locally similar affinity values.
        rng.shuffle(block)  # Randomize which similarly labelled compound receives which fold.
        fold_order = list(range(folds))  # Create one candidate occurrence of every fold number.
        rng.shuffle(fold_order)  # Avoid systematic alignment of fold number with pKD ordering.
        for compound, fold in zip(block, fold_order):  # Allocate at most one block member to each fold.
            compound_id = str(compound["official_compound_group_id"])  # Normalize the grouping identifier to text.
            assignment[compound_id] = fold  # Freeze this compound and all of its structures in one fold.
    return assignment  # Return complete compound-to-fold membership.


def validate_random_manifest(rows: list[dict[str, object]]) -> dict[str, object]:  # Audit compound separation while allowing scaffold overlap.
    """Assert non-empty partitions and zero compound leakage; report scaffold overlap."""
    partitions = ("train", "validation", "test")  # Define the only valid supervised data roles.
    compound_sets = {part: {str(row["official_compound_group_id"]) for row in rows if row["split"] == part} for part in partitions}  # Collect compound identities.
    scaffold_sets = {part: {str(row["scaffold"]) for row in rows if row["split"] == part} for part in partitions}  # Collect scaffolds for descriptive overlap reporting.
    if any(not compound_sets[part] for part in partitions):  # Require usable training, checkpoint-selection and test sets.
        raise AssertionError("Train, validation and test partitions must all be non-empty")  # Reject incomplete random folds.
    overlap_report: dict[str, int] = {}  # Record expected scaffold sharing without treating it as leakage in random CV.
    for left_index, left in enumerate(partitions):  # Compare each partition with those following it.
        for right in partitions[left_index + 1 :]:  # Avoid duplicate symmetric comparisons.
            if compound_sets[left] & compound_sets[right]:  # Detect repeated compound identity across partitions.
                raise AssertionError(f"Compound leakage between {left} and {right}")  # Reject direct target/structure leakage.
            overlap_report[f"{left}_{right}"] = len(scaffold_sets[left] & scaffold_sets[right])  # Quantify allowed scaffold overlap.
    return {  # Return partition sizes and explicit overlap semantics.
        "partitions": {  # Nest counts under their data-role names.
            part: {  # Create counts for one partition.
                "structures": sum(row["split"] == part for row in rows),  # Count crystallographic records.
                "compounds": len(compound_sets[part]),  # Count independent compound groups.
                "scaffolds": len(scaffold_sets[part]),  # Count represented achiral frameworks.
            }  # Finish this partition count record.
            for part in partitions  # Repeat for training, validation and testing.
        },  # Finish all partition counts.
        "scaffold_overlap_counts": overlap_report,  # Preserve expected analogue sharing for interpretation.
    }  # Finish the validation report.


def generate(data_root: Path, source: Path, output_dir: Path, folds: int, seed: int) -> dict[str, object]:  # Run the complete random-CV freeze operation.
    """Generate stratified compound folds, inner validation sets and checksummed metadata."""
    source_rows = read_csv(source)  # Load all curated crystallographic structure records.
    compounds = compound_table(source_rows)  # Collapse repeats and recompute one achiral scaffold per compound.
    outer_assignment = assign_compounds(compounds, folds, seed)  # Allocate every compound to one outer test fold.
    scaffold_by_compound = {str(item["official_compound_group_id"]): str(item["scaffold"]) for item in compounds}  # Preserve reporting frameworks.
    assignment_rows = [  # Construct the concise compound-level outer-fold table.
        {**item, "outer_fold": outer_assignment[str(item["official_compound_group_id"])]}  # Attach each compound's single held-out fold.
        for item in sorted(compounds, key=lambda row: str(row["official_compound_group_id"]))  # Preserve deterministic identifier ordering.
    ]  # Finish the assignment rows.
    assignment_path = output_dir / f"random_grouped_{folds}fold_seed_{seed}.csv"  # Define the master random assignment artifact.
    write_csv(assignment_path, assignment_rows, list(assignment_rows[0]))  # Freeze its exact rows and schema.
    fold_records: list[dict[str, object]] = []  # Collect checksums/counts for every outer experiment.
    test_occurrences: dict[str, int] = {str(item["official_compound_group_id"]): 0 for item in compounds}  # Count outer-test use per compound.
    for outer_fold in range(folds):  # Build one independent test experiment for each fold.
        development = [item for item in compounds if outer_assignment[str(item["official_compound_group_id"])] != outer_fold]  # Exclude locked test compounds.
        inner_assignment = assign_compounds(development, folds, seed + 1000 + outer_fold)  # Stratify remaining compounds into inner groups.
        validation_inner_fold = outer_fold % folds  # Select one inner group deterministically for early stopping.
        split_by_compound: dict[str, str] = {}  # Map every compound to train, validation or test.
        for item in compounds:  # Assign all compounds without consulting any model result.
            compound_id = str(item["official_compound_group_id"])  # Read the official grouping key.
            if outer_assignment[compound_id] == outer_fold:  # Identify this run's locked outer test membership.
                split_by_compound[compound_id] = "test"  # Reserve the compound for final evaluation only.
                test_occurrences[compound_id] += 1  # Record its single out-of-fold evaluation.
            elif inner_assignment[compound_id] == validation_inner_fold:  # Identify the selected inner validation group.
                split_by_compound[compound_id] = "validation"  # Use the compound only for checkpoint selection.
            else:  # Remaining development compounds are eligible for fitting.
                split_by_compound[compound_id] = "train"  # Use them for normalization, masking and supervised gradients.
        fold_rows: list[dict[str, object]] = []  # Prepare the structure-level schema consumed by all trainers.
        for row in sorted(source_rows, key=lambda item: item["complex_name"]):  # Retain every structure in deterministic order.
            compound_id = row["official_compound_group_id"]  # Read the structure's compound group.
            fold_rows.append({  # Create one trainer-compatible membership record.
                "complex_name": row["complex_name"],  # Preserve dataset lookup identity.
                "official_compound_group_id": compound_id,  # Preserve repeated-structure aggregation identity.
                "canonical_smiles": row["canonical_smiles"],  # Preserve structure-specific audited molecular representation.
                "scaffold": scaffold_by_compound[compound_id],  # Store achiral scaffold for overlap reporting only.
                "experimental_pKD": row["experimental_pKD"],  # Preserve the frozen regression target.
                "split": split_by_compound[compound_id],  # Apply compound-level random membership to every repeat.
                "split_method": "random",  # Retain compatibility with existing trainer checks.
                "seed": seed,  # Retain the shared paired experiment seed.
                "outer_fold": outer_fold,  # Record which fold is untouched test data.
                "inner_validation_fold": validation_inner_fold,  # Record checkpoint-selection group.
            })  # Finish this structure record.
        audit = validate_random_manifest(fold_rows)  # Assert zero compound leakage and quantify scaffold sharing.
        manifest_path = output_dir / f"random_cv{folds}_fold_{outer_fold}_seed_{seed}.csv"  # Define this fold's exact split file.
        write_csv(manifest_path, fold_rows, list(fold_rows[0]))  # Freeze structure membership for all seven models.
        fold_records.append({  # Add this fold to the metadata report.
            "outer_fold": outer_fold,  # Identify locked test membership.
            "inner_validation_fold": validation_inner_fold,  # Identify inner checkpoint-selection membership.
            "manifest": str(manifest_path.resolve()),  # Link the absolute trainer input path.
            "sha256": sha256_file(manifest_path),  # Fingerprint exact row content/order.
            "counts": audit["partitions"],  # Store structure, compound and scaffold sizes.
            "compound_leakage_zero": True,  # Record the successfully asserted grouping invariant.
            "scaffold_overlap_allowed": True,  # Distinguish random interpolation from scaffold extrapolation.
            "scaffold_overlap_counts": audit["scaffold_overlap_counts"],  # Quantify related-framework sharing.
        })  # Finish this fold metadata record.
    if set(test_occurrences.values()) != {1}:  # Every compound must be held out once and only once.
        raise AssertionError("Each compound must occur in exactly one outer test fold")  # Reject incomplete or duplicated outer coverage.
    metadata = {  # Build the complete reproducibility record.
        "method": "five-fold compound-grouped stratified random cross-validation",  # Name the scientific design.
        "folds": folds,  # Record outer fold count.
        "seed": seed,  # Record deterministic allocation seed.
        "source_manifest": str(source.resolve()),  # Link the existing audited source membership.
        "source_sha256": sha256_file(source),  # Freeze exact source bytes.
        "assignment_manifest": str(assignment_path.resolve()),  # Link compound outer-fold assignments.
        "assignment_sha256": sha256_file(assignment_path),  # Freeze assignment bytes.
        "grouping_rule": "official_compound_group_id; all repeated crystal structures remain together",  # Document leakage prevention.
        "stratification_rule": "pKD-sorted blocks with seeded within-block and fold-order shuffling",  # Document label balancing.
        "scaffold_rule": "scaffold overlap is allowed and reported, not used for assignment",  # Clarify interpolation interpretation.
        "inner_validation_rule": "one of five stratified groups from remaining development compounds",  # Document epoch selection data.
        "training_only_operations": ["target normalization", "atom masking pretraining", "gradient optimization"],  # State leakage-sensitive operations.
        "total_structures": len(source_rows),  # Record crystallographic observation count.
        "total_compounds": len(compounds),  # Record independent compound count.
        "fold_records": fold_records,  # Embed all counts, overlaps and checksums.
    }  # Finish metadata.
    metadata_path = output_dir / "cv_metadata.json"  # Define canonical random-CV metadata location.
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")  # Persist readable deterministic JSON.
    return metadata  # Return metadata for the runner, tests or CLI display.


def main() -> None:  # Define the standalone random-fold generation entry point.
    """Parse paths and freeze random compound-grouped CV artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)  # Create a self-documenting CLI.
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)  # Accept the curated experiment root.
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)  # Accept an alternative audited source manifest.
    parser.add_argument("--output_dir", type=Path, default=None)  # Allow explicit artifact placement.
    parser.add_argument("--folds", type=int, default=5)  # Default to requested five-fold evaluation.
    parser.add_argument("--seed", type=int, default=123)  # Default to established project seed.
    args = parser.parse_args()  # Parse command-line values.
    data_root = args.data_root.resolve()  # Normalize the dataset path.
    source = args.source.resolve()  # Normalize the source manifest path.
    output_dir = (args.output_dir or data_root / "cv_random").resolve()  # Default to a random-specific CV directory.
    metadata = generate(data_root, source, output_dir, args.folds, args.seed)  # Generate, validate and checksum folds.
    print(json.dumps(metadata, indent=2, sort_keys=True))  # Display exact counts and artifact paths.


if __name__ == "__main__":  # Run only when invoked as a script.
    main()  # Freeze requested random CV artifacts.
