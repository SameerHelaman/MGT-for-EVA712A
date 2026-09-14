# =============================================================================
# MODULE: utils/prepare_openbind_original_entrypoints.py
# PURPOSE: Creates non-overlapping roots consumed by original training.py and testing.py.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Original-script train/validation and test directory roots.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Prepare leak-free OpenBind roots for the original training.py/testing.py."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import argparse  # Load a standard-library, scientific, or local project dependency.
import csv  # Load a standard-library, scientific, or local project dependency.
import shutil  # Load a standard-library, scientific, or local project dependency.
from pathlib import Path  # Import selected classes or functions from the named dependency.


# FUNCTION: write_partition — see its docstring and inline comments.
def write_partition(  # Define this callable; its indented block implements the documented operation.
    source_root: Path,  # Perform this step of the surrounding calculation or control-flow block.
    destination_root: Path,  # Perform this step of the surrounding calculation or control-flow block.
    rows: list[dict[str, str]],  # Perform this step of the surrounding calculation or control-flow block.
) -> None:  # Continue or close the surrounding multiline expression or collection.
    """Write the original StructureDataset layout without duplicating graphs."""
    destination_root.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.

    atom_init_destination = destination_root / "atom_init.json"
    if not atom_init_destination.exists():  # Evaluate this condition before executing the associated branch.
        shutil.copy2(source_root / "atom_init.json", atom_init_destination)

    processed_destination = destination_root / "processed"
    if processed_destination.is_symlink():  # Evaluate this condition before executing the associated branch.
        if processed_destination.resolve() != (source_root / "processed").resolve():
            raise RuntimeError(f"Unexpected processed symlink: {processed_destination}")  # Reject invalid input or state with an explicit exception.
    elif processed_destination.exists():  # Test this additional condition when earlier branches were not selected.
        raise RuntimeError(  # Reject invalid input or state with an explicit exception.
            f"{processed_destination} exists but is not the expected symlink"  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
    else:  # Handle the remaining case not covered by earlier conditions.
        processed_destination.symlink_to(  # Perform this step of the surrounding calculation or control-flow block.
            (source_root / "processed").resolve(), target_is_directory=True
        )  # Continue or close the surrounding multiline expression or collection.

    with (destination_root / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)  # Bind this name to an intermediate value, configuration setting, or result.
        for row in rows:  # Iterate over the stated records, layers, batches, or graph elements.
            writer.writerow([row["complex_name"], row["experimental_pKD"]])


# FUNCTION: main — see its docstring and inline comments.
def main() -> None:  # Define this callable; its indented block implements the documented operation.
    """Create separate development and held-out test roots for original scripts."""
    parser = argparse.ArgumentParser()  # Bind this name to an intermediate value, configuration setting, or result.
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--source_root",
        type=Path,  # Bind this name to an intermediate value, configuration setting, or result.
        default=Path("OpenBind_EV-A71_2A/experiment_a_ligand_mgt"),
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--output_root",
        type=Path,  # Bind this name to an intermediate value, configuration setting, or result.
        default=None,  # Bind this name to an intermediate value, configuration setting, or result.
        help="Destination root; defaults to a split- and seed-specific path.",
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--split_method",
        choices=["random", "scaffold"],
        default="random",
        help="Frozen manifest used to create development and held-out roots.",
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.
    if args.output_root is None:  # Evaluate this condition before executing the associated branch.
        args.output_root = Path(  # Bind this name to an intermediate value, configuration setting, or result.
            "OpenBind_EV-A71_2A"
        ) / f"original_entrypoints_{args.split_method}_seed_{args.seed}"  # Continue or close the surrounding multiline expression or collection.

    manifest = (  # Prepare dataset membership or batched data access for the experiment.
        args.source_root  # Perform this step of the surrounding calculation or control-flow block.
        / "splits"
        / f"{args.split_method}_seed_{args.seed}.csv"  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))  # Bind this name to an intermediate value, configuration setting, or result.

    train_validation = [  # Bind this name to an intermediate value, configuration setting, or result.
        row for row in rows if row["split"] in {"train", "validation"}
    ]  # Continue or close the surrounding multiline expression or collection.
    test = [row for row in rows if row["split"] == "test"]

    if len(train_validation) != 528 or len(test) != 93:  # Evaluate this condition before executing the associated branch.
        raise RuntimeError(  # Reject invalid input or state with an explicit exception.
            f"Expected 528 train+validation and 93 test structures; "  # Perform this step of the surrounding calculation or control-flow block.
            f"found {len(train_validation)} and {len(test)}"  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.

    write_partition(  # Perform this step of the surrounding calculation or control-flow block.
        args.source_root, args.output_root / "train_validation", train_validation
    )  # Continue or close the surrounding multiline expression or collection.
    write_partition(args.source_root, args.output_root / "test", test)

    print(f"training.py root: {args.output_root / 'train_validation'}")
    print(f"testing.py root:  {args.output_root / 'test'}")
    print(f"train+validation structures: {len(train_validation)}")  # Report progress, predictions, or metrics to the selected output/logging backend.
    print(f"test structures: {len(test)}")  # Report progress, predictions, or metrics to the selected output/logging backend.


if __name__ == "__main__":
    main()  # Perform this step of the surrounding calculation or control-flow block.
