"""Prepare leak-free OpenBind roots for the original training.py/testing.py."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def write_partition(
    source_root: Path,
    destination_root: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write the original StructureDataset layout without duplicating graphs."""
    destination_root.mkdir(parents=True, exist_ok=True)

    atom_init_destination = destination_root / "atom_init.json"
    if not atom_init_destination.exists():
        shutil.copy2(source_root / "atom_init.json", atom_init_destination)

    processed_destination = destination_root / "processed"
    if processed_destination.is_symlink():
        if processed_destination.resolve() != (source_root / "processed").resolve():
            raise RuntimeError(f"Unexpected processed symlink: {processed_destination}")
    elif processed_destination.exists():
        raise RuntimeError(
            f"{processed_destination} exists but is not the expected symlink"
        )
    else:
        processed_destination.symlink_to(
            (source_root / "processed").resolve(), target_is_directory=True
        )

    with (destination_root / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow([row["complex_name"], row["experimental_pKD"]])


def main() -> None:
    """Create separate development and held-out test roots for original scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("OpenBind_EV-A71_2A/experiment_a_ligand_mgt"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("OpenBind_EV-A71_2A/original_entrypoints_random_seed_123"),
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    manifest = args.source_root / "splits" / f"random_seed_{args.seed}.csv"
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    train_validation = [
        row for row in rows if row["split"] in {"train", "validation"}
    ]
    test = [row for row in rows if row["split"] == "test"]

    if len(train_validation) != 528 or len(test) != 93:
        raise RuntimeError(
            f"Expected 528 train+validation and 93 test structures; "
            f"found {len(train_validation)} and {len(test)}"
        )

    write_partition(
        args.source_root, args.output_root / "train_validation", train_validation
    )
    write_partition(args.source_root, args.output_root / "test", test)

    print(f"training.py root: {args.output_root / 'train_validation'}")
    print(f"testing.py root:  {args.output_root / 'test'}")
    print(f"train+validation structures: {len(train_validation)}")
    print(f"test structures: {len(test)}")


if __name__ == "__main__":
    main()
