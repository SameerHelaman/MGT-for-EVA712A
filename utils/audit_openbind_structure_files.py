"""Create an immutable file manifest for the OpenBind EV-A71 2A release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase


FILE_SUFFIXES = {
    "complex_ref_pdb": "_complex_ref.pdb",
    "prepared_protein_pdb": "_prepared.pdb",
    "ligand_ref_sdf": "_ligand_ref.sdf",
    "ligand_prepared_sdf": "_ligand_prepared.sdf",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdb_counts(path: Path) -> dict[str, int]:
    """Count coordinate records in one PDB file."""
    counts = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record in {"ATOM", "HETATM"}:
                counts[record] += 1
                counts["coordinate_records"] += 1
                if line[17:20].strip() == "LIG":
                    counts["ligand_records"] += 1
    return dict(counts)


def sdf_audit(path: Path) -> dict[str, object]:
    """Parse one SDF and report molecule and atom information."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecules = [mol for mol in supplier if mol is not None]
    if len(molecules) != 1:
        return {
            "parse_ok": False,
            "molecule_count": len(molecules),
            "atom_count": "",
            "heavy_atom_count": "",
            "formal_charge": "",
            "elements": "",
        }
    molecule = molecules[0]
    elements = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    return {
        "parse_ok": True,
        "molecule_count": 1,
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "elements": ";".join(f"{key}:{elements[key]}" for key in sorted(elements)),
    }


def locate_group_directory(structures_root: Path, compound_group: str) -> Path:
    """Resolve the unique directory whose name matches one compound group."""
    direct = structures_root / compound_group
    if direct.is_dir():
        return direct
    matches = sorted(structures_root.glob(compound_group))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one directory for compound_group={compound_group}, "
            f"found {len(matches)}"
        )
    return matches[0]


def build_manifest(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    """Map every metadata row to its four expected structure files."""
    metadata_path = dataset_root / "EV-A71_2A_metadata.csv"
    structures_root = dataset_root / "structures"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "structure_file_manifest.csv"
    report_path = output_dir / "structure_file_manifest_summary.json"

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))

    manifest_rows = []
    seen_complexes = Counter(row["complex_name"] for row in metadata_rows)
    expected_paths = set()
    missing_paths = []
    parse_failures = []

    for metadata_index, metadata in enumerate(metadata_rows, start=1):
        complex_name = metadata["complex_name"]
        group_dir = locate_group_directory(
            structures_root, metadata["compound_group"]
        )
        complex_dir = group_dir / complex_name
        row = {
            "metadata_row": metadata_index,
            **metadata,
            "complex_directory": str(complex_dir.relative_to(dataset_root)),
        }

        for file_role, suffix in FILE_SUFFIXES.items():
            path = complex_dir / f"{complex_name}{suffix}"
            expected_paths.add(path.resolve())
            exists = path.is_file()
            row[f"{file_role}_path"] = (
                str(path.relative_to(dataset_root)) if exists else ""
            )
            row[f"{file_role}_exists"] = exists
            row[f"{file_role}_bytes"] = path.stat().st_size if exists else ""
            row[f"{file_role}_sha256"] = sha256_file(path) if exists else ""
            if not exists:
                missing_paths.append(str(path))

        for role in ("complex_ref_pdb", "prepared_protein_pdb"):
            path_text = row[f"{role}_path"]
            counts = (
                pdb_counts(dataset_root / path_text) if path_text else {}
            )
            row[f"{role}_coordinate_records"] = counts.get(
                "coordinate_records", ""
            )
            row[f"{role}_atom_records"] = counts.get("ATOM", "")
            row[f"{role}_hetatm_records"] = counts.get("HETATM", "")
            row[f"{role}_ligand_records"] = counts.get("ligand_records", "")

        for role in ("ligand_ref_sdf", "ligand_prepared_sdf"):
            path_text = row[f"{role}_path"]
            audit = (
                sdf_audit(dataset_root / path_text)
                if path_text
                else {"parse_ok": False}
            )
            for key in (
                "parse_ok",
                "molecule_count",
                "atom_count",
                "heavy_atom_count",
                "formal_charge",
                "elements",
            ):
                row[f"{role}_{key}"] = audit.get(key, "")
            if not audit.get("parse_ok", False):
                parse_failures.append(f"{complex_name}:{role}")

        manifest_rows.append(row)

    actual_structure_files = {
        path.resolve()
        for path in structures_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdb", ".sdf"}
    }
    unexpected_paths = sorted(str(path) for path in actual_structure_files - expected_paths)
    duplicate_complex_names = sorted(
        name for name, count in seen_complexes.items() if count != 1
    )

    fieldnames = list(manifest_rows[0].keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "dataset": "OpenBind EV-A71 2A",
        "manifest_version": "1.0",
        "metadata_path": str(metadata_path.relative_to(dataset_root)),
        "metadata_sha256": sha256_file(metadata_path),
        "rdkit_version": rdBase.rdkitVersion,
        "metadata_rows": len(metadata_rows),
        "unique_complex_names": len(seen_complexes),
        "duplicate_or_missing_complex_names": duplicate_complex_names,
        "expected_files_per_complex": len(FILE_SUFFIXES),
        "expected_structure_files": len(expected_paths),
        "actual_pdb_sdf_files": len(actual_structure_files),
        "missing_expected_files": missing_paths,
        "unexpected_pdb_sdf_files": unexpected_paths,
        "sdf_parse_failures": parse_failures,
        "all_metadata_rows_mapped": (
            len(metadata_rows) == 925
            and len(seen_complexes) == len(metadata_rows)
            and not duplicate_complex_names
            and not missing_paths
            and not unexpected_paths
            and not parse_failures
        ),
        "manifest_path": str(manifest_path.relative_to(dataset_root)),
        "manifest_sha256": sha256_file(manifest_path),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    """Parse paths, create the manifest, and print its audit summary."""
    project_root = Path(__file__).resolve().parents[1]
    default_root = (
        project_root / "OpenBind_EV-A71_2A" / "OpenBind_EV-A71_2A"
    )
    parser = argparse.ArgumentParser(
        description="Audit and map all OpenBind EV-A71 2A structure files."
    )
    parser.add_argument("--dataset_root", type=Path, default=default_root)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else dataset_root / "reports"
    )
    summary = build_manifest(dataset_root, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
