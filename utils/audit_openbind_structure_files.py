# =============================================================================
# MODULE: utils/audit_openbind_structure_files.py
# PURPOSE: Audits OpenBind metadata-to-file mappings, molecular identities and checksums.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Structure-file manifest CSV and audit summary JSON.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Create an immutable file manifest for the OpenBind EV-A71 2A release."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import argparse  # Load a standard-library, scientific, or local project dependency.
import csv  # Load a standard-library, scientific, or local project dependency.
import hashlib  # Load a standard-library, scientific, or local project dependency.
import json  # Load a standard-library, scientific, or local project dependency.
from collections import Counter  # Import selected classes or functions from the named dependency.
from pathlib import Path  # Import selected classes or functions from the named dependency.

from rdkit import Chem, rdBase  # Import selected classes or functions from the named dependency.


FILE_SUFFIXES = {  # Bind this name to an intermediate value, configuration setting, or result.
    "complex_ref_pdb": "_complex_ref.pdb",
    "prepared_protein_pdb": "_prepared.pdb",
    "ligand_ref_sdf": "_ligand_ref.sdf",
    "ligand_prepared_sdf": "_ligand_prepared.sdf",
}  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: sha256_file — see its docstring and inline comments.
def sha256_file(path: Path) -> str:  # Define this callable; its indented block implements the documented operation.
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()  # Bind this name to an intermediate value, configuration setting, or result.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)  # Perform this step of the surrounding calculation or control-flow block.
    return digest.hexdigest()  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: pdb_counts — see its docstring and inline comments.
def pdb_counts(path: Path) -> dict[str, int]:  # Define this callable; its indented block implements the documented operation.
    """Count coordinate records in one PDB file."""
    counts = Counter()  # Bind this name to an intermediate value, configuration setting, or result.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:  # Iterate over the stated records, layers, batches, or graph elements.
            record = line[:6].strip()  # Bind this name to an intermediate value, configuration setting, or result.
            if record in {"ATOM", "HETATM"}:
                counts[record] += 1  # Bind this name to an intermediate value, configuration setting, or result.
                counts["coordinate_records"] += 1
                if line[17:20].strip() == "LIG":
                    counts["ligand_records"] += 1
    return dict(counts)  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: sdf_audit — see its docstring and inline comments.
def sdf_audit(path: Path) -> dict[str, object]:  # Define this callable; its indented block implements the documented operation.
    """Parse one SDF and report molecule and atom information."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)  # Bind this name to an intermediate value, configuration setting, or result.
    molecules = [mol for mol in supplier if mol is not None]  # Bind this name to an intermediate value, configuration setting, or result.
    if len(molecules) != 1:  # Evaluate this condition before executing the associated branch.
        return {  # Return this computed tensor, metric, object, or collection to the caller.
            "parse_ok": False,
            "molecule_count": len(molecules),
            "atom_count": "",
            "heavy_atom_count": "",
            "formal_charge": "",
            "elements": "",
        }  # Continue or close the surrounding multiline expression or collection.
    molecule = molecules[0]  # Bind this name to an intermediate value, configuration setting, or result.
    elements = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())  # Bind this name to an intermediate value, configuration setting, or result.
    return {  # Return this computed tensor, metric, object, or collection to the caller.
        "parse_ok": True,
        "molecule_count": 1,
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "elements": ";".join(f"{key}:{elements[key]}" for key in sorted(elements)),
    }  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: locate_group_directory — see its docstring and inline comments.
def locate_group_directory(structures_root: Path, compound_group: str) -> Path:  # Define this callable; its indented block implements the documented operation.
    """Resolve the unique directory whose name matches one compound group."""
    direct = structures_root / compound_group  # Bind this name to an intermediate value, configuration setting, or result.
    if direct.is_dir():  # Evaluate this condition before executing the associated branch.
        return direct  # Return this computed tensor, metric, object, or collection to the caller.
    matches = sorted(structures_root.glob(compound_group))  # Bind this name to an intermediate value, configuration setting, or result.
    if len(matches) != 1:  # Evaluate this condition before executing the associated branch.
        raise ValueError(  # Reject invalid input or state with an explicit exception.
            f"Expected one directory for compound_group={compound_group}, "  # Bind this name to an intermediate value, configuration setting, or result.
            f"found {len(matches)}"  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
    return matches[0]  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: build_manifest — see its docstring and inline comments.
def build_manifest(dataset_root: Path, output_dir: Path) -> dict[str, object]:  # Define this callable; its indented block implements the documented operation.
    """Map every metadata row to its four expected structure files."""
    metadata_path = dataset_root / "EV-A71_2A_metadata.csv"
    structures_root = dataset_root / "structures"
    output_dir.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    manifest_path = output_dir / "structure_file_manifest.csv"
    report_path = output_dir / "structure_file_manifest_summary.json"

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))  # Bind this name to an intermediate value, configuration setting, or result.

    manifest_rows = []  # Prepare dataset membership or batched data access for the experiment.
    seen_complexes = Counter(row["complex_name"] for row in metadata_rows)
    expected_paths = set()  # Bind this name to an intermediate value, configuration setting, or result.
    missing_paths = []  # Bind this name to an intermediate value, configuration setting, or result.
    parse_failures = []  # Bind this name to an intermediate value, configuration setting, or result.

    for metadata_index, metadata in enumerate(metadata_rows, start=1):  # Iterate over the stated records, layers, batches, or graph elements.
        complex_name = metadata["complex_name"]
        group_dir = locate_group_directory(  # Bind this name to an intermediate value, configuration setting, or result.
            structures_root, metadata["compound_group"]
        )  # Continue or close the surrounding multiline expression or collection.
        complex_dir = group_dir / complex_name  # Bind this name to an intermediate value, configuration setting, or result.
        row = {  # Bind this name to an intermediate value, configuration setting, or result.
            "metadata_row": metadata_index,
            **metadata,  # Perform this step of the surrounding calculation or control-flow block.
            "complex_directory": str(complex_dir.relative_to(dataset_root)),
        }  # Continue or close the surrounding multiline expression or collection.

        for file_role, suffix in FILE_SUFFIXES.items():  # Iterate over the stated records, layers, batches, or graph elements.
            path = complex_dir / f"{complex_name}{suffix}"  # Bind this name to an intermediate value, configuration setting, or result.
            expected_paths.add(path.resolve())  # Perform this step of the surrounding calculation or control-flow block.
            exists = path.is_file()  # Bind this name to an intermediate value, configuration setting, or result.
            row[f"{file_role}_path"] = (  # Bind this name to an intermediate value, configuration setting, or result.
                str(path.relative_to(dataset_root)) if exists else ""
            )  # Continue or close the surrounding multiline expression or collection.
            row[f"{file_role}_exists"] = exists  # Bind this name to an intermediate value, configuration setting, or result.
            row[f"{file_role}_bytes"] = path.stat().st_size if exists else ""
            row[f"{file_role}_sha256"] = sha256_file(path) if exists else ""
            if not exists:  # Evaluate this condition before executing the associated branch.
                missing_paths.append(str(path))  # Perform this step of the surrounding calculation or control-flow block.

        for role in ("complex_ref_pdb", "prepared_protein_pdb"):
            path_text = row[f"{role}_path"]  # Bind this name to an intermediate value, configuration setting, or result.
            counts = (  # Bind this name to an intermediate value, configuration setting, or result.
                pdb_counts(dataset_root / path_text) if path_text else {}  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
            row[f"{role}_coordinate_records"] = counts.get(  # Bind this name to an intermediate value, configuration setting, or result.
                "coordinate_records", ""
            )  # Continue or close the surrounding multiline expression or collection.
            row[f"{role}_atom_records"] = counts.get("ATOM", "")
            row[f"{role}_hetatm_records"] = counts.get("HETATM", "")
            row[f"{role}_ligand_records"] = counts.get("ligand_records", "")

        for role in ("ligand_ref_sdf", "ligand_prepared_sdf"):
            path_text = row[f"{role}_path"]  # Bind this name to an intermediate value, configuration setting, or result.
            audit = (  # Bind this name to an intermediate value, configuration setting, or result.
                sdf_audit(dataset_root / path_text)  # Perform this step of the surrounding calculation or control-flow block.
                if path_text  # Evaluate this condition before executing the associated branch.
                else {"parse_ok": False}
            )  # Continue or close the surrounding multiline expression or collection.
            for key in (  # Iterate over the stated records, layers, batches, or graph elements.
                "parse_ok",
                "molecule_count",
                "atom_count",
                "heavy_atom_count",
                "formal_charge",
                "elements",
            ):  # Continue or close the surrounding multiline expression or collection.
                row[f"{role}_{key}"] = audit.get(key, "")
            if not audit.get("parse_ok", False):
                parse_failures.append(f"{complex_name}:{role}")  # Perform this step of the surrounding calculation or control-flow block.

        manifest_rows.append(row)  # Perform this step of the surrounding calculation or control-flow block.

    actual_structure_files = {  # Bind this name to an intermediate value, configuration setting, or result.
        path.resolve()  # Perform this step of the surrounding calculation or control-flow block.
        for path in structures_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdb", ".sdf"}
    }  # Continue or close the surrounding multiline expression or collection.
    unexpected_paths = sorted(str(path) for path in actual_structure_files - expected_paths)  # Bind this name to an intermediate value, configuration setting, or result.
    duplicate_complex_names = sorted(  # Bind this name to an intermediate value, configuration setting, or result.
        name for name, count in seen_complexes.items() if count != 1  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.

    fieldnames = list(manifest_rows[0].keys())  # Prepare dataset membership or batched data access for the experiment.
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)  # Bind this name to an intermediate value, configuration setting, or result.
        writer.writeheader()  # Perform this step of the surrounding calculation or control-flow block.
        writer.writerows(manifest_rows)  # Perform this step of the surrounding calculation or control-flow block.

    summary = {  # Bind this name to an intermediate value, configuration setting, or result.
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
            len(metadata_rows) == 925  # Bind this name to an intermediate value, configuration setting, or result.
            and len(seen_complexes) == len(metadata_rows)  # Bind this name to an intermediate value, configuration setting, or result.
            and not duplicate_complex_names  # Perform this step of the surrounding calculation or control-flow block.
            and not missing_paths  # Perform this step of the surrounding calculation or control-flow block.
            and not unexpected_paths  # Perform this step of the surrounding calculation or control-flow block.
            and not parse_failures  # Perform this step of the surrounding calculation or control-flow block.
        ),  # Continue or close the surrounding multiline expression or collection.
        "manifest_path": str(manifest_path.relative_to(dataset_root)),
        "manifest_sha256": sha256_file(manifest_path),
    }  # Continue or close the surrounding multiline expression or collection.
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)  # Bind this name to an intermediate value, configuration setting, or result.
        handle.write("\n")
    return summary  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: main — see its docstring and inline comments.
def main() -> None:  # Define this callable; its indented block implements the documented operation.
    """Parse paths, create the manifest, and print its audit summary."""
    project_root = Path(__file__).resolve().parents[1]  # Bind this name to an intermediate value, configuration setting, or result.
    default_root = (  # Bind this name to an intermediate value, configuration setting, or result.
        project_root / "OpenBind_EV-A71_2A" / "OpenBind_EV-A71_2A"
    )  # Continue or close the surrounding multiline expression or collection.
    parser = argparse.ArgumentParser(  # Bind this name to an intermediate value, configuration setting, or result.
        description="Audit and map all OpenBind EV-A71 2A structure files."
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument("--dataset_root", type=Path, default=default_root)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.
    dataset_root = args.dataset_root.resolve()  # Prepare dataset membership or batched data access for the experiment.
    output_dir = (  # Bind this name to an intermediate value, configuration setting, or result.
        args.output_dir.resolve()  # Perform this step of the surrounding calculation or control-flow block.
        if args.output_dir  # Evaluate this condition before executing the associated branch.
        else dataset_root / "reports"
    )  # Continue or close the surrounding multiline expression or collection.
    summary = build_manifest(dataset_root, output_dir)  # Prepare dataset membership or batched data access for the experiment.
    print(json.dumps(summary, indent=2, sort_keys=True))  # Report progress, predictions, or metrics to the selected output/logging backend.


if __name__ == "__main__":
    main()  # Perform this step of the surrounding calculation or control-flow block.
