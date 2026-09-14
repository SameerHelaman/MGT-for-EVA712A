# =============================================================================
# MODULE: utils/prepare_openbind_ligand_mgt.py
# PURPOSE: Curates OpenBind structures, exports ligand PDBs, groups compounds and freezes random/scaffold splits.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Curated CSVs, ligand PDB files, split manifests and dataset metadata.
# CALCULATIONS: pKD values are taken from curated release metadata; Bemis–Murcko scaffolds define scaffold-disjoint partitions; SHA-256 freezes artifacts.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Prepare quality-controlled crystallographic ligand PDBs for original MGT."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import argparse  # Load a standard-library, scientific, or local project dependency.
import csv  # Load a standard-library, scientific, or local project dependency.
import hashlib  # Load a standard-library, scientific, or local project dependency.
import json  # Load a standard-library, scientific, or local project dependency.
import random  # Load a standard-library, scientific, or local project dependency.
import shutil  # Load a standard-library, scientific, or local project dependency.
from collections import Counter, defaultdict  # Import selected classes or functions from the named dependency.
from pathlib import Path  # Import selected classes or functions from the named dependency.

from rdkit import Chem, rdBase  # Import selected classes or functions from the named dependency.
from rdkit.Chem.Scaffolds import MurckoScaffold  # Import selected classes or functions from the named dependency.


SEED = 123  # Bind this name to an intermediate value, configuration setting, or result.
PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


# FUNCTION: sha256_file — see its docstring and inline comments.
def sha256_file(path: Path) -> str:  # Define this callable; its indented block implements the documented operation.
    """Calculate a file checksum in bounded-memory blocks."""
    digest = hashlib.sha256()  # Bind this name to an intermediate value, configuration setting, or result.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)  # Perform this step of the surrounding calculation or control-flow block.
    return digest.hexdigest()  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: read_csv — see its docstring and inline comments.
def read_csv(path: Path) -> list[dict[str, str]]:  # Define this callable; its indented block implements the documented operation.
    """Read a CSV as dictionaries without changing source values."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: write_csv — see its docstring and inline comments.
def write_csv(  # Define this callable; its indented block implements the documented operation.
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]  # Perform this step of the surrounding calculation or control-flow block.
) -> None:  # Continue or close the surrounding multiline expression or collection.
    """Write a stable CSV with an explicit column order."""
    path.parent.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)  # Bind this name to an intermediate value, configuration setting, or result.
        writer.writeheader()  # Perform this step of the surrounding calculation or control-flow block.
        writer.writerows(rows)  # Perform this step of the surrounding calculation or control-flow block.


# FUNCTION: as_bool — see its docstring and inline comments.
def as_bool(value: str) -> bool:  # Define this callable; its indented block implements the documented operation.
    """Interpret the literal boolean fields supplied by OpenBind."""
    if value not in {"True", "False"}:
        raise ValueError(f"Unexpected boolean value: {value!r}")  # Reject invalid input or state with an explicit exception.
    return value == "True"


# FUNCTION: canonical_smiles — see its docstring and inline comments.
def canonical_smiles(molecule: Chem.Mol) -> str:  # Define this callable; its indented block implements the documented operation.
    """Return an isomeric canonical identity for one ligand."""
    return Chem.MolToSmiles(  # Return this computed tensor, metric, object, or collection to the caller.
        Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: scaffold_smiles — see its docstring and inline comments.
def scaffold_smiles(molecule: Chem.Mol, fallback: str) -> str:  # Define this callable; its indented block implements the documented operation.
    """Return a chiral Bemis-Murcko scaffold or a unique acyclic key."""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(  # Bind this name to an intermediate value, configuration setting, or result.
        mol=Chem.RemoveHs(molecule), includeChirality=True  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    return scaffold if scaffold else f"ACYCLIC::{fallback}"  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: exclusion_reasons — see its docstring and inline comments.
def exclusion_reasons(row: dict[str, str], benchmark_codes: set[str]) -> list[str]:  # Define this callable; its indented block implements the documented operation.
    """Return every prespecified reason that makes a structure ineligible."""
    reasons = []  # Bind this name to an intermediate value, configuration setting, or result.
    if not row["experimental_pKD"]:
        reasons.append("missing_experimental_pKD")
    if row["complex_name"] not in benchmark_codes:
        reasons.append("absent_from_official_affinity_reference")
    if as_bool(row["covalent"]):
        reasons.append("covalent_ligand")
    if as_bool(row["suspected_artefact"]):
        reasons.append("suspected_crystallographic_artefact")
    if not as_bool(row["pb_valid_ref"]):
        reasons.append("reference_pose_fails_posebusters")
    return reasons  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: write_ligand_pdb — see its docstring and inline comments.
def write_ligand_pdb(source_sdf: Path, destination: Path) -> dict[str, object]:  # Define this callable; its indented block implements the documented operation.
    """Convert one reference SDF to PDB while preserving its coordinates."""
    supplier = Chem.SDMolSupplier(  # Bind this name to an intermediate value, configuration setting, or result.
        str(source_sdf), removeHs=False, sanitize=True  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    molecules = [mol for mol in supplier if mol is not None]  # Bind this name to an intermediate value, configuration setting, or result.
    if len(molecules) != 1:  # Evaluate this condition before executing the associated branch.
        raise ValueError(f"{source_sdf} contains {len(molecules)} molecules")  # Reject invalid input or state with an explicit exception.
    molecule = molecules[0]  # Bind this name to an intermediate value, configuration setting, or result.
    if molecule.GetNumConformers() != 1:  # Evaluate this condition before executing the associated branch.
        raise ValueError(f"{source_sdf} does not contain exactly one conformer")  # Reject invalid input or state with an explicit exception.
    destination.parent.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    Chem.MolToPDBFile(molecule, str(destination), flavor=4)  # Bind this name to an intermediate value, configuration setting, or result.
    parsed = Chem.MolFromPDBFile(  # Bind this name to an intermediate value, configuration setting, or result.
        str(destination), sanitize=True, removeHs=False  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    if parsed is None:  # Evaluate this condition before executing the associated branch.
        raise ValueError(f"RDKit failed to parse generated PDB {destination}")  # Reject invalid input or state with an explicit exception.
    if parsed.GetNumAtoms() != molecule.GetNumAtoms():  # Evaluate this condition before executing the associated branch.
        raise ValueError(f"Atom count changed while writing {destination}")  # Reject invalid input or state with an explicit exception.
    source_positions = molecule.GetConformer().GetPositions()  # Bind this name to an intermediate value, configuration setting, or result.
    pdb_positions = parsed.GetConformer().GetPositions()  # Bind this name to an intermediate value, configuration setting, or result.
    maximum_coordinate_difference = float(  # Bind this name to an intermediate value, configuration setting, or result.
        abs(source_positions - pdb_positions).max()  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    if maximum_coordinate_difference > 0.00011:  # Evaluate this condition before executing the associated branch.
        raise ValueError(  # Reject invalid input or state with an explicit exception.
            f"Coordinates changed by {maximum_coordinate_difference} Å "  # Perform this step of the surrounding calculation or control-flow block.
            f"while writing {destination}"  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
    return {  # Return this computed tensor, metric, object, or collection to the caller.
        "canonical_smiles": canonical_smiles(molecule),
        "scaffold": scaffold_smiles(molecule, destination.stem),
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "maximum_coordinate_difference_angstrom": maximum_coordinate_difference,
        "pdb_sha256": sha256_file(destination),
    }  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: allocate_groups — see its docstring and inline comments.
def allocate_groups(  # Define this callable; its indented block implements the documented operation.
    group_to_members: dict[str, list[str]], seed: int  # Perform this step of the surrounding calculation or control-flow block.
) -> dict[str, str]:  # Continue or close the surrounding multiline expression or collection.
    """Allocate indivisible groups toward 70/15/15 structure-level targets."""
    rng = random.Random(seed)  # Bind this name to an intermediate value, configuration setting, or result.
    items = list(group_to_members.items())  # Bind this name to an intermediate value, configuration setting, or result.
    rng.shuffle(items)  # Perform this step of the surrounding calculation or control-flow block.
    items.sort(key=lambda item: len(item[1]), reverse=True)  # Bind this name to an intermediate value, configuration setting, or result.
    total = sum(len(members) for _, members in items)  # Bind this name to an intermediate value, configuration setting, or result.
    targets = {name: total * value for name, value in PROPORTIONS.items()}  # Bind this name to an intermediate value, configuration setting, or result.
    counts = Counter()  # Bind this name to an intermediate value, configuration setting, or result.
    assignments = {}  # Bind this name to an intermediate value, configuration setting, or result.
    for group, members in items:  # Iterate over the stated records, layers, batches, or graph elements.
        split = min(  # Prepare dataset membership or batched data access for the experiment.
            PROPORTIONS,  # Perform this step of the surrounding calculation or control-flow block.
            key=lambda name: (  # Bind this name to an intermediate value, configuration setting, or result.
                counts[name] / targets[name]  # Perform this step of the surrounding calculation or control-flow block.
                if targets[name] > 0  # Evaluate this condition before executing the associated branch.
                else float("inf"),
                counts[name],  # Perform this step of the surrounding calculation or control-flow block.
                name,  # Perform this step of the surrounding calculation or control-flow block.
            ),  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
        assignments[group] = split  # Prepare dataset membership or batched data access for the experiment.
        counts[split] += len(members)  # Prepare dataset membership or batched data access for the experiment.
    return assignments  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: build_split_rows — see its docstring and inline comments.
def build_split_rows(  # Define this callable; its indented block implements the documented operation.
    records: list[dict[str, object]], method: str  # Perform this step of the surrounding calculation or control-flow block.
) -> list[dict[str, object]]:  # Continue or close the surrounding multiline expression or collection.
    """Create structure-level rows from compound- or scaffold-group assignments."""
    groups = defaultdict(list)  # Bind this name to an intermediate value, configuration setting, or result.
    for row in records:  # Iterate over the stated records, layers, batches, or graph elements.
        key = (  # Bind this name to an intermediate value, configuration setting, or result.
            str(row["official_compound_group_id"])
            if method == "random"
            else str(row["scaffold"])
        )  # Continue or close the surrounding multiline expression or collection.
        groups[key].append(str(row["complex_name"]))
    assignments = allocate_groups(groups, SEED)  # Bind this name to an intermediate value, configuration setting, or result.
    result = []  # Bind this name to an intermediate value, configuration setting, or result.
    for row in records:  # Iterate over the stated records, layers, batches, or graph elements.
        group_key = (  # Bind this name to an intermediate value, configuration setting, or result.
            str(row["official_compound_group_id"])
            if method == "random"
            else str(row["scaffold"])
        )  # Continue or close the surrounding multiline expression or collection.
        result.append(  # Perform this step of the surrounding calculation or control-flow block.
            {  # Continue or close the surrounding multiline expression or collection.
                "complex_name": row["complex_name"],
                "official_compound_group_id": row["official_compound_group_id"],
                "canonical_smiles": row["canonical_smiles"],
                "scaffold": row["scaffold"],
                "experimental_pKD": row["experimental_pKD"],
                "split": assignments[group_key],
                "split_method": method,
                "seed": SEED,
            }  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
    return result  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: validate_splits — see its docstring and inline comments.
def validate_splits(  # Define this callable; its indented block implements the documented operation.
    records: list[dict[str, object]],  # Perform this step of the surrounding calculation or control-flow block.
    random_rows: list[dict[str, object]],  # Perform this step of the surrounding calculation or control-flow block.
    scaffold_rows: list[dict[str, object]],  # Perform this step of the surrounding calculation or control-flow block.
) -> dict[str, object]:  # Continue or close the surrounding multiline expression or collection.
    """Prove coverage and absence of compound/scaffold leakage."""
    expected = {str(row["complex_name"]) for row in records}
    if {str(row["complex_name"]) for row in random_rows} != expected:
        raise ValueError("Random split does not cover the curated dataset")
    if {str(row["complex_name"]) for row in scaffold_rows} != expected:
        raise ValueError("Scaffold split does not cover the curated dataset")

    def leakage(rows: list[dict[str, object]], column: str) -> list[str]:  # Define this callable; its indented block implements the documented operation.
        """Return group identifiers assigned to more than one split."""
        memberships = defaultdict(set)  # Bind this name to an intermediate value, configuration setting, or result.
        for row in rows:  # Iterate over the stated records, layers, batches, or graph elements.
            memberships[str(row[column])].add(str(row["split"]))
        return sorted(key for key, splits in memberships.items() if len(splits) > 1)  # Return this computed tensor, metric, object, or collection to the caller.

    random_compound_leakage = leakage(  # Bind this name to an intermediate value, configuration setting, or result.
        random_rows, "official_compound_group_id"
    )  # Continue or close the surrounding multiline expression or collection.
    scaffold_compound_leakage = leakage(  # Bind this name to an intermediate value, configuration setting, or result.
        scaffold_rows, "official_compound_group_id"
    )  # Continue or close the surrounding multiline expression or collection.
    scaffold_leakage = leakage(scaffold_rows, "scaffold")
    if random_compound_leakage or scaffold_compound_leakage or scaffold_leakage:  # Evaluate this condition before executing the associated branch.
        raise ValueError("Leakage detected in generated split manifests")
    return {  # Return this computed tensor, metric, object, or collection to the caller.
        "random_counts": dict(Counter(row["split"] for row in random_rows)),
        "scaffold_counts": dict(
            Counter(row["split"] for row in scaffold_rows)
        ),  # Continue or close the surrounding multiline expression or collection.
        "random_compound_leakage_count": len(random_compound_leakage),
        "scaffold_compound_leakage_count": len(scaffold_compound_leakage),
        "scaffold_leakage_count": len(scaffold_leakage),
    }  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: prepare — see its docstring and inline comments.
def prepare(  # Define this callable; its indented block implements the documented operation.
    project_root: Path,  # Perform this step of the surrounding calculation or control-flow block.
    dataset_root: Path,  # Perform this step of the surrounding calculation or control-flow block.
    benchmark_reference: Path,  # Perform this step of the surrounding calculation or control-flow block.
    output_root: Path,  # Perform this step of the surrounding calculation or control-flow block.
) -> dict[str, object]:  # Continue or close the surrounding multiline expression or collection.
    """Create the complete quality-controlled Experiment A dataset."""
    metadata_path = dataset_root / "EV-A71_2A_metadata.csv"
    manifest_path = dataset_root / "reports" / "structure_file_manifest.csv"
    atom_init_source = project_root / "examples" / "example_data" / "atom_init.json"
    metadata = read_csv(metadata_path)  # Bind this name to an intermediate value, configuration setting, or result.
    manifest = {  # Prepare dataset membership or batched data access for the experiment.
        row["complex_name"]: row for row in read_csv(manifest_path)
    }  # Continue or close the surrounding multiline expression or collection.
    benchmark = {  # Bind this name to an intermediate value, configuration setting, or result.
        row["fragalysis_code"]: row for row in read_csv(benchmark_reference)
    }  # Continue or close the surrounding multiline expression or collection.
    if len(metadata) != 925 or len(manifest) != 925:  # Evaluate this condition before executing the associated branch.
        raise ValueError("Expected exactly 925 OpenBind metadata mappings")

    output_root.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    raw_dir = output_root / "raw"
    curated_dir = output_root / "curated"
    split_dir = output_root / "splits"
    report_dir = output_root / "reports"
    for directory in (raw_dir, curated_dir, split_dir, report_dir):  # Iterate over the stated records, layers, batches, or graph elements.
        directory.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    atom_init_destination = output_root / "atom_init.json"
    if atom_init_source.is_file():  # Evaluate this condition before executing the associated branch.
        shutil.copyfile(atom_init_source, atom_init_destination)  # Perform this step of the surrounding calculation or control-flow block.
    elif not atom_init_destination.is_file():  # Test this additional condition when earlier branches were not selected.
        raise FileNotFoundError(  # Reject invalid input or state with an explicit exception.
            "Original MGT atom_init.json is absent from both examples and output"
        )  # Continue or close the surrounding multiline expression or collection.

    curated_rows = []  # Bind this name to an intermediate value, configuration setting, or result.
    excluded_rows = []  # Bind this name to an intermediate value, configuration setting, or result.
    for row in metadata:  # Iterate over the stated records, layers, batches, or graph elements.
        reasons = exclusion_reasons(row, set(benchmark))  # Bind this name to an intermediate value, configuration setting, or result.
        if reasons:  # Evaluate this condition before executing the associated branch.
            excluded_rows.append(  # Perform this step of the surrounding calculation or control-flow block.
                {  # Continue or close the surrounding multiline expression or collection.
                    "complex_name": row["complex_name"],
                    "compound_group": row["compound_group"],
                    "smiles": row["smiles"],
                    "experimental_pKD": row["experimental_pKD"],
                    "exclusion_reasons": ";".join(reasons),
                }  # Continue or close the surrounding multiline expression or collection.
            )  # Continue or close the surrounding multiline expression or collection.
            continue  # Alter loop control for this condition.
        mapping = manifest[row["complex_name"]]
        reference = benchmark[row["complex_name"]]
        source_sdf = dataset_root / mapping["ligand_ref_sdf_path"]
        destination = raw_dir / f"{row['complex_name']}.pdb"
        pdb_audit = write_ligand_pdb(source_sdf, destination)  # Bind this name to an intermediate value, configuration setting, or result.
        metadata_molecule = Chem.MolFromSmiles(row["smiles"])
        if metadata_molecule is None:  # Evaluate this condition before executing the associated branch.
            raise ValueError(f"Invalid metadata SMILES for {row['complex_name']}")
        if canonical_smiles(metadata_molecule) != pdb_audit["canonical_smiles"]:
            raise ValueError(  # Reject invalid input or state with an explicit exception.
                f"Metadata/SDF identity mismatch for {row['complex_name']}"
            )  # Continue or close the surrounding multiline expression or collection.
        metadata_pkd = float(row["experimental_pKD"])
        benchmark_pkd = float(reference["experimental_pKD"])
        if abs(metadata_pkd - benchmark_pkd) > 0.000051:  # Evaluate this condition before executing the associated branch.
            raise ValueError(f"pKD mismatch for {row['complex_name']}")
        official_compound_group_id = hashlib.sha256(  # Bind this name to an intermediate value, configuration setting, or result.
            reference["smiles"].encode("utf-8")
        ).hexdigest()[:16]  # Continue or close the surrounding multiline expression or collection.
        curated_rows.append(  # Perform this step of the surrounding calculation or control-flow block.
            {  # Continue or close the surrounding multiline expression or collection.
                "complex_name": row["complex_name"],
                "compound_group": row["compound_group"],
                "official_compound_group_id": official_compound_group_id,
                "metadata_smiles": row["smiles"],
                "benchmark_smiles": reference["smiles"],
                "canonical_smiles": pdb_audit["canonical_smiles"],
                "scaffold": pdb_audit["scaffold"],
                "experimental_pKD": benchmark_pkd,
                "fragment_screen": row["fragment_screen"],
                "pb_valid_ref": row["pb_valid_ref"],
                "source_ligand_ref_sdf": mapping["ligand_ref_sdf_path"],
                "source_ligand_ref_sdf_sha256": mapping[
                    "ligand_ref_sdf_sha256"
                ],  # Continue or close the surrounding multiline expression or collection.
                "ligand_pdb": f"raw/{destination.name}",
                "ligand_pdb_sha256": pdb_audit["pdb_sha256"],
                "atom_count": pdb_audit["atom_count"],
                "heavy_atom_count": pdb_audit["heavy_atom_count"],
                "formal_charge": pdb_audit["formal_charge"],
                "maximum_coordinate_difference_angstrom": pdb_audit[
                    "maximum_coordinate_difference_angstrom"
                ],  # Continue or close the surrounding multiline expression or collection.
            }  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.

    curated_fields = list(curated_rows[0])  # Bind this name to an intermediate value, configuration setting, or result.
    write_csv(  # Perform this step of the surrounding calculation or control-flow block.
        curated_dir / "openbind_ligand_structures.csv",
        curated_rows,  # Perform this step of the surrounding calculation or control-flow block.
        curated_fields,  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    write_csv(  # Perform this step of the surrounding calculation or control-flow block.
        curated_dir / "excluded_records.csv",
        excluded_rows,  # Perform this step of the surrounding calculation or control-flow block.
        [  # Continue or close the surrounding multiline expression or collection.
            "complex_name",
            "compound_group",
            "smiles",
            "experimental_pKD",
            "exclusion_reasons",
        ],  # Continue or close the surrounding multiline expression or collection.
    )  # Continue or close the surrounding multiline expression or collection.
    with (output_root / "id_prop.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        writer = csv.writer(handle)  # Bind this name to an intermediate value, configuration setting, or result.
        for row in curated_rows:  # Iterate over the stated records, layers, batches, or graph elements.
            writer.writerow([row["complex_name"], row["experimental_pKD"]])

    compounds = defaultdict(list)  # Bind this name to an intermediate value, configuration setting, or result.
    for row in curated_rows:  # Iterate over the stated records, layers, batches, or graph elements.
        compounds[str(row["official_compound_group_id"])].append(row)
    compound_rows = []  # Bind this name to an intermediate value, configuration setting, or result.
    for compound_id, rows in sorted(compounds.items()):  # Iterate over the stated records, layers, batches, or graph elements.
        compound_rows.append(  # Perform this step of the surrounding calculation or control-flow block.
            {  # Continue or close the surrounding multiline expression or collection.
                "official_compound_group_id": compound_id,
                "benchmark_smiles": rows[0]["benchmark_smiles"],
                "experimental_pKD": rows[0]["experimental_pKD"],
                "structure_count": len(rows),
                "complex_names": ";".join(
                    sorted(str(row["complex_name"]) for row in rows)
                ),  # Continue or close the surrounding multiline expression or collection.
                "metadata_identity_count": len(
                    {str(row["canonical_smiles"]) for row in rows}
                ),  # Continue or close the surrounding multiline expression or collection.
                "scaffold_count": len({str(row["scaffold"]) for row in rows}),
            }  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
    write_csv(  # Perform this step of the surrounding calculation or control-flow block.
        curated_dir / "openbind_compounds.csv",
        compound_rows,  # Perform this step of the surrounding calculation or control-flow block.
        list(compound_rows[0]),  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.

    random_rows = build_split_rows(curated_rows, "random")
    scaffold_rows = build_split_rows(curated_rows, "scaffold")
    split_fields = list(random_rows[0])  # Prepare dataset membership or batched data access for the experiment.
    random_path = split_dir / "random_seed_123.csv"
    scaffold_path = split_dir / "scaffold_seed_123.csv"
    write_csv(random_path, random_rows, split_fields)  # Perform this step of the surrounding calculation or control-flow block.
    write_csv(scaffold_path, scaffold_rows, split_fields)  # Perform this step of the surrounding calculation or control-flow block.
    split_audit = validate_splits(curated_rows, random_rows, scaffold_rows)  # Prepare dataset membership or batched data access for the experiment.

    curated_path = curated_dir / "openbind_ligand_structures.csv"
    excluded_path = curated_dir / "excluded_records.csv"
    compound_path = curated_dir / "openbind_compounds.csv"
    id_prop_path = output_root / "id_prop.csv"
    summary = {  # Bind this name to an intermediate value, configuration setting, or result.
        "dataset": "OpenBind EV-A71 2A Experiment A",
        "dataset_version": "1.0",
        "representation": "crystallographic ligand-only PDB",
        "source_structure": "ligand_ref.sdf",
        "coordinate_preservation_tolerance_angstrom": 0.00011,
        "label": "experimental_pKD from official benchmark reference",
        "seed": SEED,
        "requested_split_proportions": PROPORTIONS,
        "cleaning_rules": [
            "experimental_pKD is present",
            "complex is present in official affinity reference",
            "covalent is False",
            "suspected_artefact is False",
            "pb_valid_ref is True",
            "metadata SMILES and reference SDF canonical identities agree",
        ],  # Continue or close the surrounding multiline expression or collection.
        "counts": {
            "source_metadata_rows": len(metadata),
            "curated_structure_rows": len(curated_rows),
            "excluded_structure_rows": len(excluded_rows),
            "benchmark_compound_groups": len(compounds),
            "generated_ligand_pdb_files": len(list(raw_dir.glob("*.pdb"))),
        },  # Continue or close the surrounding multiline expression or collection.
        "exclusion_reason_counts": dict(
            Counter(  # Perform this step of the surrounding calculation or control-flow block.
                reason  # Perform this step of the surrounding calculation or control-flow block.
                for row in excluded_rows  # Iterate over the stated records, layers, batches, or graph elements.
                for reason in str(row["exclusion_reasons"]).split(";")
            )  # Continue or close the surrounding multiline expression or collection.
        ),  # Continue or close the surrounding multiline expression or collection.
        "split_audit": split_audit,
        "rdkit_version": rdBase.rdkitVersion,
        "sha256": {
            "source_metadata": sha256_file(metadata_path),
            "source_structure_manifest": sha256_file(manifest_path),
            "official_benchmark_reference": sha256_file(benchmark_reference),
            "atom_init": sha256_file(output_root / "atom_init.json"),
            "id_prop": sha256_file(id_prop_path),
            "curated_structures": sha256_file(curated_path),
            "curated_compounds": sha256_file(compound_path),
            "excluded_records": sha256_file(excluded_path),
            "random_split": sha256_file(random_path),
            "scaffold_split": sha256_file(scaffold_path),
        },  # Continue or close the surrounding multiline expression or collection.
    }  # Continue or close the surrounding multiline expression or collection.
    with (report_dir / "dataset_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        json.dump(summary, handle, indent=2, sort_keys=True)  # Bind this name to an intermediate value, configuration setting, or result.
        handle.write("\n")
    return summary  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: main — see its docstring and inline comments.
def main() -> None:  # Define this callable; its indented block implements the documented operation.
    """Parse paths and create the complete Experiment A dataset."""
    project_root = Path(__file__).resolve().parents[1]  # Bind this name to an intermediate value, configuration setting, or result.
    parser = argparse.ArgumentParser(  # Bind this name to an intermediate value, configuration setting, or result.
        description="Prepare OpenBind crystallographic ligand PDBs for MGT."
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--dataset_root",
        type=Path,  # Bind this name to an intermediate value, configuration setting, or result.
        default=project_root  # Bind this name to an intermediate value, configuration setting, or result.
        / "OpenBind_EV-A71_2A"
        / "OpenBind_EV-A71_2A",
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--benchmark_reference",
        type=Path,  # Bind this name to an intermediate value, configuration setting, or result.
        default=project_root.parent  # Bind this name to an intermediate value, configuration setting, or result.
        / "EV-A71_2A_benchmark"
        / "affinity"
        / "reference"
        / "fragalysis_compound_reference.csv",
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--output_root",
        type=Path,  # Bind this name to an intermediate value, configuration setting, or result.
        default=project_root  # Bind this name to an intermediate value, configuration setting, or result.
        / "OpenBind_EV-A71_2A"
        / "experiment_a_ligand_mgt",
    )  # Continue or close the surrounding multiline expression or collection.
    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.
    summary = prepare(  # Bind this name to an intermediate value, configuration setting, or result.
        project_root=project_root,  # Bind this name to an intermediate value, configuration setting, or result.
        dataset_root=args.dataset_root.resolve(),  # Prepare dataset membership or batched data access for the experiment.
        benchmark_reference=args.benchmark_reference.resolve(),  # Bind this name to an intermediate value, configuration setting, or result.
        output_root=args.output_root.resolve(),  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    print(json.dumps(summary, indent=2, sort_keys=True))  # Report progress, predictions, or metrics to the selected output/logging backend.


if __name__ == "__main__":
    main()  # Perform this step of the surrounding calculation or control-flow block.
