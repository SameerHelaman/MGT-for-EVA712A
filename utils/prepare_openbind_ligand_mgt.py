"""Prepare quality-controlled crystallographic ligand PDBs for original MGT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


SEED = 123
PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def sha256_file(path: Path) -> str:
    """Calculate a file checksum in bounded-memory blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dictionaries without changing source values."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    """Write a stable CSV with an explicit column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    """Interpret the literal boolean fields supplied by OpenBind."""
    if value not in {"True", "False"}:
        raise ValueError(f"Unexpected boolean value: {value!r}")
    return value == "True"


def canonical_smiles(molecule: Chem.Mol) -> str:
    """Return an isomeric canonical identity for one ligand."""
    return Chem.MolToSmiles(
        Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True
    )


def scaffold_smiles(molecule: Chem.Mol, fallback: str) -> str:
    """Return a chiral Bemis-Murcko scaffold or a unique acyclic key."""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=Chem.RemoveHs(molecule), includeChirality=True
    )
    return scaffold if scaffold else f"ACYCLIC::{fallback}"


def exclusion_reasons(row: dict[str, str], benchmark_codes: set[str]) -> list[str]:
    """Return every prespecified reason that makes a structure ineligible."""
    reasons = []
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
    return reasons


def write_ligand_pdb(source_sdf: Path, destination: Path) -> dict[str, object]:
    """Convert one reference SDF to PDB while preserving its coordinates."""
    supplier = Chem.SDMolSupplier(
        str(source_sdf), removeHs=False, sanitize=True
    )
    molecules = [mol for mol in supplier if mol is not None]
    if len(molecules) != 1:
        raise ValueError(f"{source_sdf} contains {len(molecules)} molecules")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1:
        raise ValueError(f"{source_sdf} does not contain exactly one conformer")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToPDBFile(molecule, str(destination), flavor=4)
    parsed = Chem.MolFromPDBFile(
        str(destination), sanitize=True, removeHs=False
    )
    if parsed is None:
        raise ValueError(f"RDKit failed to parse generated PDB {destination}")
    if parsed.GetNumAtoms() != molecule.GetNumAtoms():
        raise ValueError(f"Atom count changed while writing {destination}")
    source_positions = molecule.GetConformer().GetPositions()
    pdb_positions = parsed.GetConformer().GetPositions()
    maximum_coordinate_difference = float(
        abs(source_positions - pdb_positions).max()
    )
    if maximum_coordinate_difference > 0.00011:
        raise ValueError(
            f"Coordinates changed by {maximum_coordinate_difference} Å "
            f"while writing {destination}"
        )
    return {
        "canonical_smiles": canonical_smiles(molecule),
        "scaffold": scaffold_smiles(molecule, destination.stem),
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "maximum_coordinate_difference_angstrom": maximum_coordinate_difference,
        "pdb_sha256": sha256_file(destination),
    }


def allocate_groups(
    group_to_members: dict[str, list[str]], seed: int
) -> dict[str, str]:
    """Allocate indivisible groups toward 70/15/15 structure-level targets."""
    rng = random.Random(seed)
    items = list(group_to_members.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    total = sum(len(members) for _, members in items)
    targets = {name: total * value for name, value in PROPORTIONS.items()}
    counts = Counter()
    assignments = {}
    for group, members in items:
        split = min(
            PROPORTIONS,
            key=lambda name: (
                counts[name] / targets[name]
                if targets[name] > 0
                else float("inf"),
                counts[name],
                name,
            ),
        )
        assignments[group] = split
        counts[split] += len(members)
    return assignments


def build_split_rows(
    records: list[dict[str, object]], method: str
) -> list[dict[str, object]]:
    """Create structure-level rows from compound- or scaffold-group assignments."""
    groups = defaultdict(list)
    for row in records:
        key = (
            str(row["official_compound_group_id"])
            if method == "random"
            else str(row["scaffold"])
        )
        groups[key].append(str(row["complex_name"]))
    assignments = allocate_groups(groups, SEED)
    result = []
    for row in records:
        group_key = (
            str(row["official_compound_group_id"])
            if method == "random"
            else str(row["scaffold"])
        )
        result.append(
            {
                "complex_name": row["complex_name"],
                "official_compound_group_id": row["official_compound_group_id"],
                "canonical_smiles": row["canonical_smiles"],
                "scaffold": row["scaffold"],
                "experimental_pKD": row["experimental_pKD"],
                "split": assignments[group_key],
                "split_method": method,
                "seed": SEED,
            }
        )
    return result


def validate_splits(
    records: list[dict[str, object]],
    random_rows: list[dict[str, object]],
    scaffold_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Prove coverage and absence of compound/scaffold leakage."""
    expected = {str(row["complex_name"]) for row in records}
    if {str(row["complex_name"]) for row in random_rows} != expected:
        raise ValueError("Random split does not cover the curated dataset")
    if {str(row["complex_name"]) for row in scaffold_rows} != expected:
        raise ValueError("Scaffold split does not cover the curated dataset")

    def leakage(rows: list[dict[str, object]], column: str) -> list[str]:
        """Return group identifiers assigned to more than one split."""
        memberships = defaultdict(set)
        for row in rows:
            memberships[str(row[column])].add(str(row["split"]))
        return sorted(key for key, splits in memberships.items() if len(splits) > 1)

    random_compound_leakage = leakage(
        random_rows, "official_compound_group_id"
    )
    scaffold_compound_leakage = leakage(
        scaffold_rows, "official_compound_group_id"
    )
    scaffold_leakage = leakage(scaffold_rows, "scaffold")
    if random_compound_leakage or scaffold_compound_leakage or scaffold_leakage:
        raise ValueError("Leakage detected in generated split manifests")
    return {
        "random_counts": dict(Counter(row["split"] for row in random_rows)),
        "scaffold_counts": dict(
            Counter(row["split"] for row in scaffold_rows)
        ),
        "random_compound_leakage_count": len(random_compound_leakage),
        "scaffold_compound_leakage_count": len(scaffold_compound_leakage),
        "scaffold_leakage_count": len(scaffold_leakage),
    }


def prepare(
    project_root: Path,
    dataset_root: Path,
    benchmark_reference: Path,
    output_root: Path,
) -> dict[str, object]:
    """Create the complete quality-controlled Experiment A dataset."""
    metadata_path = dataset_root / "EV-A71_2A_metadata.csv"
    manifest_path = dataset_root / "reports" / "structure_file_manifest.csv"
    atom_init_source = project_root / "examples" / "example_data" / "atom_init.json"
    metadata = read_csv(metadata_path)
    manifest = {
        row["complex_name"]: row for row in read_csv(manifest_path)
    }
    benchmark = {
        row["fragalysis_code"]: row for row in read_csv(benchmark_reference)
    }
    if len(metadata) != 925 or len(manifest) != 925:
        raise ValueError("Expected exactly 925 OpenBind metadata mappings")

    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "raw"
    curated_dir = output_root / "curated"
    split_dir = output_root / "splits"
    report_dir = output_root / "reports"
    for directory in (raw_dir, curated_dir, split_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    atom_init_destination = output_root / "atom_init.json"
    if atom_init_source.is_file():
        shutil.copyfile(atom_init_source, atom_init_destination)
    elif not atom_init_destination.is_file():
        raise FileNotFoundError(
            "Original MGT atom_init.json is absent from both examples and output"
        )

    curated_rows = []
    excluded_rows = []
    for row in metadata:
        reasons = exclusion_reasons(row, set(benchmark))
        if reasons:
            excluded_rows.append(
                {
                    "complex_name": row["complex_name"],
                    "compound_group": row["compound_group"],
                    "smiles": row["smiles"],
                    "experimental_pKD": row["experimental_pKD"],
                    "exclusion_reasons": ";".join(reasons),
                }
            )
            continue
        mapping = manifest[row["complex_name"]]
        reference = benchmark[row["complex_name"]]
        source_sdf = dataset_root / mapping["ligand_ref_sdf_path"]
        destination = raw_dir / f"{row['complex_name']}.pdb"
        pdb_audit = write_ligand_pdb(source_sdf, destination)
        metadata_molecule = Chem.MolFromSmiles(row["smiles"])
        if metadata_molecule is None:
            raise ValueError(f"Invalid metadata SMILES for {row['complex_name']}")
        if canonical_smiles(metadata_molecule) != pdb_audit["canonical_smiles"]:
            raise ValueError(
                f"Metadata/SDF identity mismatch for {row['complex_name']}"
            )
        metadata_pkd = float(row["experimental_pKD"])
        benchmark_pkd = float(reference["experimental_pKD"])
        if abs(metadata_pkd - benchmark_pkd) > 0.000051:
            raise ValueError(f"pKD mismatch for {row['complex_name']}")
        official_compound_group_id = hashlib.sha256(
            reference["smiles"].encode("utf-8")
        ).hexdigest()[:16]
        curated_rows.append(
            {
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
                ],
                "ligand_pdb": f"raw/{destination.name}",
                "ligand_pdb_sha256": pdb_audit["pdb_sha256"],
                "atom_count": pdb_audit["atom_count"],
                "heavy_atom_count": pdb_audit["heavy_atom_count"],
                "formal_charge": pdb_audit["formal_charge"],
                "maximum_coordinate_difference_angstrom": pdb_audit[
                    "maximum_coordinate_difference_angstrom"
                ],
            }
        )

    curated_fields = list(curated_rows[0])
    write_csv(
        curated_dir / "openbind_ligand_structures.csv",
        curated_rows,
        curated_fields,
    )
    write_csv(
        curated_dir / "excluded_records.csv",
        excluded_rows,
        [
            "complex_name",
            "compound_group",
            "smiles",
            "experimental_pKD",
            "exclusion_reasons",
        ],
    )
    with (output_root / "id_prop.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        for row in curated_rows:
            writer.writerow([row["complex_name"], row["experimental_pKD"]])

    compounds = defaultdict(list)
    for row in curated_rows:
        compounds[str(row["official_compound_group_id"])].append(row)
    compound_rows = []
    for compound_id, rows in sorted(compounds.items()):
        compound_rows.append(
            {
                "official_compound_group_id": compound_id,
                "benchmark_smiles": rows[0]["benchmark_smiles"],
                "experimental_pKD": rows[0]["experimental_pKD"],
                "structure_count": len(rows),
                "complex_names": ";".join(
                    sorted(str(row["complex_name"]) for row in rows)
                ),
                "metadata_identity_count": len(
                    {str(row["canonical_smiles"]) for row in rows}
                ),
                "scaffold_count": len({str(row["scaffold"]) for row in rows}),
            }
        )
    write_csv(
        curated_dir / "openbind_compounds.csv",
        compound_rows,
        list(compound_rows[0]),
    )

    random_rows = build_split_rows(curated_rows, "random")
    scaffold_rows = build_split_rows(curated_rows, "scaffold")
    split_fields = list(random_rows[0])
    random_path = split_dir / "random_seed_123.csv"
    scaffold_path = split_dir / "scaffold_seed_123.csv"
    write_csv(random_path, random_rows, split_fields)
    write_csv(scaffold_path, scaffold_rows, split_fields)
    split_audit = validate_splits(curated_rows, random_rows, scaffold_rows)

    curated_path = curated_dir / "openbind_ligand_structures.csv"
    excluded_path = curated_dir / "excluded_records.csv"
    compound_path = curated_dir / "openbind_compounds.csv"
    id_prop_path = output_root / "id_prop.csv"
    summary = {
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
        ],
        "counts": {
            "source_metadata_rows": len(metadata),
            "curated_structure_rows": len(curated_rows),
            "excluded_structure_rows": len(excluded_rows),
            "benchmark_compound_groups": len(compounds),
            "generated_ligand_pdb_files": len(list(raw_dir.glob("*.pdb"))),
        },
        "exclusion_reason_counts": dict(
            Counter(
                reason
                for row in excluded_rows
                for reason in str(row["exclusion_reasons"]).split(";")
            )
        ),
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
        },
    }
    with (report_dir / "dataset_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    """Parse paths and create the complete Experiment A dataset."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare OpenBind crystallographic ligand PDBs for MGT."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=project_root
        / "OpenBind_EV-A71_2A"
        / "OpenBind_EV-A71_2A",
    )
    parser.add_argument(
        "--benchmark_reference",
        type=Path,
        default=project_root.parent
        / "EV-A71_2A_benchmark"
        / "affinity"
        / "reference"
        / "fragalysis_compound_reference.csv",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=project_root
        / "OpenBind_EV-A71_2A"
        / "experiment_a_ligand_mgt",
    )
    args = parser.parse_args()
    summary = prepare(
        project_root=project_root,
        dataset_root=args.dataset_root.resolve(),
        benchmark_reference=args.benchmark_reference.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
