"""Matched 2D and crystallographic 3D graph datasets for OpenBind."""

from __future__ import annotations

import csv
from pathlib import Path

import dgl
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from utils.molecular_features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    atom_features,
    bond_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "OpenBind_EV-A71_2A" / "experiment_a_ligand_mgt"
)
SOURCE_DATASET_ROOT = (
    PROJECT_ROOT / "OpenBind_EV-A71_2A" / "OpenBind_EV-A71_2A"
)


def read_reference_molecule(path: Path) -> Chem.Mol:
    """Read one crystallographic reference SDF with its atom order and pose."""
    supplier = Chem.SDMolSupplier(
        str(path), removeHs=False, sanitize=True
    )
    molecules = [molecule for molecule in supplier if molecule is not None]
    if len(molecules) != 1:
        raise ValueError(f"Expected one molecule in {path}")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1:
        raise ValueError(f"Expected one conformer in {path}")
    return molecule


def molecule_to_2d_graph(molecule: Chem.Mol) -> dgl.DGLGraph:
    """Build a directed chemical-bond graph without coordinate information."""
    sources = []
    destinations = []
    edge_values = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        sources.extend([begin, end])
        destinations.extend([end, begin])
        edge_values.extend([features, features])
    graph = dgl.graph(
        (sources, destinations), num_nodes=molecule.GetNumAtoms()
    )
    graph.ndata["atom_features"] = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )
    graph.edata["bond_features"] = (
        torch.tensor(edge_values, dtype=torch.float32)
        if edge_values
        else torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    )
    return graph


def molecule_to_3d_graph(
    molecule: Chem.Mol,
    spatial_cutoff: float,
    max_neighbors: int,
) -> dgl.DGLGraph:
    """Build the matched bond/spatial graph from crystallographic coordinates."""
    coordinates = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=np.float32
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("Reference SDF contains non-finite coordinates")
    atom_count = molecule.GetNumAtoms()
    displacement = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(displacement, axis=-1)
    bond_lookup = {}
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        bond_lookup[(begin, end)] = features
        bond_lookup[(end, begin)] = features
    edges = set(bond_lookup)
    for source in range(atom_count):
        candidates = [
            (float(distances[source, destination]), destination)
            for destination in range(atom_count)
            if destination != source
            and distances[source, destination] <= spatial_cutoff
        ]
        for _, destination in sorted(candidates)[:max_neighbors]:
            edges.add((source, destination))
    ordered_edges = sorted(edges)
    graph = dgl.graph(
        (
            [edge[0] for edge in ordered_edges],
            [edge[1] for edge in ordered_edges],
        ),
        num_nodes=atom_count,
    )
    graph.ndata["atom_features"] = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )
    graph.ndata["coordinates"] = torch.from_numpy(coordinates.copy())
    graph.edata["bond_features"] = torch.tensor(
        [
            bond_lookup.get(edge, [0.0] * BOND_FEATURE_DIM)
            for edge in ordered_edges
        ],
        dtype=torch.float32,
    )
    graph.edata["distance"] = torch.tensor(
        [[float(distances[source, destination])]
         for source, destination in ordered_edges],
        dtype=torch.float32,
    )
    graph.edata["r"] = torch.from_numpy(
        np.asarray(
            [
                coordinates[destination] - coordinates[source]
                for source, destination in ordered_edges
            ],
            dtype=np.float32,
        )
    )
    return graph


class OpenBindGraphDataset(Dataset):
    """Load the curated OpenBind cohort as 2D or crystal-pose 3D graphs."""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        mode: str = "2d_gnn",
        spatial_cutoff: float = 5.0,
        max_neighbors: int = 32,
    ) -> None:
        """Initialize this object and its required state."""
        if mode not in {"2d_gnn", "3d_gnn", "3d_alignn"}:
            raise ValueError(f"Unsupported graph mode: {mode}")
        self.data_root = Path(data_root).resolve()
        curated_path = (
            self.data_root
            / "curated"
            / "openbind_ligand_structures.csv"
        )
        with curated_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            self.rows = list(csv.DictReader(handle))
        self.sample_ids = [row["complex_name"] for row in self.rows]
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("Duplicate complex names in curated dataset")
        self.mode = mode
        self.spatial_cutoff = spatial_cutoff
        self.max_neighbors = max_neighbors

    def __len__(self) -> int:
        """Return the number of dataset records."""
        return len(self.rows)

    def __getitem__(self, index: int):
        """Load and return one indexed dataset record."""
        row = self.rows[index]
        source_path = SOURCE_DATASET_ROOT / row["source_ligand_ref_sdf"]
        molecule = read_reference_molecule(source_path)
        graph = (
            molecule_to_2d_graph(molecule)
            if self.mode == "2d_gnn"
            else molecule_to_3d_graph(
                molecule, self.spatial_cutoff, self.max_neighbors
            )
        )
        target = torch.tensor(
            float(row["experimental_pKD"]), dtype=torch.float32
        )
        return graph, target, row["complex_name"]


def validate_feature_dimensions() -> tuple[int, int]:
    """Expose and validate the declared feature dimensions."""
    if ATOM_FEATURE_DIM != 152 or BOND_FEATURE_DIM != 12:
        raise RuntimeError("Unexpected molecular feature dimensions")
    return ATOM_FEATURE_DIM, BOND_FEATURE_DIM
