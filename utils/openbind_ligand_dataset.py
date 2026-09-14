# =============================================================================
# MODULE: utils/openbind_ligand_dataset.py
# PURPOSE: Loads curated ligand SDFs and constructs controlled 2D/3D DGL graphs.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: Spatial edges satisfy distance <= cutoff and maximum-neighbour constraints; distance is stored for RBF expansion.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Matched 2D and crystallographic 3D graph datasets for OpenBind."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import csv  # Load a standard-library, scientific, or local project dependency.
from pathlib import Path  # Import selected classes or functions from the named dependency.

import dgl  # Load a standard-library, scientific, or local project dependency.
import numpy as np  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
from rdkit import Chem  # Import selected classes or functions from the named dependency.
from torch.utils.data import Dataset  # Import selected classes or functions from the named dependency.

from utils.molecular_features import (  # Import selected classes or functions from the named dependency.
    ATOM_FEATURE_DIM,  # Perform this step of the surrounding calculation or control-flow block.
    BOND_FEATURE_DIM,  # Perform this step of the surrounding calculation or control-flow block.
    atom_features,  # Perform this step of the surrounding calculation or control-flow block.
    bond_features,  # Perform this step of the surrounding calculation or control-flow block.
)  # Continue or close the surrounding multiline expression or collection.


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Bind this name to an intermediate value, configuration setting, or result.
DEFAULT_DATA_ROOT = (  # Bind this name to an intermediate value, configuration setting, or result.
    PROJECT_ROOT / "OpenBind_EV-A71_2A" / "experiment_a_ligand_mgt"
)  # Continue or close the surrounding multiline expression or collection.
SOURCE_DATASET_ROOT = (  # Prepare dataset membership or batched data access for the experiment.
    PROJECT_ROOT / "OpenBind_EV-A71_2A" / "OpenBind_EV-A71_2A"
)  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: read_reference_molecule — see its docstring and inline comments.
def read_reference_molecule(path: Path) -> Chem.Mol:  # Define this callable; its indented block implements the documented operation.
    """Read one crystallographic reference SDF with its atom order and pose."""
    supplier = Chem.SDMolSupplier(  # Bind this name to an intermediate value, configuration setting, or result.
        str(path), removeHs=False, sanitize=True  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    molecules = [molecule for molecule in supplier if molecule is not None]  # Bind this name to an intermediate value, configuration setting, or result.
    if len(molecules) != 1:  # Evaluate this condition before executing the associated branch.
        raise ValueError(f"Expected one molecule in {path}")  # Reject invalid input or state with an explicit exception.
    molecule = molecules[0]  # Bind this name to an intermediate value, configuration setting, or result.
    if molecule.GetNumConformers() != 1:  # Evaluate this condition before executing the associated branch.
        raise ValueError(f"Expected one conformer in {path}")  # Reject invalid input or state with an explicit exception.
    return molecule  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: molecule_to_2d_graph — see its docstring and inline comments.
def molecule_to_2d_graph(molecule: Chem.Mol) -> dgl.DGLGraph:  # Define this callable; its indented block implements the documented operation.
    """Build a directed chemical-bond graph without coordinate information."""
    sources = []  # Bind this name to an intermediate value, configuration setting, or result.
    destinations = []  # Bind this name to an intermediate value, configuration setting, or result.
    edge_values = []  # Construct or transform graph topology, geometry, or molecular feature data.
    for bond in molecule.GetBonds():  # Iterate over the stated records, layers, batches, or graph elements.
        begin = bond.GetBeginAtomIdx()  # Construct or transform graph topology, geometry, or molecular feature data.
        end = bond.GetEndAtomIdx()  # Construct or transform graph topology, geometry, or molecular feature data.
        features = bond_features(bond)  # Construct or transform graph topology, geometry, or molecular feature data.
        sources.extend([begin, end])  # Perform this step of the surrounding calculation or control-flow block.
        destinations.extend([end, begin])  # Perform this step of the surrounding calculation or control-flow block.
        edge_values.extend([features, features])  # Perform this step of the surrounding calculation or control-flow block.
    graph = dgl.graph(  # Construct or transform graph topology, geometry, or molecular feature data.
        (sources, destinations), num_nodes=molecule.GetNumAtoms()  # Construct or transform graph topology, geometry, or molecular feature data.
    )  # Continue or close the surrounding multiline expression or collection.
    graph.ndata["atom_features"] = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()],  # Continue or close the surrounding multiline expression or collection.
        dtype=torch.float32,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    graph.edata["bond_features"] = (
        torch.tensor(edge_values, dtype=torch.float32)  # Construct or transform graph topology, geometry, or molecular feature data.
        if edge_values  # Evaluate this condition before executing the associated branch.
        else torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)  # Handle the remaining case not covered by earlier conditions.
    )  # Continue or close the surrounding multiline expression or collection.
    return graph  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: molecule_to_3d_graph — see its docstring and inline comments.
def molecule_to_3d_graph(  # Define this callable; its indented block implements the documented operation.
    molecule: Chem.Mol,  # Perform this step of the surrounding calculation or control-flow block.
    spatial_cutoff: float,  # Perform this step of the surrounding calculation or control-flow block.
    max_neighbors: int,  # Perform this step of the surrounding calculation or control-flow block.
) -> dgl.DGLGraph:  # Continue or close the surrounding multiline expression or collection.
    """Build the matched bond/spatial graph from crystallographic coordinates."""
    coordinates = np.asarray(  # Bind this name to an intermediate value, configuration setting, or result.
        molecule.GetConformer().GetPositions(), dtype=np.float32  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    if not np.isfinite(coordinates).all():  # Evaluate this condition before executing the associated branch.
        raise ValueError("Reference SDF contains non-finite coordinates")
    atom_count = molecule.GetNumAtoms()  # Bind this name to an intermediate value, configuration setting, or result.
    displacement = coordinates[:, None, :] - coordinates[None, :, :]  # Bind this name to an intermediate value, configuration setting, or result.
    distances = np.linalg.norm(displacement, axis=-1)  # Construct or transform graph topology, geometry, or molecular feature data.
    bond_lookup = {}  # Construct or transform graph topology, geometry, or molecular feature data.
    for bond in molecule.GetBonds():  # Iterate over the stated records, layers, batches, or graph elements.
        begin = bond.GetBeginAtomIdx()  # Construct or transform graph topology, geometry, or molecular feature data.
        end = bond.GetEndAtomIdx()  # Construct or transform graph topology, geometry, or molecular feature data.
        features = bond_features(bond)  # Construct or transform graph topology, geometry, or molecular feature data.
        bond_lookup[(begin, end)] = features  # Construct or transform graph topology, geometry, or molecular feature data.
        bond_lookup[(end, begin)] = features  # Construct or transform graph topology, geometry, or molecular feature data.
    edges = set(bond_lookup)  # Construct or transform graph topology, geometry, or molecular feature data.
    for source in range(atom_count):  # Iterate over the stated records, layers, batches, or graph elements.
        candidates = [  # Bind this name to an intermediate value, configuration setting, or result.
            (float(distances[source, destination]), destination)  # Continue or close the surrounding multiline expression or collection.
            for destination in range(atom_count)  # Iterate over the stated records, layers, batches, or graph elements.
            if destination != source  # Evaluate this condition before executing the associated branch.
            and distances[source, destination] <= spatial_cutoff  # Construct or transform graph topology, geometry, or molecular feature data.
        ]  # Continue or close the surrounding multiline expression or collection.
        for _, destination in sorted(candidates)[:max_neighbors]:  # Iterate over the stated records, layers, batches, or graph elements.
            edges.add((source, destination))  # Perform this step of the surrounding calculation or control-flow block.
    ordered_edges = sorted(edges)  # Construct or transform graph topology, geometry, or molecular feature data.
    graph = dgl.graph(  # Construct or transform graph topology, geometry, or molecular feature data.
        (  # Continue or close the surrounding multiline expression or collection.
            [edge[0] for edge in ordered_edges],  # Continue or close the surrounding multiline expression or collection.
            [edge[1] for edge in ordered_edges],  # Continue or close the surrounding multiline expression or collection.
        ),  # Continue or close the surrounding multiline expression or collection.
        num_nodes=atom_count,  # Construct or transform graph topology, geometry, or molecular feature data.
    )  # Continue or close the surrounding multiline expression or collection.
    graph.ndata["atom_features"] = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()],  # Continue or close the surrounding multiline expression or collection.
        dtype=torch.float32,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    graph.ndata["coordinates"] = torch.from_numpy(coordinates.copy())
    graph.edata["bond_features"] = torch.tensor(
        [  # Continue or close the surrounding multiline expression or collection.
            bond_lookup.get(edge, [0.0] * BOND_FEATURE_DIM)  # Perform this step of the surrounding calculation or control-flow block.
            for edge in ordered_edges  # Iterate over the stated records, layers, batches, or graph elements.
        ],  # Continue or close the surrounding multiline expression or collection.
        dtype=torch.float32,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    graph.edata["distance"] = torch.tensor(
        [[float(distances[source, destination])]  # Continue or close the surrounding multiline expression or collection.
         for source, destination in ordered_edges],  # Iterate over the stated records, layers, batches, or graph elements.
        dtype=torch.float32,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    graph.edata["r"] = torch.from_numpy(
        np.asarray(  # Invoke the relevant tensor, graph, numerical, or tabular operation.
            [  # Continue or close the surrounding multiline expression or collection.
                coordinates[destination] - coordinates[source]  # Perform this step of the surrounding calculation or control-flow block.
                for source, destination in ordered_edges  # Iterate over the stated records, layers, batches, or graph elements.
            ],  # Continue or close the surrounding multiline expression or collection.
            dtype=np.float32,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
    )  # Continue or close the surrounding multiline expression or collection.
    return graph  # Return this computed tensor, metric, object, or collection to the caller.


# CLASS: OpenBindGraphDataset — reusable model/data abstraction.
class OpenBindGraphDataset(Dataset):  # Define this reusable class and its inheritance contract.
    """Load the curated OpenBind cohort as 2D or crystal-pose 3D graphs."""

    def __init__(  # Define this callable; its indented block implements the documented operation.
        self,  # Perform this step of the surrounding calculation or control-flow block.
        data_root: str | Path = DEFAULT_DATA_ROOT,  # Bind this name to an intermediate value, configuration setting, or result.
        mode: str = "2d_gnn",
        spatial_cutoff: float = 5.0,  # Bind this name to an intermediate value, configuration setting, or result.
        max_neighbors: int = 32,  # Bind this name to an intermediate value, configuration setting, or result.
    ) -> None:  # Continue or close the surrounding multiline expression or collection.
        """Initialize this object and its required state."""
        if mode not in {"2d_gnn", "3d_gnn", "3d_alignn"}:
            raise ValueError(f"Unsupported graph mode: {mode}")  # Reject invalid input or state with an explicit exception.
        self.data_root = Path(data_root).resolve()  # Store this configuration value or neural-network submodule on the instance.
        curated_path = (  # Bind this name to an intermediate value, configuration setting, or result.
            self.data_root  # Perform this step of the surrounding calculation or control-flow block.
            / "curated"
            / "openbind_ligand_structures.csv"
        )  # Continue or close the surrounding multiline expression or collection.
        with curated_path.open(  # Enter a managed context so resources and graph state are cleaned up safely.
            "r", encoding="utf-8", newline=""
        ) as handle:  # Continue or close the surrounding multiline expression or collection.
            self.rows = list(csv.DictReader(handle))  # Store this configuration value or neural-network submodule on the instance.
        self.sample_ids = [row["complex_name"] for row in self.rows]
        if len(self.sample_ids) != len(set(self.sample_ids)):  # Evaluate this condition before executing the associated branch.
            raise ValueError("Duplicate complex names in curated dataset")
        self.mode = mode  # Store this configuration value or neural-network submodule on the instance.
        self.spatial_cutoff = spatial_cutoff  # Store this configuration value or neural-network submodule on the instance.
        self.max_neighbors = max_neighbors  # Store this configuration value or neural-network submodule on the instance.

    def __len__(self) -> int:  # Define this callable; its indented block implements the documented operation.
        """Return the number of dataset records."""
        return len(self.rows)  # Return this computed tensor, metric, object, or collection to the caller.

    def __getitem__(self, index: int):  # Define this callable; its indented block implements the documented operation.
        """Load and return one indexed dataset record."""
        row = self.rows[index]  # Bind this name to an intermediate value, configuration setting, or result.
        source_path = SOURCE_DATASET_ROOT / row["source_ligand_ref_sdf"]
        molecule = read_reference_molecule(source_path)  # Bind this name to an intermediate value, configuration setting, or result.
        graph = (  # Construct or transform graph topology, geometry, or molecular feature data.
            molecule_to_2d_graph(molecule)  # Perform this step of the surrounding calculation or control-flow block.
            if self.mode == "2d_gnn"
            else molecule_to_3d_graph(  # Handle the remaining case not covered by earlier conditions.
                molecule, self.spatial_cutoff, self.max_neighbors  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
        target = torch.tensor(  # Bind this name to an intermediate value, configuration setting, or result.
            float(row["experimental_pKD"]), dtype=torch.float32
        )  # Continue or close the surrounding multiline expression or collection.
        return graph, target, row["complex_name"]


# FUNCTION: validate_feature_dimensions — see its docstring and inline comments.
def validate_feature_dimensions() -> tuple[int, int]:  # Define this callable; its indented block implements the documented operation.
    """Expose and validate the declared feature dimensions."""
    if ATOM_FEATURE_DIM != 152 or BOND_FEATURE_DIM != 12:  # Evaluate this condition before executing the associated branch.
        raise RuntimeError("Unexpected molecular feature dimensions")
    return ATOM_FEATURE_DIM, BOND_FEATURE_DIM  # Return this computed tensor, metric, object, or collection to the caller.
