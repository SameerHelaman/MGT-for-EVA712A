# =============================================================================
# MODULE: model/ligand_3d_gnn.py
# PURPOSE: Controlled ligand model using atom chemistry and distance-RBF spatial edges without angular processing.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Distance-aware crystallographic 3D GNN without angular/Coulomb processing."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

# DGL supplies graph typing and graph-level average pooling.
import dgl  # Load a standard-library, scientific, or local project dependency.
from dgl.nn.pytorch import AvgPooling  # Import selected classes or functions from the named dependency.
# Torch supplies trainable layers and tensor concatenation.
import torch  # Load a standard-library, scientific, or local project dependency.
from torch import nn  # Import selected classes or functions from the named dependency.

# Reuse the original MGT local atom–edge update without changing its equations.
from model.alignn import EdgeGatedGraphConv  # Import selected classes or functions from the named dependency.
# Reuse the original MGT projection and radial-basis implementations.
from modules.modules import MLPLayer, RBFExpansion  # Import selected classes or functions from the named dependency.


# CLASS: Ligand3DGNN — reusable model/data abstraction.
class Ligand3DGNN(nn.Module):  # Define this reusable class and its inheritance contract.
    """Predict pKD from RDKit chemistry and crystallographic pair distances."""

    def __init__(  # Define this callable; its indented block implements the documented operation.
        self,  # Perform this step of the surrounding calculation or control-flow block.
        atom_feature_dim: int,  # Perform this step of the surrounding calculation or control-flow block.
        bond_feature_dim: int,  # Perform this step of the surrounding calculation or control-flow block.
        hidden_dim: int = 512,  # Bind this name to an intermediate value, configuration setting, or result.
        embedding_dim: int = 128,  # Create or apply a trainable neural-network component.
        num_layers: int = 3,  # Create or apply a trainable neural-network component.
        distance_bins: int = 40,  # Construct or transform graph topology, geometry, or molecular feature data.
        spatial_cutoff: float = 5.0,  # Bind this name to an intermediate value, configuration setting, or result.
    ) -> None:  # Continue or close the surrounding multiline expression or collection.
        """Construct the distance member of the controlled model ablation."""
        # Register this class as a Torch module so parameters are trainable/saved.
        super().__init__()  # Initialize or delegate to the parent class implementation.
        # Project each 152-dimensional raw atom vector to the hidden dimension.
        self.atom_embedding = MLPLayer(atom_feature_dim, hidden_dim)  # Store this configuration value or neural-network submodule on the instance.
        # Convert each scalar distance into a smooth Gaussian RBF vector.
        self.distance_expansion = RBFExpansion(  # Store this configuration value or neural-network submodule on the instance.
            vmin=0.0,  # Bind this name to an intermediate value, configuration setting, or result.
            vmax=spatial_cutoff,  # Bind this name to an intermediate value, configuration setting, or result.
            bins=distance_bins,  # Construct or transform graph topology, geometry, or molecular feature data.
        )  # Continue or close the surrounding multiline expression or collection.
        # Join bond chemistry and distance RBFs, then project them to hidden_dim.
        self.edge_embedding = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            MLPLayer(bond_feature_dim + distance_bins, embedding_dim),  # Perform this step of the surrounding calculation or control-flow block.
            MLPLayer(embedding_dim, hidden_dim),  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        # Stack the requested number of original edge-gated local graph updates.
        self.convolutions = nn.ModuleList(  # Store this configuration value or neural-network submodule on the instance.
            EdgeGatedGraphConv(hidden_dim) for _ in range(num_layers)  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        # Average atom states independently for every ligand in a DGL batch.
        self.pooling = AvgPooling()  # Store this configuration value or neural-network submodule on the instance.
        # Map each pooled ligand representation to one normalized pKD prediction.
        self.output = nn.Linear(hidden_dim, 1)  # Store this configuration value or neural-network submodule on the instance.

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:  # Define this callable; its indented block implements the documented operation.
        """Embed one graph batch, update it by distance, and predict pKD."""
        # Embed the fixed RDKit atom descriptors stored by the dataset.
        atom_attributes = self.atom_embedding(  # Create or apply a trainable neural-network component.
            graph.ndata["atom_features"]
        )  # Continue or close the surrounding multiline expression or collection.
        # Remove the singleton column from the E×1 distance tensor.
        distances = graph.edata["distance"].squeeze(1)
        # Smoothly expand distances so nearby values have similar encodings.
        distance_attributes = self.distance_expansion(distances)  # Construct or transform graph topology, geometry, or molecular feature data.
        # Concatenate 12 bond-chemistry values with 40 distance RBF values.
        edge_attributes = torch.cat(  # Construct or transform graph topology, geometry, or molecular feature data.
            [graph.edata["bond_features"], distance_attributes],
            dim=1,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        # Project the combined raw edge vector into the shared hidden space.
        edge_attributes = self.edge_embedding(edge_attributes)  # Construct or transform graph topology, geometry, or molecular feature data.
        # Repeatedly update atom and edge states on the same local/spatial graph.
        for convolution in self.convolutions:  # Iterate over the stated records, layers, batches, or graph elements.
            atom_attributes, edge_attributes = convolution(  # Construct or transform graph topology, geometry, or molecular feature data.
                graph,  # Perform this step of the surrounding calculation or control-flow block.
                atom_attributes,  # Perform this step of the surrounding calculation or control-flow block.
                edge_attributes,  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
        # Collapse variable atom counts into one fixed-size ligand vector.
        ligand_attributes = self.pooling(graph, atom_attributes)  # Construct or transform graph topology, geometry, or molecular feature data.
        # Return a flat tensor containing one scalar for every ligand.
        return self.output(ligand_attributes).squeeze(1)  # Return this computed tensor, metric, object, or collection to the caller.
