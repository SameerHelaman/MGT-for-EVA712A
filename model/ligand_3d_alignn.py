# =============================================================================
# MODULE: model/ligand_3d_alignn.py
# PURPOSE: Controlled ligand model using crystallographic distances and bond-angle line graphs.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Angular ligand model using the original MGT ALIGNN calculations."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import dgl  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
from dgl.nn.pytorch import AvgPooling  # Import selected classes or functions from the named dependency.
from torch import nn  # Import selected classes or functions from the named dependency.

from model.alignn import ALIGNNLayer  # Import selected classes or functions from the named dependency.
from modules.modules import MLPLayer, RBFExpansion  # Import selected classes or functions from the named dependency.
from utils.datasets import compute_bond_cosines  # Import selected classes or functions from the named dependency.


# CLASS: Ligand3DALIGNN — reusable model/data abstraction.
class Ligand3DALIGNN(nn.Module):  # Define this reusable class and its inheritance contract.
    """Add original MGT line-graph angle updates to the matched 3D GNN."""

    def __init__(  # Define this callable; its indented block implements the documented operation.
        self,  # Perform this step of the surrounding calculation or control-flow block.
        atom_feature_dim: int,  # Perform this step of the surrounding calculation or control-flow block.
        bond_feature_dim: int,  # Perform this step of the surrounding calculation or control-flow block.
        hidden_dim: int = 512,  # Bind this name to an intermediate value, configuration setting, or result.
        embedding_dim: int = 128,  # Create or apply a trainable neural-network component.
        num_layers: int = 3,  # Create or apply a trainable neural-network component.
        distance_bins: int = 40,  # Construct or transform graph topology, geometry, or molecular feature data.
        angle_bins: int = 40,  # Construct or transform graph topology, geometry, or molecular feature data.
        spatial_cutoff: float = 5.0,  # Bind this name to an intermediate value, configuration setting, or result.
    ) -> None:  # Continue or close the surrounding multiline expression or collection.
        """Construct the angular member of the controlled architecture series."""
        super().__init__()  # Initialize or delegate to the parent class implementation.
        # Keep the atom representation identical to the 3D GNN.
        self.atom_embedding = MLPLayer(atom_feature_dim, hidden_dim)  # Store this configuration value or neural-network submodule on the instance.
        # Reuse the original MGT radial basis implementation for distances.
        self.distance_expansion = RBFExpansion(  # Store this configuration value or neural-network submodule on the instance.
            vmin=0.0,  # Bind this name to an intermediate value, configuration setting, or result.
            vmax=spatial_cutoff,  # Bind this name to an intermediate value, configuration setting, or result.
            bins=distance_bins,  # Construct or transform graph topology, geometry, or molecular feature data.
        )  # Continue or close the surrounding multiline expression or collection.
        # Keep chemical-bond and distance embedding identical to the 3D GNN.
        self.edge_embedding = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            MLPLayer(bond_feature_dim + distance_bins, embedding_dim),  # Perform this step of the surrounding calculation or control-flow block.
            MLPLayer(embedding_dim, hidden_dim),  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        # Preserve the original MGT cosine range and angular RBF calculation.
        self.angle_expansion = RBFExpansion(  # Store this configuration value or neural-network submodule on the instance.
            vmin=-1.0,  # Bind this name to an intermediate value, configuration setting, or result.
            vmax=1.0,  # Bind this name to an intermediate value, configuration setting, or result.
            bins=angle_bins,  # Construct or transform graph topology, geometry, or molecular feature data.
        )  # Continue or close the surrounding multiline expression or collection.
        # Preserve the original MGT two-stage angular embedding pattern.
        self.angle_embedding = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            MLPLayer(angle_bins, embedding_dim),  # Perform this step of the surrounding calculation or control-flow block.
            MLPLayer(embedding_dim, hidden_dim),  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        # Use the original MGT line-graph then atom-graph update without changes.
        self.alignn_layers = nn.ModuleList(  # Store this configuration value or neural-network submodule on the instance.
            ALIGNNLayer(hidden_dim) for _ in range(num_layers)  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        # Match the previous ligand models' graph-level pooling and regression.
        self.pooling = AvgPooling()  # Store this configuration value or neural-network submodule on the instance.
        self.output = nn.Linear(hidden_dim, 1)  # Store this configuration value or neural-network submodule on the instance.

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:  # Define this callable; its indented block implements the documented operation.
        """Calculate original MGT angles and predict one scaled pKD value."""
        # Construct the original DGL line graph with local edges as its nodes.
        line_graph = graph.line_graph(shared=True)  # Construct or transform graph topology, geometry, or molecular feature data.
        # Run the original MGT bond-angle cosine function unchanged.
        line_graph.apply_edges(compute_bond_cosines)  # Perform this step of the surrounding calculation or control-flow block.
        # Embed atom chemistry exactly as in the matched 3D GNN.
        atom_attributes = self.atom_embedding(graph.ndata["atom_features"])
        # Expand and combine the same distance and bond-chemistry inputs.
        distance_attributes = self.distance_expansion(  # Construct or transform graph topology, geometry, or molecular feature data.
            graph.edata["distance"].squeeze(1)
        )  # Continue or close the surrounding multiline expression or collection.
        edge_attributes = self.edge_embedding(  # Construct or transform graph topology, geometry, or molecular feature data.
            torch.cat(  # Invoke the relevant tensor, graph, numerical, or tabular operation.
                [graph.edata["bond_features"], distance_attributes],
                dim=1,  # Bind this name to an intermediate value, configuration setting, or result.
            )  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
        # Match Graphformer: expand the cosine tensor then remove its size-1 axis.
        angle_attributes = torch.squeeze(  # Construct or transform graph topology, geometry, or molecular feature data.
            self.angle_expansion(line_graph.edata["angle_feats"]),
            dim=1,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        angle_attributes = self.angle_embedding(angle_attributes)  # Construct or transform graph topology, geometry, or molecular feature data.
        # Apply the original angle-edge update followed by the atom-edge update.
        for alignn_layer in self.alignn_layers:  # Iterate over the stated records, layers, batches, or graph elements.
            atom_attributes, edge_attributes, angle_attributes = alignn_layer(  # Construct or transform graph topology, geometry, or molecular feature data.
                graph,  # Perform this step of the surrounding calculation or control-flow block.
                line_graph,  # Perform this step of the surrounding calculation or control-flow block.
                atom_attributes,  # Perform this step of the surrounding calculation or control-flow block.
                edge_attributes,  # Perform this step of the surrounding calculation or control-flow block.
                angle_attributes,  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
        # Average updated atoms and predict ligand affinity.
        ligand_attributes = self.pooling(graph, atom_attributes)  # Construct or transform graph topology, geometry, or molecular feature data.
        return self.output(ligand_attributes).squeeze(1)  # Return this computed tensor, metric, object, or collection to the caller.
