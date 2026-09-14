# =============================================================================
# MODULE: model/ligand_gnn.py
# PURPOSE: Controlled 2D atom–bond message-passing baseline.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Chemistry-only edge-gated GNN for ligand pKD regression."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

# Provide DGL graph typing and pooling.
import dgl  # Load a standard-library, scientific, or local project dependency.
# Provide graph-average pooling to create one ligand representation.
from dgl.nn.pytorch import AvgPooling  # Import selected classes or functions from the named dependency.
# Build neural-network layers.
import torch  # Load a standard-library, scientific, or local project dependency.
from torch import nn  # Import selected classes or functions from the named dependency.

# Reuse the original MGT edge-gated graph convolution.
from model.alignn import EdgeGatedGraphConv  # Import selected classes or functions from the named dependency.
# Reuse the original MGT normalized SiLU projection block.
from modules.modules import MLPLayer  # Import selected classes or functions from the named dependency.


# CLASS: Ligand2DGNN — reusable model/data abstraction.
class Ligand2DGNN(nn.Module):  # Define this reusable class and its inheritance contract.
    """Predict affinity from atoms and chemical bonds without geometric inputs."""

    def __init__(  # Define this callable; its indented block implements the documented operation.
        self,  # Perform this step of the surrounding calculation or control-flow block.
        atom_feature_dim: int,  # Perform this step of the surrounding calculation or control-flow block.
        bond_feature_dim: int,  # Perform this step of the surrounding calculation or control-flow block.
        hidden_dim: int = 512,  # Bind this name to an intermediate value, configuration setting, or result.
        embedding_dim: int = 128,  # Create or apply a trainable neural-network component.
        num_layers: int = 3,  # Create or apply a trainable neural-network component.
    ) -> None:  # Continue or close the surrounding multiline expression or collection.
        """Construct the chemistry-only member of the architecture ablation."""
        # Initialize the base neural-network module.
        super().__init__()  # Initialize or delegate to the parent class implementation.
        # Embed raw RDKit atom features into the common hidden dimension.
        self.atom_embedding = MLPLayer(atom_feature_dim, hidden_dim)  # Store this configuration value or neural-network submodule on the instance.
        # Embed raw bond chemistry through the original two-stage MGT pattern.
        self.bond_embedding = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            MLPLayer(bond_feature_dim, embedding_dim),  # Perform this step of the surrounding calculation or control-flow block.
            MLPLayer(embedding_dim, hidden_dim),  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        # Apply only local edge-gated atom/bond convolutions.
        self.convolutions = nn.ModuleList(  # Store this configuration value or neural-network submodule on the instance.
            EdgeGatedGraphConv(hidden_dim)  # Perform this step of the surrounding calculation or control-flow block.
            for _ in range(num_layers)  # Iterate over the stated records, layers, batches, or graph elements.
        )  # Continue or close the surrounding multiline expression or collection.
        # Average atom representations into one ligand vector as in Graphformer.
        self.pooling = AvgPooling()  # Store this configuration value or neural-network submodule on the instance.
        # Predict one scaled pKD value.
        self.output = nn.Linear(hidden_dim, 1)  # Store this configuration value or neural-network submodule on the instance.

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:  # Define this callable; its indented block implements the documented operation.
        """Run chemistry-only message passing and graph-level regression."""
        # Read but do not remove the graph's raw atom features.
        atom_attributes = graph.ndata["atom_features"]
        # Read but do not remove the graph's raw chemical bond features.
        bond_attributes = graph.edata["bond_features"]
        # Project atoms into the shared hidden representation.
        atom_attributes = self.atom_embedding(atom_attributes)  # Create or apply a trainable neural-network component.
        # Project bonds into the same hidden representation.
        bond_attributes = self.bond_embedding(bond_attributes)  # Construct or transform graph topology, geometry, or molecular feature data.
        # Update atom and bond states through local chemical connectivity only.
        for convolution in self.convolutions:  # Iterate over the stated records, layers, batches, or graph elements.
            atom_attributes, bond_attributes = convolution(  # Construct or transform graph topology, geometry, or molecular feature data.
                graph,  # Perform this step of the surrounding calculation or control-flow block.
                atom_attributes,  # Perform this step of the surrounding calculation or control-flow block.
                bond_attributes,  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
        # Average atom embeddings separately for each batched ligand.
        ligand_attributes = self.pooling(graph, atom_attributes)  # Construct or transform graph topology, geometry, or molecular feature data.
        # Produce one regression output per ligand.
        return self.output(ligand_attributes).squeeze(1)  # Return this computed tensor, metric, object, or collection to the caller.
