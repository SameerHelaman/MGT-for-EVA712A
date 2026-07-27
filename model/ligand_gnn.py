"""Chemistry-only edge-gated GNN for ligand pKD regression."""

from __future__ import annotations

# Provide DGL graph typing and pooling.
import dgl
# Provide graph-average pooling to create one ligand representation.
from dgl.nn.pytorch import AvgPooling
# Build neural-network layers.
import torch
from torch import nn

# Reuse the original MGT edge-gated graph convolution.
from model.alignn import EdgeGatedGraphConv
# Reuse the original MGT normalized SiLU projection block.
from modules.modules import MLPLayer


class Ligand2DGNN(nn.Module):
    """Predict affinity from atoms and chemical bonds without geometric inputs."""

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        """Construct the chemistry-only member of the architecture ablation."""
        # Initialize the base neural-network module.
        super().__init__()
        # Embed raw RDKit atom features into the common hidden dimension.
        self.atom_embedding = MLPLayer(atom_feature_dim, hidden_dim)
        # Embed raw bond chemistry through the original two-stage MGT pattern.
        self.bond_embedding = nn.Sequential(
            MLPLayer(bond_feature_dim, embedding_dim),
            MLPLayer(embedding_dim, hidden_dim),
        )
        # Apply only local edge-gated atom/bond convolutions.
        self.convolutions = nn.ModuleList(
            EdgeGatedGraphConv(hidden_dim)
            for _ in range(num_layers)
        )
        # Average atom representations into one ligand vector as in Graphformer.
        self.pooling = AvgPooling()
        # Predict one scaled pKD value.
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Run chemistry-only message passing and graph-level regression."""
        # Read but do not remove the graph's raw atom features.
        atom_attributes = graph.ndata["atom_features"]
        # Read but do not remove the graph's raw chemical bond features.
        bond_attributes = graph.edata["bond_features"]
        # Project atoms into the shared hidden representation.
        atom_attributes = self.atom_embedding(atom_attributes)
        # Project bonds into the same hidden representation.
        bond_attributes = self.bond_embedding(bond_attributes)
        # Update atom and bond states through local chemical connectivity only.
        for convolution in self.convolutions:
            atom_attributes, bond_attributes = convolution(
                graph,
                atom_attributes,
                bond_attributes,
            )
        # Average atom embeddings separately for each batched ligand.
        ligand_attributes = self.pooling(graph, atom_attributes)
        # Produce one regression output per ligand.
        return self.output(ligand_attributes).squeeze(1)
