"""Distance-aware crystallographic 3D GNN without angular/Coulomb processing."""

from __future__ import annotations

# DGL supplies graph typing and graph-level average pooling.
import dgl
from dgl.nn.pytorch import AvgPooling
# Torch supplies trainable layers and tensor concatenation.
import torch
from torch import nn

# Reuse the original MGT local atom–edge update without changing its equations.
from model.alignn import EdgeGatedGraphConv
# Reuse the original MGT projection and radial-basis implementations.
from modules.modules import MLPLayer, RBFExpansion


class Ligand3DGNN(nn.Module):
    """Predict pKD from RDKit chemistry and crystallographic pair distances."""

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        num_layers: int = 3,
        distance_bins: int = 40,
        spatial_cutoff: float = 5.0,
    ) -> None:
        """Construct the distance member of the controlled model ablation."""
        # Register this class as a Torch module so parameters are trainable/saved.
        super().__init__()
        # Project each 152-dimensional raw atom vector to the hidden dimension.
        self.atom_embedding = MLPLayer(atom_feature_dim, hidden_dim)
        # Convert each scalar distance into a smooth Gaussian RBF vector.
        self.distance_expansion = RBFExpansion(
            vmin=0.0,
            vmax=spatial_cutoff,
            bins=distance_bins,
        )
        # Join bond chemistry and distance RBFs, then project them to hidden_dim.
        self.edge_embedding = nn.Sequential(
            MLPLayer(bond_feature_dim + distance_bins, embedding_dim),
            MLPLayer(embedding_dim, hidden_dim),
        )
        # Stack the requested number of original edge-gated local graph updates.
        self.convolutions = nn.ModuleList(
            EdgeGatedGraphConv(hidden_dim) for _ in range(num_layers)
        )
        # Average atom states independently for every ligand in a DGL batch.
        self.pooling = AvgPooling()
        # Map each pooled ligand representation to one normalized pKD prediction.
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Embed one graph batch, update it by distance, and predict pKD."""
        # Embed the fixed RDKit atom descriptors stored by the dataset.
        atom_attributes = self.atom_embedding(
            graph.ndata["atom_features"]
        )
        # Remove the singleton column from the E×1 distance tensor.
        distances = graph.edata["distance"].squeeze(1)
        # Smoothly expand distances so nearby values have similar encodings.
        distance_attributes = self.distance_expansion(distances)
        # Concatenate 12 bond-chemistry values with 40 distance RBF values.
        edge_attributes = torch.cat(
            [graph.edata["bond_features"], distance_attributes],
            dim=1,
        )
        # Project the combined raw edge vector into the shared hidden space.
        edge_attributes = self.edge_embedding(edge_attributes)
        # Repeatedly update atom and edge states on the same local/spatial graph.
        for convolution in self.convolutions:
            atom_attributes, edge_attributes = convolution(
                graph,
                atom_attributes,
                edge_attributes,
            )
        # Collapse variable atom counts into one fixed-size ligand vector.
        ligand_attributes = self.pooling(graph, atom_attributes)
        # Return a flat tensor containing one scalar for every ligand.
        return self.output(ligand_attributes).squeeze(1)
