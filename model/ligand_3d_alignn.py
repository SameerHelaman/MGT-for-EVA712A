"""Angular ligand model using the original MGT ALIGNN calculations."""

from __future__ import annotations

import dgl
import torch
from dgl.nn.pytorch import AvgPooling
from torch import nn

from model.alignn import ALIGNNLayer
from modules.modules import MLPLayer, RBFExpansion
from utils.datasets import compute_bond_cosines


class Ligand3DALIGNN(nn.Module):
    """Add original MGT line-graph angle updates to the matched 3D GNN."""

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        num_layers: int = 3,
        distance_bins: int = 40,
        angle_bins: int = 40,
        spatial_cutoff: float = 5.0,
    ) -> None:
        """Construct the angular member of the controlled architecture series."""
        super().__init__()
        # Keep the atom representation identical to the 3D GNN.
        self.atom_embedding = MLPLayer(atom_feature_dim, hidden_dim)
        # Reuse the original MGT radial basis implementation for distances.
        self.distance_expansion = RBFExpansion(
            vmin=0.0,
            vmax=spatial_cutoff,
            bins=distance_bins,
        )
        # Keep chemical-bond and distance embedding identical to the 3D GNN.
        self.edge_embedding = nn.Sequential(
            MLPLayer(bond_feature_dim + distance_bins, embedding_dim),
            MLPLayer(embedding_dim, hidden_dim),
        )
        # Preserve the original MGT cosine range and angular RBF calculation.
        self.angle_expansion = RBFExpansion(
            vmin=-1.0,
            vmax=1.0,
            bins=angle_bins,
        )
        # Preserve the original MGT two-stage angular embedding pattern.
        self.angle_embedding = nn.Sequential(
            MLPLayer(angle_bins, embedding_dim),
            MLPLayer(embedding_dim, hidden_dim),
        )
        # Use the original MGT line-graph then atom-graph update without changes.
        self.alignn_layers = nn.ModuleList(
            ALIGNNLayer(hidden_dim) for _ in range(num_layers)
        )
        # Match the previous ligand models' graph-level pooling and regression.
        self.pooling = AvgPooling()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Calculate original MGT angles and predict one scaled pKD value."""
        # Construct the original DGL line graph with local edges as its nodes.
        line_graph = graph.line_graph(shared=True)
        # Run the original MGT bond-angle cosine function unchanged.
        line_graph.apply_edges(compute_bond_cosines)
        # Embed atom chemistry exactly as in the matched 3D GNN.
        atom_attributes = self.atom_embedding(graph.ndata["atom_features"])
        # Expand and combine the same distance and bond-chemistry inputs.
        distance_attributes = self.distance_expansion(
            graph.edata["distance"].squeeze(1)
        )
        edge_attributes = self.edge_embedding(
            torch.cat(
                [graph.edata["bond_features"], distance_attributes],
                dim=1,
            )
        )
        # Match Graphformer: expand the cosine tensor then remove its size-1 axis.
        angle_attributes = torch.squeeze(
            self.angle_expansion(line_graph.edata["angle_feats"]),
            dim=1,
        )
        angle_attributes = self.angle_embedding(angle_attributes)
        # Apply the original angle-edge update followed by the atom-edge update.
        for alignn_layer in self.alignn_layers:
            atom_attributes, edge_attributes, angle_attributes = alignn_layer(
                graph,
                line_graph,
                atom_attributes,
                edge_attributes,
                angle_attributes,
            )
        # Average updated atoms and predict ligand affinity.
        ligand_attributes = self.pooling(graph, atom_attributes)
        return self.output(ligand_attributes).squeeze(1)
