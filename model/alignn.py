# =============================================================================
# MODULE: model/alignn.py
# PURPOSE: Implements the original edge-gated graph convolution and ALIGNN angle–edge–atom update.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: Edge gates are sigmoid(edge features); gated neighbour messages are normalized by summed gates. ALIGNN alternates line-graph and atom-graph gated updates.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Original MGT edge-gated graph convolution and ALIGNN update layers."""

from typing import Tuple, Union  # Import selected classes or functions from the named dependency.
import dgl  # Load a standard-library, scientific, or local project dependency.
import numpy  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
import dgl.function as fn  # Load a standard-library, scientific, or local project dependency.
from torch import nn, Tensor  # Import selected classes or functions from the named dependency.
from torch.nn import functional as F  # Import selected classes or functions from the named dependency.


# CLASS: EdgeGatedGraphConv — reusable model/data abstraction.
class EdgeGatedGraphConv(nn.Module):  # Define this reusable class and its inheritance contract.
    """Edge gated graph convolution from arxiv:1711.07553.
    see also arxiv:2003.0098.
    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(self, feature_dims: int, norm: Union[bool, Tuple[bool, bool]] = True, residual: bool = True):  # Define this callable; its indented block implements the documented operation.
        """Initialize parameters for ALIGNN update."""
        super().__init__()  # Initialize or delegate to the parent class implementation.
        self.residual = residual  # Store this configuration value or neural-network submodule on the instance.

        if isinstance(norm, bool):  # Evaluate this condition before executing the associated branch.
            norm = (norm, norm)  # Bind this name to an intermediate value, configuration setting, or result.
        self.norm = norm  # Store this configuration value or neural-network submodule on the instance.

        self.src_gate = nn.Linear(feature_dims, feature_dims)  # Store this configuration value or neural-network submodule on the instance.
        self.dst_gate = nn.Linear(feature_dims, feature_dims)  # Store this configuration value or neural-network submodule on the instance.
        self.edge_gate = nn.Linear(feature_dims, feature_dims)  # Store this configuration value or neural-network submodule on the instance.
        if norm[0]:  # Evaluate this condition before executing the associated branch.
            self.norm_nodes = nn.LayerNorm(feature_dims)  # Store this configuration value or neural-network submodule on the instance.

        self.src_update = nn.Linear(feature_dims, feature_dims)  # Store this configuration value or neural-network submodule on the instance.
        self.dst_update = nn.Linear(feature_dims, feature_dims)  # Store this configuration value or neural-network submodule on the instance.
        if norm[1]:  # Evaluate this condition before executing the associated branch.
            self.norm_edges = nn.LayerNorm(feature_dims)  # Store this configuration value or neural-network submodule on the instance.

    def forward(self, g: dgl.DGLGraph, node_feats: Tensor, edge_feats: Tensor) -> torch.Tensor:  # Define this callable; its indented block implements the documented operation.
        """Edge-gated graph convolution.
        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        # Isolate temporary DGL fields so the caller graph is not permanently changed.
        g = g.local_var()  # Bind this name to an intermediate value, configuration setting, or result.

        # Project source and destination node states for the learned edge gate.
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        # Convert each edge channel to a soft gate in the interval (0, 1).
        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(  # Perform this step of the surrounding calculation or control-flow block.
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )  # Continue or close the surrounding multiline expression or collection.
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        # Normalize gated neighbour messages by total incoming gate strength.
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        if self.norm[0]:  # Evaluate this condition before executing the associated branch.
            x = self.norm_nodes(x)  # Construct or transform graph topology, geometry, or molecular feature data.
            x = F.silu(x)  # Bind this name to an intermediate value, configuration setting, or result.
        else:  # Handle the remaining case not covered by earlier conditions.
            x = F.silu(x)  # Bind this name to an intermediate value, configuration setting, or result.
        if self.norm[1]:  # Evaluate this condition before executing the associated branch.
            m = self.norm_edges(m)  # Construct or transform graph topology, geometry, or molecular feature data.
            y = F.silu(m)  # Bind this name to an intermediate value, configuration setting, or result.
        else:  # Handle the remaining case not covered by earlier conditions.
            y = F.silu(m)  # Bind this name to an intermediate value, configuration setting, or result.

        # Retain previous node/edge states to stabilize deep message passing.
        if self.residual:  # Evaluate this condition before executing the associated branch.
            x = node_feats + x  # Construct or transform graph topology, geometry, or molecular feature data.
            y = edge_feats + y  # Construct or transform graph topology, geometry, or molecular feature data.

        return x, y  # Return this computed tensor, metric, object, or collection to the caller.


# CLASS: ALIGNNLayer — reusable model/data abstraction.
class ALIGNNLayer(nn.Module):  # Define this reusable class and its inheritance contract.
    """Compose a line-graph edge update with an atom-graph update."""
    def __init__(self, feature_dims: int, edge_norm: Union[bool, Tuple[bool, bool]] = True, node_norm: Union[bool, Tuple[bool, bool]] = True):  # Define this callable; its indented block implements the documented operation.
        """Initialize this object and its required state."""
        super(ALIGNNLayer, self).__init__()  # Initialize or delegate to the parent class implementation.
        if isinstance(edge_norm, bool):  # Evaluate this condition before executing the associated branch.
            edge_norm = (edge_norm, edge_norm)  # Construct or transform graph topology, geometry, or molecular feature data.
        if isinstance(node_norm, bool):  # Evaluate this condition before executing the associated branch.
            node_norm = (node_norm, node_norm)  # Construct or transform graph topology, geometry, or molecular feature data.
        # First update bonds-as-nodes using angles-as-edges on the line graph.
        self.edge_update = EdgeGatedGraphConv(feature_dims=feature_dims, norm=edge_norm)  # Store this configuration value or neural-network submodule on the instance.
        # Then update atoms-as-nodes using the updated bonds on the atom graph.
        self.atom_update = EdgeGatedGraphConv(feature_dims=feature_dims, norm=node_norm)  # Store this configuration value or neural-network submodule on the instance.

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, x: Tensor, y: Tensor, z: Tensor):  # Define this callable; its indented block implements the documented operation.
        # Convolution on line graph
        """Apply this module to its input tensors or graphs."""
        y, z = self.edge_update(g=lg, node_feats=y, edge_feats=z)  # Construct or transform graph topology, geometry, or molecular feature data.
        # Convolution on atomistic graph
        x, y = self.atom_update(g=g, node_feats=x, edge_feats=y)  # Construct or transform graph topology, geometry, or molecular feature data.

        return x, y, z  # Return this computed tensor, metric, object, or collection to the caller.
