# =============================================================================
# MODULE: model/transformer.py
# PURPOSE: Implements original wider/Coulomb-graph multi-head attention.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: Projects queries, keys and values per head; computes edge-aware attention logits; applies edge softmax, evaluation-safe dropout, weighted message aggregation and head reduction.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Original wider-graph multi-head attention used by MGT Graphformer."""

import math  # Load a standard-library, scientific, or local project dependency.
from typing import Union, Tuple, Optional  # Import selected classes or functions from the named dependency.

import dgl  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
import dgl.function as fn  # Load a standard-library, scientific, or local project dependency.
import torch.nn.functional as F  # Load a standard-library, scientific, or local project dependency.
from torch import nn  # Import selected classes or functions from the named dependency.
from dgl import softmax_edges  # Import selected classes or functions from the named dependency.
from dgl.nn.functional import edge_softmax  # Import selected classes or functions from the named dependency.

# CLASS: multiheaded — reusable model/data abstraction.
class multiheaded(nn.Module):  # Define this reusable class and its inheritance contract.
    """Perform original multi-head attention over the wider Coulomb graph."""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,   # Define this callable; its indented block implements the documented operation.
                 concat: bool = True, dropout: float = 0.2, bias: bool = True,   # Bind this name to an intermediate value, configuration setting, or result.
                 residual: bool = True, norm: Union[bool, Tuple[bool, bool]] = True):  # Compute or store a loss, error, residual, or regression evaluation statistic.
        """Initialize this object and its required state."""
        super(multiheaded, self).__init__()  # Initialize or delegate to the parent class implementation.

        if isinstance(norm, bool):  # Evaluate this condition before executing the associated branch.
            norm = (norm, norm)  # Bind this name to an intermediate value, configuration setting, or result.

        self.in_channels = in_channels  # Store this configuration value or neural-network submodule on the instance.
        self.out_channels = out_channels  # Store this configuration value or neural-network submodule on the instance.
        # Store the number of attention heads and per-head output width.
        self.heads = heads  # Store this configuration value or neural-network submodule on the instance.
        self.concat = concat  # Store this configuration value or neural-network submodule on the instance.
        self.dropout = dropout  # Store this configuration value or neural-network submodule on the instance.
        self.residual = residual  # Store this configuration value or neural-network submodule on the instance.
        self.norm = norm  # Store this configuration value or neural-network submodule on the instance.

        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)  # Store this configuration value or neural-network submodule on the instance.
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)  # Store this configuration value or neural-network submodule on the instance.
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)  # Store this configuration value or neural-network submodule on the instance.

        self.lin_edge = nn.Linear(in_channels, heads * out_channels, bias=bias)  # Store this configuration value or neural-network submodule on the instance.

        if self.concat:  # Evaluate this condition before executing the associated branch.
            if self.norm[0]:  # Evaluate this condition before executing the associated branch.
                self.norm_node = nn.LayerNorm(heads * out_channels)  # Store this configuration value or neural-network submodule on the instance.
            if self.norm[1]:  # Evaluate this condition before executing the associated branch.
                self.norm_edge = nn.LayerNorm(heads * out_channels)  # Store this configuration value or neural-network submodule on the instance.
        else:  # Handle the remaining case not covered by earlier conditions.
            if self.norm[0]:  # Evaluate this condition before executing the associated branch.
                self.norm_node = nn.LayerNorm(out_channels)  # Store this configuration value or neural-network submodule on the instance.
            if self.norm[1]:  # Evaluate this condition before executing the associated branch.
                self.norm_edge = nn.LayerNorm(out_channels)  # Store this configuration value or neural-network submodule on the instance.

        self.reset_parameters()  # Perform this step of the surrounding calculation or control-flow block.

    def reset_parameters(self):  # Define this callable; its indented block implements the documented operation.
        """Initialize attention projection weights with Xavier initialization."""
        self.lin_query.reset_parameters()  # Perform this step of the surrounding calculation or control-flow block.
        self.lin_key.reset_parameters()  # Perform this step of the surrounding calculation or control-flow block.
        self.lin_value.reset_parameters()  # Perform this step of the surrounding calculation or control-flow block.
        if self.lin_edge is not None:  # Evaluate this condition before executing the associated branch.
            self.lin_edge.reset_parameters()  # Perform this step of the surrounding calculation or control-flow block.

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):  # Define this callable; its indented block implements the documented operation.
        r"""
        Args:
            g: dgl.DGLGraph
                The graph
            node_feats: torch.Tensor
                The node features
            edge_feats: torch.Tensor
                The edge features
            return_attention (bool, optional):
                If set to :obj:`True`, will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        # Confine temporary query/key/value/message fields to this forward call.
        g = g.local_var()  # Bind this name to an intermediate value, configuration setting, or result.

        g.ndata['query'] = self.lin_query(node_feats).view(-1, self.heads, self.out_channels)
        g.ndata['key'] = self.lin_key(node_feats).view(-1, self.heads, self.out_channels)
        g.ndata['value'] = self.lin_value(node_feats).view(-1, self.heads, self.out_channels)
        g.apply_edges(fn.u_mul_v('query', 'key', 'scores'))
        m = g.edata.pop('scores') + self.lin_edge(edge_feats).view(-1, self.heads, self.out_channels)
        
        # Scale logits by sqrt(head width) to control softmax sharpness.
        scores = m / math.sqrt(self.out_channels)  # Bind this name to an intermediate value, configuration setting, or result.
        # Normalize attention over incoming wider-graph edges and regularize only in training.
        g.edata['alpha'] = F.dropout(
            edge_softmax(g, scores).sum(dim=-1).unsqueeze(dim=-1),  # Construct or transform graph topology, geometry, or molecular feature data.
            p=self.dropout,  # Bind this name to an intermediate value, configuration setting, or result.
            training=self.training,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        # Weight source values by attention and sum them at destination atoms.
        g.update_all(fn.u_mul_e('value', 'alpha', 'm'), fn.sum('m', 'h'))
        x = g.ndata.pop('h')

        if self.concat:  # Evaluate this condition before executing the associated branch.
            x = x.view(-1, self.heads * self.out_channels)  # Bind this name to an intermediate value, configuration setting, or result.
            m = m.view(-1, self.heads * self.out_channels)  # Bind this name to an intermediate value, configuration setting, or result.
        else:  # Handle the remaining case not covered by earlier conditions.
            x = x.mean(dim=1)  # Bind this name to an intermediate value, configuration setting, or result.
            m = m.mean(dim=1)  # Bind this name to an intermediate value, configuration setting, or result.

        if self.norm[0]:  # Evaluate this condition before executing the associated branch.
            x = self.norm_node(x)  # Construct or transform graph topology, geometry, or molecular feature data.
        if self.norm[1]:  # Evaluate this condition before executing the associated branch.
            m = self.norm_edge(m)  # Construct or transform graph topology, geometry, or molecular feature data.

        if self.residual:  # Evaluate this condition before executing the associated branch.
            x += node_feats  # Construct or transform graph topology, geometry, or molecular feature data.
            y = m + edge_feats  # Construct or transform graph topology, geometry, or molecular feature data.

        return x, y  # Return this computed tensor, metric, object, or collection to the caller.
