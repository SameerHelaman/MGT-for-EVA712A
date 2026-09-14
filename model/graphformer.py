# =============================================================================
# MODULE: model/graphformer.py
# PURPOSE: Composes Coulomb attention, ALIGNN, local graph convolutions, feed-forward/residual blocks, pooling and regression.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Original complete Molecular Graph Transformer encoder and regressor."""

import dgl  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
from torch import nn, Tensor  # Import selected classes or functions from the named dependency.
from dgl.nn.pytorch import AvgPooling  # Import selected classes or functions from the named dependency.

import model.alignn as alignn, model.transformer as transformer  # Load a standard-library, scientific, or local project dependency.
import modules.modules as modules  # Load a standard-library, scientific, or local project dependency.


# CLASS: encoder — reusable model/data abstraction.
class encoder(nn.Module):  # Define this reusable class and its inheritance contract.
    """Apply Coulomb attention, ALIGNN, local GNN, FFN, normalization and residual updates."""
    def __init__(self, encoder_dims: int, n_heads: int = 4, n_mha: int = 1, n_alignn: int = 4, n_gnn: int = 4,  # Define this callable; its indented block implements the documented operation.
                 residual: bool = True, norm: bool = True, last: bool = False):  # Compute or store a loss, error, residual, or regression evaluation statistic.
        """Initialize this object and its required state."""
        super(encoder, self).__init__()  # Initialize or delegate to the parent class implementation.

        # Each attention head must receive an equal slice of the hidden vector.
        assert encoder_dims % n_heads == 0  # Enforce an invariant required by the following calculation.
        assert n_heads >= 1  # Enforce an invariant required by the following calculation.
        assert n_gnn >= 1  # Enforce an invariant required by the following calculation.

        # Store the per-head projection width used by multiheaded attention.
        self.head_dim = int(encoder_dims / n_heads)  # Store this configuration value or neural-network submodule on the instance.
        self.residual = residual  # Store this configuration value or neural-network submodule on the instance.
        self.norm = norm  # Store this configuration value or neural-network submodule on the instance.

        # The final encoder disables selected normalizations inside its last sublayers.
        if last:  # Evaluate this condition before executing the associated branch.
            # Multi-Headed Attention Layers
            mha_layers = [transformer.multiheaded(encoder_dims, self.head_dim, heads=n_heads) for _ in range(n_mha - 1)]  # Create or apply a trainable neural-network component.
            mha_layers.append(transformer.multiheaded(encoder_dims, self.head_dim, heads=n_heads, norm=(True, False)))  # Create or apply a trainable neural-network component.

            # ALIGNN Blocks
            alignns = [alignn.ALIGNNLayer(encoder_dims) for _ in range(n_alignn - 1)]  # Create or apply a trainable neural-network component.
            alignns.append(alignn.ALIGNNLayer(encoder_dims, edge_norm=(True, False)))  # Construct or transform graph topology, geometry, or molecular feature data.

            # Graph Convolution Layers
            eggcs = [alignn.EdgeGatedGraphConv(encoder_dims) for _ in range(n_gnn - 1)]  # Construct or transform graph topology, geometry, or molecular feature data.
            eggcs.append(alignn.EdgeGatedGraphConv(encoder_dims, norm=(True, False)))  # Construct or transform graph topology, geometry, or molecular feature data.
        else:  # Handle the remaining case not covered by earlier conditions.
            # Multi-Headed Attention Layers
            mha_layers = [transformer.multiheaded(encoder_dims, self.head_dim, heads=n_heads) for _ in range(n_mha)]  # Create or apply a trainable neural-network component.

            # ALIGNN Blocks
            alignns = [alignn.ALIGNNLayer(encoder_dims) for _ in range(n_alignn)]  # Create or apply a trainable neural-network component.

            # Graph Convolution Layers
            eggcs = [alignn.EdgeGatedGraphConv(encoder_dims) for _ in range(n_gnn)]  # Construct or transform graph topology, geometry, or molecular feature data.
        
        self.mha_layers = nn.ModuleList(mha_layers)  # Store this configuration value or neural-network submodule on the instance.
        self.alignn_layers = nn.ModuleList(alignns)  # Store this configuration value or neural-network submodule on the instance.
        self.eggc_layers = nn.ModuleList(eggcs)  # Store this configuration value or neural-network submodule on the instance.

        # Linear layers with ReLU inbetween
        # Refine every atom independently with the encoder feed-forward block.
        self.lin_block = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            nn.Linear(encoder_dims, encoder_dims),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
            nn.ReLU(inplace=True),  # Bind this name to an intermediate value, configuration setting, or result.
            nn.Linear(encoder_dims, encoder_dims)  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        )  # Continue or close the surrounding multiline expression or collection.

        # Normalize the feed-forward output when encoder normalization is enabled.
        if self.norm:  # Evaluate this condition before executing the associated branch.
            self.normalizer = nn.LayerNorm(encoder_dims)  # Store this configuration value or neural-network submodule on the instance.

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, fg: dgl.DGLGraph,  # Define this callable; its indented block implements the documented operation.
                x: Tensor, y: Tensor, z: Tensor, f: Tensor):  # Perform this step of the surrounding calculation or control-flow block.

        # Get attended atom attributes
        """Apply this module to its input tensors or graphs."""
        # Propagate long-range information over the wider Coulomb graph.
        for mha in self.mha_layers:  # Iterate over the stated records, layers, batches, or graph elements.
            x, f = mha(fg, x, f)  # Bind this name to an intermediate value, configuration setting, or result.

        # Perform graph convolutions on the attended atom attributes
        # Couple atom, local-edge and angle states through each ALIGNN block.
        for alignn_layer in self.alignn_layers:  # Iterate over the stated records, layers, batches, or graph elements.
            x, y, z = alignn_layer(g, lg, x, y, z)  # Create or apply a trainable neural-network component.

        # Refine atom and local-edge states after angular processing.
        for eggc_layer in self.eggc_layers:  # Iterate over the stated records, layers, batches, or graph elements.
            x, y = eggc_layer(g, x, y)  # Create or apply a trainable neural-network component.

        # Apply final two linear layers
        out = self.lin_block(x)  # Bind this name to an intermediate value, configuration setting, or result.

        if self.norm:  # Evaluate this condition before executing the associated branch.
            out = self.normalizer(out)  # Bind this name to an intermediate value, configuration setting, or result.

        # Preserve the encoder input through an additive residual connection.
        if self.residual:  # Evaluate this condition before executing the associated branch.
            out += x  # Bind this name to an intermediate value, configuration setting, or result.

        return out, y, z, f  # Return this computed tensor, metric, object, or collection to the caller.


# CLASS: Graphformer — reusable model/data abstraction.
class Graphformer(nn.Module):  # Define this reusable class and its inheritance contract.
    """Embed three graph views, run MGT encoders, pool atoms and predict properties."""
    def __init__(self, args):  # Define this callable; its indented block implements the documented operation.
        """Initialize this object and its required state."""
        super(Graphformer, self).__init__()  # Initialize or delegate to the parent class implementation.

        # Embed elemental descriptors and positional encodings into one hidden width.
        self.atom_embedding = modules.MLPLayer(args.num_atom_fea, args.hidden_dims)  # Store this configuration value or neural-network submodule on the instance.
        self.positional_embedding = modules.MLPLayer(args.num_pe_fea, args.hidden_dims)  # Store this configuration value or neural-network submodule on the instance.
        # Expand scalar local distances before learned edge projection.
        self.edge_expansion = modules.RBFExpansion(vmin=0, vmax=args.local_radius, bins=args.num_edge_bins)  # Store this configuration value or neural-network submodule on the instance.
        self.edge_embedding = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            modules.MLPLayer(args.num_edge_bins, args.embedding_dims),  # Perform this step of the surrounding calculation or control-flow block.
            modules.MLPLayer(args.embedding_dims, args.hidden_dims)  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        # Expand line-graph cosine values across the physical [-1, 1] range.
        self.angle_expansion = modules.RBFExpansion(vmin=-1, vmax=1, bins=args.num_angle_bins)  # Store this configuration value or neural-network submodule on the instance.
        self.angle_embedding = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            modules.MLPLayer(args.num_angle_bins, args.embedding_dims),  # Perform this step of the surrounding calculation or control-flow block.
            modules.MLPLayer(args.embedding_dims, args.hidden_dims)  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        # Embed long-range ZiZj/rij Coulomb edge values for attention.
        self.fc_embedding = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            modules.MLPLayer(1, args.num_clmb_bins),  # Perform this step of the surrounding calculation or control-flow block.
            modules.MLPLayer(args.num_clmb_bins, args.embedding_dims),  # Perform this step of the surrounding calculation or control-flow block.
            modules.MLPLayer(args.embedding_dims, args.hidden_dims)  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.

        encoders = [encoder(args.hidden_dims, n_heads=args.n_heads, n_mha=args.n_mha, n_alignn=args.n_alignn, n_gnn=args.n_gnn) for _ in range(args.num_layers - 1)]  # Create or apply a trainable neural-network component.
        encoders.append(encoder(args.hidden_dims, n_heads=args.n_heads, n_mha=args.n_mha, n_alignn=args.n_alignn, n_gnn=args.n_gnn, last=True))  # had to add a last identifier due to training issues
        self.encoders = nn.ModuleList(encoders)  # Store this configuration value or neural-network submodule on the instance.

        self.global_pool = AvgPooling()  # Store this configuration value or neural-network submodule on the instance.

        self.final_fc = nn.Linear(args.hidden_dims, args.out_dims)  # Store this configuration value or neural-network submodule on the instance.

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, fg: dgl.DGLGraph):  # Define this callable; its indented block implements the documented operation.

        """Apply this module to its input tensors or graphs."""
        # Consume raw graph fields once; DGL loaders supply fresh batched graphs.
        atom_attr, edge_attr = g.ndata.pop('node_feats'), g.edata.pop('edge_feats')
        angle_attr = lg.edata.pop('angle_feats')
        fc_attr = fg.edata.pop('fc_feats')
        pe_attr = g.ndata.pop('pes')

        # Embed atom and edge properties
        atom_attr = self.atom_embedding(atom_attr)  # Create or apply a trainable neural-network component.
        pe_attr = self.positional_embedding(pe_attr)  # Create or apply a trainable neural-network component.
        # Add deterministic Laplacian position information to atom chemistry.
        atom_attr = atom_attr + pe_attr  # Bind this name to an intermediate value, configuration setting, or result.
        edge_attr = torch.squeeze(self.edge_expansion(edge_attr), dim=1)  # Construct or transform graph topology, geometry, or molecular feature data.
        edge_attr = self.edge_embedding(edge_attr)  # Construct or transform graph topology, geometry, or molecular feature data.
        angle_attr = torch.squeeze(self.angle_expansion(angle_attr), dim=1)  # Construct or transform graph topology, geometry, or molecular feature data.
        angle_attr = self.angle_embedding(angle_attr)  # Construct or transform graph topology, geometry, or molecular feature data.
        fc_attr = self.fc_embedding(fc_attr)  # Create or apply a trainable neural-network component.

        # Pass through graformer encoder layers
        for encdr in self.encoders:  # Iterate over the stated records, layers, batches, or graph elements.
            atom_attr, edge_attr, angle_attr, fc_attr = encdr(g, lg, fg, atom_attr, edge_attr, angle_attr, fc_attr)  # Construct or transform graph topology, geometry, or molecular feature data.

        # Perform pooling operation to get single a single feature vector for the entire molecule
        # Mean-pool variable atom counts into one vector per structure.
        graph_attr = self.global_pool(g, atom_attr)  # Construct or transform graph topology, geometry, or molecular feature data.

        out = self.final_fc(graph_attr)  # Construct or transform graph topology, geometry, or molecular feature data.
        return out, atom_attr, edge_attr, angle_attr, fc_attr  # Return this computed tensor, metric, object, or collection to the caller.
        

    def freeze_pretrain(self):  # Define this callable; its indented block implements the documented operation.
        """Keep the encoder trainable while excluding the final property head."""
        for name, param in self.named_parameters():  # Iterate over the stated records, layers, batches, or graph elements.
            if 'final_fc' not in name:
                param.requires_grad = True  # Bind this name to an intermediate value, configuration setting, or result.
            elif 'final_fc' in name:
                param.requires_grad = False  # Bind this name to an intermediate value, configuration setting, or result.

    def freeze_train(self):  # Define this callable; its indented block implements the documented operation.
        """Freeze the encoder and leave only the final property head trainable."""
        for name, param in self.named_parameters():  # Iterate over the stated records, layers, batches, or graph elements.
            if 'final_fc' not in name:
                param.requires_grad = False  # Bind this name to an intermediate value, configuration setting, or result.
            elif 'final_fc' in name:
                param.requires_grad = True  # Bind this name to an intermediate value, configuration setting, or result.
