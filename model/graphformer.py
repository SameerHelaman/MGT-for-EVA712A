"""Original complete Molecular Graph Transformer encoder and regressor."""

import dgl
import torch
from torch import nn, Tensor
from dgl.nn.pytorch import AvgPooling

import model.alignn as alignn, model.transformer as transformer
import modules.modules as modules


class encoder(nn.Module):
    """Apply Coulomb attention, ALIGNN, local GNN, FFN, normalization and residual updates."""
    def __init__(self, encoder_dims: int, n_heads: int = 4, n_mha: int = 1, n_alignn: int = 4, n_gnn: int = 4,
                 residual: bool = True, norm: bool = True, last: bool = False):
        """Initialize this object and its required state."""
        super(encoder, self).__init__()

        # Each attention head must receive an equal slice of the hidden vector.
        assert encoder_dims % n_heads == 0
        assert n_heads >= 1
        assert n_gnn >= 1

        # Store the per-head projection width used by multiheaded attention.
        self.head_dim = int(encoder_dims / n_heads)
        self.residual = residual
        self.norm = norm

        # The final encoder disables selected normalizations inside its last sublayers.
        if last:
            # Multi-Headed Attention Layers
            mha_layers = [transformer.multiheaded(encoder_dims, self.head_dim, heads=n_heads) for _ in range(n_mha - 1)]
            mha_layers.append(transformer.multiheaded(encoder_dims, self.head_dim, heads=n_heads, norm=(True, False)))

            # ALIGNN Blocks
            alignns = [alignn.ALIGNNLayer(encoder_dims) for _ in range(n_alignn - 1)]
            alignns.append(alignn.ALIGNNLayer(encoder_dims, edge_norm=(True, False)))

            # Graph Convolution Layers
            eggcs = [alignn.EdgeGatedGraphConv(encoder_dims) for _ in range(n_gnn - 1)]
            eggcs.append(alignn.EdgeGatedGraphConv(encoder_dims, norm=(True, False)))
        else:
            # Multi-Headed Attention Layers
            mha_layers = [transformer.multiheaded(encoder_dims, self.head_dim, heads=n_heads) for _ in range(n_mha)]

            # ALIGNN Blocks
            alignns = [alignn.ALIGNNLayer(encoder_dims) for _ in range(n_alignn)]

            # Graph Convolution Layers
            eggcs = [alignn.EdgeGatedGraphConv(encoder_dims) for _ in range(n_gnn)]
        
        self.mha_layers = nn.ModuleList(mha_layers)
        self.alignn_layers = nn.ModuleList(alignns)
        self.eggc_layers = nn.ModuleList(eggcs)

        # Linear layers with ReLU inbetween
        # Refine every atom independently with the encoder feed-forward block.
        self.lin_block = nn.Sequential(
            nn.Linear(encoder_dims, encoder_dims),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dims, encoder_dims)
        )

        # Normalize the feed-forward output when encoder normalization is enabled.
        if self.norm:
            self.normalizer = nn.LayerNorm(encoder_dims)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, fg: dgl.DGLGraph,
                x: Tensor, y: Tensor, z: Tensor, f: Tensor):

        # Get attended atom attributes
        """Apply this module to its input tensors or graphs."""
        # Propagate long-range information over the wider Coulomb graph.
        for mha in self.mha_layers:
            x, f = mha(fg, x, f)

        # Perform graph convolutions on the attended atom attributes
        # Couple atom, local-edge and angle states through each ALIGNN block.
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # Refine atom and local-edge states after angular processing.
        for eggc_layer in self.eggc_layers:
            x, y = eggc_layer(g, x, y)

        # Apply final two linear layers
        out = self.lin_block(x)

        if self.norm:
            out = self.normalizer(out)

        # Preserve the encoder input through an additive residual connection.
        if self.residual:
            out += x

        return out, y, z, f


class Graphformer(nn.Module):
    """Embed three graph views, run MGT encoders, pool atoms and predict properties."""
    def __init__(self, args):
        """Initialize this object and its required state."""
        super(Graphformer, self).__init__()

        # Embed elemental descriptors and positional encodings into one hidden width.
        self.atom_embedding = modules.MLPLayer(args.num_atom_fea, args.hidden_dims)
        self.positional_embedding = modules.MLPLayer(args.num_pe_fea, args.hidden_dims)
        # Expand scalar local distances before learned edge projection.
        self.edge_expansion = modules.RBFExpansion(vmin=0, vmax=args.local_radius, bins=args.num_edge_bins)
        self.edge_embedding = nn.Sequential(
            modules.MLPLayer(args.num_edge_bins, args.embedding_dims),
            modules.MLPLayer(args.embedding_dims, args.hidden_dims)
        )
        # Expand line-graph cosine values across the physical [-1, 1] range.
        self.angle_expansion = modules.RBFExpansion(vmin=-1, vmax=1, bins=args.num_angle_bins)
        self.angle_embedding = nn.Sequential(
            modules.MLPLayer(args.num_angle_bins, args.embedding_dims),
            modules.MLPLayer(args.embedding_dims, args.hidden_dims)
        )
        # Embed long-range ZiZj/rij Coulomb edge values for attention.
        self.fc_embedding = nn.Sequential(
            modules.MLPLayer(1, args.num_clmb_bins),
            modules.MLPLayer(args.num_clmb_bins, args.embedding_dims),
            modules.MLPLayer(args.embedding_dims, args.hidden_dims)
        )

        encoders = [encoder(args.hidden_dims, n_heads=args.n_heads, n_mha=args.n_mha, n_alignn=args.n_alignn, n_gnn=args.n_gnn) for _ in range(args.num_layers - 1)]
        encoders.append(encoder(args.hidden_dims, n_heads=args.n_heads, n_mha=args.n_mha, n_alignn=args.n_alignn, n_gnn=args.n_gnn, last=True))  # had to add a last identifier due to training issues
        self.encoders = nn.ModuleList(encoders)

        self.global_pool = AvgPooling()

        self.final_fc = nn.Linear(args.hidden_dims, args.out_dims)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, fg: dgl.DGLGraph):

        """Apply this module to its input tensors or graphs."""
        # Consume raw graph fields once; DGL loaders supply fresh batched graphs.
        atom_attr, edge_attr = g.ndata.pop('node_feats'), g.edata.pop('edge_feats')
        angle_attr = lg.edata.pop('angle_feats')
        fc_attr = fg.edata.pop('fc_feats')
        pe_attr = g.ndata.pop('pes')

        # Embed atom and edge properties
        atom_attr = self.atom_embedding(atom_attr)
        pe_attr = self.positional_embedding(pe_attr)
        # Add deterministic Laplacian position information to atom chemistry.
        atom_attr = atom_attr + pe_attr
        edge_attr = torch.squeeze(self.edge_expansion(edge_attr), dim=1)
        edge_attr = self.edge_embedding(edge_attr)
        angle_attr = torch.squeeze(self.angle_expansion(angle_attr), dim=1)
        angle_attr = self.angle_embedding(angle_attr)
        fc_attr = self.fc_embedding(fc_attr)

        # Pass through graformer encoder layers
        for encdr in self.encoders:
            atom_attr, edge_attr, angle_attr, fc_attr = encdr(g, lg, fg, atom_attr, edge_attr, angle_attr, fc_attr)

        # Perform pooling operation to get single a single feature vector for the entire molecule
        # Mean-pool variable atom counts into one vector per structure.
        graph_attr = self.global_pool(g, atom_attr)

        out = self.final_fc(graph_attr)
        return out, atom_attr, edge_attr, angle_attr, fc_attr
        

    def freeze_pretrain(self):
        """Keep the encoder trainable while excluding the final property head."""
        for name, param in self.named_parameters():
            if 'final_fc' not in name:
                param.requires_grad = True
            elif 'final_fc' in name:
                param.requires_grad = False

    def freeze_train(self):
        """Freeze the encoder and leave only the final property head trainable."""
        for name, param in self.named_parameters():
            if 'final_fc' not in name:
                param.requires_grad = False
            elif 'final_fc' in name:
                param.requires_grad = True
