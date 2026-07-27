"""Mask-pretrain ALIGNN or complete MGT, then run the matched affinity trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dgl
import torch
from torch import nn

import train_openbind_baseline as baseline
import train_openbind_mgt as mgt
from model.graphformer import Graphformer
from utils.masker import MaskAtom
from utils.datasets import compute_bond_cosines
from utils.molecular_features import ATOM_FEATURE_DIM
from utils.openbind_ligand_dataset import OpenBindGraphDataset


def encode_alignn(model, graph):
    """Run the existing ALIGNN encoder and return its atom representations."""
    line_graph = graph.line_graph(shared=True)
    line_graph.apply_edges(compute_bond_cosines)
    atoms = model.atom_embedding(graph.ndata["atom_features"])
    distances = model.distance_expansion(graph.edata["distance"].squeeze(1))
    edges = model.edge_embedding(torch.cat([graph.edata["bond_features"], distances], dim=1))
    angles = torch.squeeze(model.angle_expansion(line_graph.edata["angle_feats"]), dim=1)
    angles = model.angle_embedding(angles)
    for layer in model.alignn_layers:
        atoms, edges, angles = layer(graph, line_graph, atoms, edges, angles)
    return atoms


def masked_pretrain(model, loader, model_key, args, device):
    """Use the original MaskAtom transform with a correct reconstruction loss."""
    # Select the node-feature key used by the original or controlled graph schema.
    feature_name = "node_feats" if model_key == "mgt" else "atom_features"
    # Match the reconstruction head width to the unmasked input feature width.
    feature_dim = 90 if model_key == "mgt" else ATOM_FEATURE_DIM
    # Instantiate the unchanged original transform that zeros sampled atom rows.
    masker = MaskAtom(feature_dim, args.mask_rate, feature_name)
    # Decode contextual hidden states back into the original atom feature space.
    decoder = nn.Linear(args.hidden_dim, feature_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=args.pretrain_learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()
    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        decoder.train()
        total = 0.0
        node_count = 0
        # Iterate only over the frozen training partition supplied by the caller.
        for batch in loader:
            # Mask the batched atom graph and retain selected original rows as labels.
            graph, masked = masker(batch[0])
            # DGL NID maps subgraph nodes back to their batched-graph positions.
            masked_indices = masked.ndata[dgl.NID].to(device)
            # These copied pre-mask vectors are the self-supervised targets.
            truth = masked.ndata[feature_name].to(device)
            optimizer.zero_grad(set_to_none=True)
            # MGT consumes local, line and full Coulomb graphs together.
            if model_key == "mgt":
                graph = graph.to(device)
                line_graph = batch[1].to(device)
                full_graph = batch[2].to(device)
                _, representations, _, _, _ = model(graph, line_graph, full_graph)
            # ALIGNN consumes the controlled local/spatial graph only.
            else:
                graph = graph.to(device)
                representations = encode_alignn(model, graph)
            # Reconstruct only atoms hidden by MaskAtom, not visible atom rows.
            prediction = decoder(representations[masked_indices])
            loss = criterion(prediction, truth)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite masked reconstruction loss")
            loss.backward()
            optimizer.step()
            count = truth.shape[0]
            total += float(loss.detach()) * count
            node_count += count
        epoch_loss = total / node_count
        history.append({"epoch": epoch, "reconstruction_mse": epoch_loss})
        print(f"mask_pretrain_epoch={epoch} reconstruction_mse={epoch_loss:.6f}", flush=True)
    return history


def run_alignn(args, device, split_rows):
    """Mask-pretrain ALIGNN, inject it into the matched trainer, and evaluate."""
    # Rebuild graphs from crystallographic reference SDFs for this split.
    dataset = OpenBindGraphDataset(
        data_root=args.data_root,
        mode="3d_alignn",
        spatial_cutoff=args.spatial_cutoff,
        max_neighbors=args.max_neighbors,
    )
    subsets, _ = baseline.build_subsets(dataset, split_rows)
    namespace = argparse.Namespace(**vars(args))
    namespace.model = "3d_alignn"
    loaders = baseline.make_loaders(dataset, subsets, namespace)
    model = baseline.build_model(namespace).to(device)
    history = masked_pretrain(model, loaders["train"], "alignn", args, device)
    # Temporarily make the unchanged matched trainer reuse pretrained weights.
    original_factory = baseline.build_model
    baseline.build_model = lambda _: model
    try:
        result = baseline.train(namespace)
    finally:
        baseline.build_model = original_factory
    metrics_path = Path(args.output_root) / "3d_alignn" / args.split_method / f"seed_{args.seed}" / "metrics.json"
    return result, history, metrics_path


def run_mgt(args, device, split_rows):
    """Mask-pretrain Graphformer, inject it into the matched trainer, and evaluate."""
    # Construct the original StructureDataset interface for frozen graph triplets.
    dataset_args = mgt.make_dataset_args(args)
    dataset = mgt.StructureDataset(dataset_args, process=False, random_seed=args.seed)
    subsets, _ = mgt.build_subsets(dataset, split_rows)
    namespace = argparse.Namespace(**vars(args))
    namespace.num_layers = args.mgt_num_layers
    loaders = mgt.make_loaders(dataset, subsets, namespace)
    with (Path(args.data_root) / "atom_init.json").open() as handle:
        feature_dim = len(next(iter(json.load(handle).values())))
    namespace.num_atom_fea = feature_dim
    namespace.num_edge_fea = 1
    namespace.num_angle_fea = 1
    namespace.num_clmb_fea = 1
    namespace.out_dims = 1
    namespace.residual = bool(namespace.residual)
    model = Graphformer(namespace).to(device)
    history = masked_pretrain(model, loaders["train"], "mgt", args, device)
    # Temporarily inject this pretrained Graphformer into matched fine-tuning.
    original_factory = mgt.Graphformer
    mgt.Graphformer = lambda _: model
    try:
        result = mgt.train(namespace)
    finally:
        mgt.Graphformer = original_factory
    metrics_path = Path(args.output_root) / args.split_method / f"seed_{args.seed}" / "metrics.json"
    return result, history, metrics_path


def main():
    """Parse a reproducible masking experiment and persist its full metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["3d_alignn", "mgt"], required=True)
    parser.add_argument("--split_method", choices=["random", "scaffold"], required=True)
    parser.add_argument("--data_root", default=str(mgt.DEFAULT_DATA_ROOT))
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.2)
    parser.add_argument("--pretrain_learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--mgt_num_layers", type=int, default=1)
    parser.add_argument("--spatial_cutoff", type=float, default=5.0)
    parser.add_argument("--max_neighbors", type=int, default=32)
    parser.add_argument("--distance_bins", type=int, default=40)
    parser.add_argument("--angle_bins", type=int, default=40)
    parser.add_argument("--local_radius", type=float, default=8.0)
    parser.add_argument("--num_pe_fea", type=int, default=10)
    parser.add_argument("--num_edge_bins", type=int, default=80)
    parser.add_argument("--num_angle_bins", type=int, default=40)
    parser.add_argument("--num_clmb_bins", type=int, default=120)
    parser.add_argument("--embedding_dims", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, default=512)
    parser.add_argument("--n_mha", type=int, default=1)
    parser.add_argument("--n_alignn", type=int, default=3)
    parser.add_argument("--n_gnn", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--residual", type=int, default=1)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()
    if args.hidden_dim != args.hidden_dims:
        raise ValueError("hidden_dim and hidden_dims must match")
    if args.output_root is None:
        args.output_root = str(Path("output") / f"openbind_masked_{args.model}")
    mgt.seed_everything(args.seed)
    device = mgt.resolve_device(args.device)
    split_path = Path(args.data_root) / "splits" / f"{args.split_method}_seed_{args.seed}.csv"
    split_rows = mgt.read_csv(split_path)
    if args.model == "3d_alignn":
        result, pretraining, metrics_path = run_alignn(args, device, split_rows)
    else:
        result, pretraining, metrics_path = run_mgt(args, device, split_rows)
    # Attach pretraining provenance and the complete reconstruction history.
    result["masked_pretraining"] = {
        "transform": "original utils.masker.MaskAtom",
        "mask_rate": args.mask_rate,
        "epochs": args.pretrain_epochs,
        "objective": "MSE atom-feature reconstruction",
        "training_partition_only": True,
        "history": pretraining,
    }
    with metrics_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
