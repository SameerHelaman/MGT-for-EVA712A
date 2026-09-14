# =============================================================================
# MODULE: train_openbind_masked.py
# PURPOSE: Atom-feature reconstruction pretraining followed by matched ALIGNN or MGT affinity training.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Pretraining metadata followed by the selected matched trainer outputs.
# CALCULATIONS: Random atom vectors are zeroed and reconstructed with MSE before supervised fine-tuning.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Mask-pretrain ALIGNN or complete MGT, then run the matched affinity trainer."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import argparse  # Load a standard-library, scientific, or local project dependency.
import json  # Load a standard-library, scientific, or local project dependency.
from pathlib import Path  # Import selected classes or functions from the named dependency.

import dgl  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
from torch import nn  # Import selected classes or functions from the named dependency.

import train_openbind_3d_alignn as baseline  # Load a standard-library, scientific, or local project dependency.
import train_openbind_mgt as mgt  # Load a standard-library, scientific, or local project dependency.
from model.graphformer import Graphformer  # Import selected classes or functions from the named dependency.
from utils.masker import MaskAtom  # Import selected classes or functions from the named dependency.
from utils.datasets import compute_bond_cosines  # Import selected classes or functions from the named dependency.
from utils.molecular_features import ATOM_FEATURE_DIM  # Import selected classes or functions from the named dependency.
from utils.openbind_ligand_dataset import OpenBindGraphDataset  # Import selected classes or functions from the named dependency.


# FUNCTION: encode_alignn — see its docstring and inline comments.
def encode_alignn(model, graph):  # Define this callable; its indented block implements the documented operation.
    """Run the existing ALIGNN encoder and return its atom representations."""
    line_graph = graph.line_graph(shared=True)  # Construct or transform graph topology, geometry, or molecular feature data.
    line_graph.apply_edges(compute_bond_cosines)  # Perform this step of the surrounding calculation or control-flow block.
    atoms = model.atom_embedding(graph.ndata["atom_features"])
    distances = model.distance_expansion(graph.edata["distance"].squeeze(1))
    edges = model.edge_embedding(torch.cat([graph.edata["bond_features"], distances], dim=1))
    angles = torch.squeeze(model.angle_expansion(line_graph.edata["angle_feats"]), dim=1)
    angles = model.angle_embedding(angles)  # Construct or transform graph topology, geometry, or molecular feature data.
    for layer in model.alignn_layers:  # Iterate over the stated records, layers, batches, or graph elements.
        atoms, edges, angles = layer(graph, line_graph, atoms, edges, angles)  # Construct or transform graph topology, geometry, or molecular feature data.
    return atoms  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: masked_pretrain — see its docstring and inline comments.
def masked_pretrain(model, loader, model_key, args, device):  # Define this callable; its indented block implements the documented operation.
    """Use the original MaskAtom transform with a correct reconstruction loss."""
    # Select the node-feature key used by the original or controlled graph schema.
    feature_name = "node_feats" if model_key == "mgt" else "atom_features"
    # Match the reconstruction head width to the unmasked input feature width.
    feature_dim = 90 if model_key == "mgt" else ATOM_FEATURE_DIM
    # Instantiate the unchanged original transform that zeros sampled atom rows.
    masker = MaskAtom(feature_dim, args.mask_rate, feature_name)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Decode contextual hidden states back into the original atom feature space.
    decoder = nn.Linear(args.hidden_dim, feature_dim).to(device)  # Construct or transform graph topology, geometry, or molecular feature data.
    optimizer = torch.optim.Adam(  # Bind this name to an intermediate value, configuration setting, or result.
        list(model.parameters()) + list(decoder.parameters()),  # Perform this step of the surrounding calculation or control-flow block.
        lr=args.pretrain_learning_rate,  # Bind this name to an intermediate value, configuration setting, or result.
        weight_decay=args.weight_decay,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    criterion = nn.MSELoss()  # Compute or store a loss, error, residual, or regression evaluation statistic.
    history = []  # Bind this name to an intermediate value, configuration setting, or result.
    for epoch in range(1, args.pretrain_epochs + 1):  # Iterate over the stated records, layers, batches, or graph elements.
        model.train()  # Switch the model to training behaviour.
        decoder.train()  # Switch the model to training behaviour.
        total = 0.0  # Bind this name to an intermediate value, configuration setting, or result.
        node_count = 0  # Construct or transform graph topology, geometry, or molecular feature data.
        # Iterate only over the frozen training partition supplied by the caller.
        for batch in loader:  # Iterate over the stated records, layers, batches, or graph elements.
            # Mask the batched atom graph and retain selected original rows as labels.
            graph, masked = masker(batch[0])  # Construct or transform graph topology, geometry, or molecular feature data.
            # DGL NID maps subgraph nodes back to their batched-graph positions.
            masked_indices = masked.ndata[dgl.NID].to(device)  # Bind this name to an intermediate value, configuration setting, or result.
            # These copied pre-mask vectors are the self-supervised targets.
            truth = masked.ndata[feature_name].to(device)  # Construct or transform graph topology, geometry, or molecular feature data.
            optimizer.zero_grad(set_to_none=True)  # Clear accumulated gradients before the next optimization update.
            # MGT consumes local, line and full Coulomb graphs together.
            if model_key == "mgt":
                graph = graph.to(device)  # Construct or transform graph topology, geometry, or molecular feature data.
                line_graph = batch[1].to(device)  # Construct or transform graph topology, geometry, or molecular feature data.
                full_graph = batch[2].to(device)  # Construct or transform graph topology, geometry, or molecular feature data.
                _, representations, _, _, _ = model(graph, line_graph, full_graph)  # Construct or transform graph topology, geometry, or molecular feature data.
            # ALIGNN consumes the controlled local/spatial graph only.
            else:  # Handle the remaining case not covered by earlier conditions.
                graph = graph.to(device)  # Construct or transform graph topology, geometry, or molecular feature data.
                representations = encode_alignn(model, graph)  # Construct or transform graph topology, geometry, or molecular feature data.
            # Reconstruct only atoms hidden by MaskAtom, not visible atom rows.
            prediction = decoder(representations[masked_indices])  # Create or apply a trainable neural-network component.
            loss = criterion(prediction, truth)  # Compute or store a loss, error, residual, or regression evaluation statistic.
            if not torch.isfinite(loss):  # Evaluate this condition before executing the associated branch.
                raise FloatingPointError("Non-finite masked reconstruction loss")
            loss.backward()  # Backpropagate the loss to compute gradients for trainable parameters.
            optimizer.step()  # Advance the optimizer or learning-rate scheduler by one update.
            count = truth.shape[0]  # Bind this name to an intermediate value, configuration setting, or result.
            total += float(loss.detach()) * count  # Compute or store a loss, error, residual, or regression evaluation statistic.
            node_count += count  # Construct or transform graph topology, geometry, or molecular feature data.
        epoch_loss = total / node_count  # Compute or store a loss, error, residual, or regression evaluation statistic.
        history.append({"epoch": epoch, "reconstruction_mse": epoch_loss})
        print(f"mask_pretrain_epoch={epoch} reconstruction_mse={epoch_loss:.6f}", flush=True)  # Report progress, predictions, or metrics to the selected output/logging backend.
    return history  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: run_alignn — see its docstring and inline comments.
def run_alignn(args, device, split_rows):  # Define this callable; its indented block implements the documented operation.
    """Mask-pretrain ALIGNN, inject it into the matched trainer, and evaluate."""
    # Rebuild graphs from crystallographic reference SDFs for this split.
    dataset = OpenBindGraphDataset(  # Construct or transform graph topology, geometry, or molecular feature data.
        data_root=args.data_root,  # Bind this name to an intermediate value, configuration setting, or result.
        mode="3d_alignn",
        spatial_cutoff=args.spatial_cutoff,  # Bind this name to an intermediate value, configuration setting, or result.
        max_neighbors=args.max_neighbors,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    subsets, _ = baseline.build_subsets(dataset, split_rows)  # Prepare dataset membership or batched data access for the experiment.
    namespace = argparse.Namespace(**vars(args))  # Bind this name to an intermediate value, configuration setting, or result.
    namespace.model = "3d_alignn"
    loaders = baseline.make_loaders(dataset, subsets, namespace)  # Prepare dataset membership or batched data access for the experiment.
    model = baseline.build_model(namespace).to(device)  # Create or apply a trainable neural-network component.
    history = masked_pretrain(model, loaders["train"], "alignn", args, device)
    # Temporarily make the unchanged matched trainer reuse pretrained weights.
    original_factory = baseline.build_model  # Create or apply a trainable neural-network component.
    baseline.build_model = lambda _: model  # Create or apply a trainable neural-network component.
    try:  # Begin protected execution for an operation that may raise an exception.
        result = baseline.train(namespace)  # Switch the model to training behaviour.
    finally:  # Run cleanup regardless of whether the protected operation succeeded.
        baseline.build_model = original_factory  # Create or apply a trainable neural-network component.
    metrics_path = Path(args.output_root) / "3d_alignn" / args.split_method / f"seed_{args.seed}" / "metrics.json"
    return result, history, metrics_path  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: run_mgt — see its docstring and inline comments.
def run_mgt(args, device, split_rows):  # Define this callable; its indented block implements the documented operation.
    """Mask-pretrain Graphformer, inject it into the matched trainer, and evaluate."""
    # Construct the original StructureDataset interface for frozen graph triplets.
    dataset_args = mgt.make_dataset_args(args)  # Prepare dataset membership or batched data access for the experiment.
    dataset = mgt.StructureDataset(dataset_args, process=False, random_seed=args.seed)  # Prepare dataset membership or batched data access for the experiment.
    subsets, _ = mgt.build_subsets(dataset, split_rows)  # Prepare dataset membership or batched data access for the experiment.
    namespace = argparse.Namespace(**vars(args))  # Bind this name to an intermediate value, configuration setting, or result.
    namespace.num_layers = args.mgt_num_layers  # Create or apply a trainable neural-network component.
    loaders = mgt.make_loaders(dataset, subsets, namespace)  # Prepare dataset membership or batched data access for the experiment.
    with (Path(args.data_root) / "atom_init.json").open() as handle:
        feature_dim = len(next(iter(json.load(handle).values())))  # Construct or transform graph topology, geometry, or molecular feature data.
    namespace.num_atom_fea = feature_dim  # Construct or transform graph topology, geometry, or molecular feature data.
    namespace.num_edge_fea = 1  # Construct or transform graph topology, geometry, or molecular feature data.
    namespace.num_angle_fea = 1  # Construct or transform graph topology, geometry, or molecular feature data.
    namespace.num_clmb_fea = 1  # Bind this name to an intermediate value, configuration setting, or result.
    namespace.out_dims = 1  # Bind this name to an intermediate value, configuration setting, or result.
    namespace.residual = bool(namespace.residual)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    model = Graphformer(namespace).to(device)  # Construct or transform graph topology, geometry, or molecular feature data.
    history = masked_pretrain(model, loaders["train"], "mgt", args, device)
    # Temporarily inject this pretrained Graphformer into matched fine-tuning.
    original_factory = mgt.Graphformer  # Construct or transform graph topology, geometry, or molecular feature data.
    mgt.Graphformer = lambda _: model  # Construct or transform graph topology, geometry, or molecular feature data.
    try:  # Begin protected execution for an operation that may raise an exception.
        result = mgt.train(namespace)  # Switch the model to training behaviour.
    finally:  # Run cleanup regardless of whether the protected operation succeeded.
        mgt.Graphformer = original_factory  # Construct or transform graph topology, geometry, or molecular feature data.
    metrics_path = Path(args.output_root) / args.split_method / f"seed_{args.seed}" / "metrics.json"
    return result, history, metrics_path  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: main — see its docstring and inline comments.
def main():  # Define this callable; its indented block implements the documented operation.
    """Parse a reproducible masking experiment and persist its full metadata."""
    parser = argparse.ArgumentParser()  # Bind this name to an intermediate value, configuration setting, or result.
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
    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.
    if args.hidden_dim != args.hidden_dims:  # Evaluate this condition before executing the associated branch.
        raise ValueError("hidden_dim and hidden_dims must match")
    if args.output_root is None:  # Evaluate this condition before executing the associated branch.
        args.output_root = str(Path("output") / f"openbind_masked_{args.model}")
    mgt.seed_everything(args.seed)  # Perform this step of the surrounding calculation or control-flow block.
    device = mgt.resolve_device(args.device)  # Bind this name to an intermediate value, configuration setting, or result.
    # Respect an explicitly frozen CV manifest for both masking pretraining and fine-tuning.
    split_path = (  # Select either the supplied CV manifest or the original fixed split.
        Path(args.split_file).resolve()  # Resolve the user-supplied manifest to an absolute path.
        if args.split_file  # Use the explicit manifest whenever one was supplied.
        else Path(args.data_root) / "splits" / f"{args.split_method}_seed_{args.seed}.csv"  # Fall back to the original split.
    )  # Finish selection of the single manifest shared by pretraining and fine-tuning.
    split_rows = mgt.read_csv(split_path)  # Prepare dataset membership or batched data access for the experiment.
    if args.model == "3d_alignn":
        result, pretraining, metrics_path = run_alignn(args, device, split_rows)  # Prepare dataset membership or batched data access for the experiment.
    else:  # Handle the remaining case not covered by earlier conditions.
        result, pretraining, metrics_path = run_mgt(args, device, split_rows)  # Prepare dataset membership or batched data access for the experiment.
    # Attach pretraining provenance and the complete reconstruction history.
    result["masked_pretraining"] = {
        "transform": "original utils.masker.MaskAtom",
        "mask_rate": args.mask_rate,
        "epochs": args.pretrain_epochs,
        "objective": "MSE atom-feature reconstruction",
        "training_partition_only": True,
        "history": pretraining,
    }  # Continue or close the surrounding multiline expression or collection.
    with metrics_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)  # Bind this name to an intermediate value, configuration setting, or result.
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))  # Report progress, predictions, or metrics to the selected output/logging backend.


if __name__ == "__main__":
    main()  # Perform this step of the surrounding calculation or control-flow block.
