# =============================================================================
# MODULE: train_openbind_mgt.py
# PURPOSE: Matched OpenBind MGT trainer, evaluator, compound aggregator and report writer.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: history.csv, metrics.json, structure and compound prediction CSV files, checkpoints.
# CALCULATIONS: Targets are normalized using training mean/scale. Optimization uses Huber loss; reports MAE, MSE, RMSE, R², MAPE, Pearson and Spearman. Repeated test structures are averaged by compound.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Train the complete original MGT Graphformer on OpenBind ligand PDBs."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import argparse  # Load a standard-library, scientific, or local project dependency.
import csv  # Load a standard-library, scientific, or local project dependency.
import json  # Load a standard-library, scientific, or local project dependency.
import math  # Load a standard-library, scientific, or local project dependency.
import random  # Load a standard-library, scientific, or local project dependency.
import time  # Load a standard-library, scientific, or local project dependency.
from collections import defaultdict  # Import selected classes or functions from the named dependency.
from pathlib import Path  # Import selected classes or functions from the named dependency.

import dgl  # Load a standard-library, scientific, or local project dependency.
import numpy as np  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
from scipy.stats import pearsonr, spearmanr  # Import selected classes or functions from the named dependency.
from torch import nn  # Import selected classes or functions from the named dependency.
from torch.utils.data import DataLoader, Subset  # Import selected classes or functions from the named dependency.

from model.graphformer import Graphformer  # Import selected classes or functions from the named dependency.
from utils.datasets import StructureDataset  # Import selected classes or functions from the named dependency.


PROJECT_ROOT = Path(__file__).resolve().parent  # Bind this name to an intermediate value, configuration setting, or result.
DEFAULT_DATA_ROOT = (  # Bind this name to an intermediate value, configuration setting, or result.
    PROJECT_ROOT / "OpenBind_EV-A71_2A" / "experiment_a_ligand_mgt"
)  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: seed_everything — see its docstring and inline comments.
def seed_everything(seed: int) -> None:  # Define this callable; its indented block implements the documented operation.
    """Seed Python, NumPy, Torch, CUDA and DGL."""
    random.seed(seed)  # Perform this step of the surrounding calculation or control-flow block.
    np.random.seed(seed)  # Invoke the relevant tensor, graph, numerical, or tabular operation.
    torch.manual_seed(seed)  # Invoke the relevant tensor, graph, numerical, or tabular operation.
    dgl.seed(seed)  # Invoke the relevant tensor, graph, numerical, or tabular operation.
    if torch.cuda.is_available():  # Evaluate this condition before executing the associated branch.
        torch.cuda.manual_seed_all(seed)  # Invoke the relevant tensor, graph, numerical, or tabular operation.


# FUNCTION: read_csv — see its docstring and inline comments.
def read_csv(path: Path) -> list[dict[str, str]]:  # Define this callable; its indented block implements the documented operation.
    """Read one CSV manifest into dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: resolve_device — see its docstring and inline comments.
def resolve_device(requested: str) -> torch.device:  # Define this callable; its indented block implements the documented operation.
    """Resolve an explicit or automatic Torch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: regression_metrics — see its docstring and inline comments.
def regression_metrics(  # Define this callable; its indented block implements the documented operation.
    observed: np.ndarray, predicted: np.ndarray  # Perform this step of the surrounding calculation or control-flow block.
) -> dict[str, float]:  # Continue or close the surrounding multiline expression or collection.
    """Calculate standard affinity-regression metrics."""
    observed = np.asarray(observed, dtype=np.float64)  # Bind this name to an intermediate value, configuration setting, or result.
    predicted = np.asarray(predicted, dtype=np.float64)  # Bind this name to an intermediate value, configuration setting, or result.
    residual = predicted - observed  # Compute or store a loss, error, residual, or regression evaluation statistic.
    mse = float(np.mean(residual**2))  # Compute or store a loss, error, residual, or regression evaluation statistic.
    mae = float(np.mean(np.abs(residual)))  # Compute or store a loss, error, residual, or regression evaluation statistic.
    denominator = float(np.sum((observed - observed.mean()) ** 2))  # Bind this name to an intermediate value, configuration setting, or result.
    r2 = (  # Compute or store a loss, error, residual, or regression evaluation statistic.
        float(1.0 - np.sum(residual**2) / denominator)  # Perform this step of the surrounding calculation or control-flow block.
        if denominator > 0  # Evaluate this condition before executing the associated branch.
        else float("nan")
    )  # Continue or close the surrounding multiline expression or collection.
    pearson = (  # Compute or store a loss, error, residual, or regression evaluation statistic.
        float(pearsonr(observed, predicted).statistic)  # Perform this step of the surrounding calculation or control-flow block.
        if len(observed) > 1  # Evaluate this condition before executing the associated branch.
        and np.std(observed) > 0  # Perform this step of the surrounding calculation or control-flow block.
        and np.std(predicted) > 0  # Perform this step of the surrounding calculation or control-flow block.
        else float("nan")
    )  # Continue or close the surrounding multiline expression or collection.
    spearman = (  # Compute or store a loss, error, residual, or regression evaluation statistic.
        float(spearmanr(observed, predicted).statistic)  # Perform this step of the surrounding calculation or control-flow block.
        if len(observed) > 1  # Evaluate this condition before executing the associated branch.
        and np.std(observed) > 0  # Perform this step of the surrounding calculation or control-flow block.
        and np.std(predicted) > 0  # Perform this step of the surrounding calculation or control-flow block.
        else float("nan")
    )  # Continue or close the surrounding multiline expression or collection.
    nonzero = np.abs(observed) > np.finfo(np.float64).eps  # Bind this name to an intermediate value, configuration setting, or result.
    mape = (  # Bind this name to an intermediate value, configuration setting, or result.
        float(np.mean(np.abs(residual[nonzero] / observed[nonzero])) * 100)  # Perform this step of the surrounding calculation or control-flow block.
        if np.any(nonzero)  # Evaluate this condition before executing the associated branch.
        else float("nan")
    )  # Continue or close the surrounding multiline expression or collection.
    return {  # Return this computed tensor, metric, object, or collection to the caller.
        "count": int(len(observed)),
        "mae": mae,
        "mape_percent": mape,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2,
        "pearson_r": pearson,
        "spearman_r": spearman,
    }  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: make_dataset_args — see its docstring and inline comments.
def make_dataset_args(args: argparse.Namespace) -> argparse.Namespace:  # Define this callable; its indented block implements the documented operation.
    """Provide the exact fields required by original StructureDataset."""
    return argparse.Namespace(  # Return this computed tensor, metric, object, or collection to the caller.
        root=str(Path(args.data_root).resolve()),  # Bind this name to an intermediate value, configuration setting, or result.
        max_nei_num=args.max_neighbors,  # Bind this name to an intermediate value, configuration setting, or result.
        num_pe_fea=args.num_pe_fea,  # Bind this name to an intermediate value, configuration setting, or result.
        local_radius=args.local_radius,  # Bind this name to an intermediate value, configuration setting, or result.
        periodic=False,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: build_subsets — see its docstring and inline comments.
def build_subsets(  # Define this callable; its indented block implements the documented operation.
    dataset: StructureDataset, split_rows: list[dict[str, str]]  # Perform this step of the surrounding calculation or control-flow block.
) -> tuple[dict[str, Subset], dict[str, str]]:  # Continue or close the surrounding multiline expression or collection.
    """Map frozen manifest membership onto original StructureDataset indices."""
    id_to_index = {  # Bind this name to an intermediate value, configuration setting, or result.
        row[0]: index for index, row in enumerate(dataset.id_prop_data)  # Perform this step of the surrounding calculation or control-flow block.
    }  # Continue or close the surrounding multiline expression or collection.
    if len(id_to_index) != len(dataset):  # Evaluate this condition before executing the associated branch.
        raise ValueError("Duplicate complex names in id_prop.csv")
    expected_ids = set(id_to_index)  # Bind this name to an intermediate value, configuration setting, or result.
    manifest_ids = {row["complex_name"] for row in split_rows}
    if expected_ids != manifest_ids:  # Evaluate this condition before executing the associated branch.
        raise ValueError("Split manifest and id_prop.csv do not have equal coverage")
    compound_by_id = {  # Bind this name to an intermediate value, configuration setting, or result.
        row["complex_name"]: row["official_compound_group_id"]
        for row in split_rows  # Iterate over the stated records, layers, batches, or graph elements.
    }  # Continue or close the surrounding multiline expression or collection.
    subsets = {}  # Prepare dataset membership or batched data access for the experiment.
    # Evaluate all partitions identically; test remains untouched during selection.
    for split in ("train", "validation", "test"):
        members = [  # Bind this name to an intermediate value, configuration setting, or result.
            id_to_index[row["complex_name"]]
            for row in split_rows  # Iterate over the stated records, layers, batches, or graph elements.
            if row["split"] == split
        ]  # Continue or close the surrounding multiline expression or collection.
        if not members:  # Evaluate this condition before executing the associated branch.
            raise ValueError(f"Frozen split {split} is empty")  # Reject invalid input or state with an explicit exception.
        subsets[split] = Subset(dataset, members)  # Prepare dataset membership or batched data access for the experiment.
    for compound_id in set(compound_by_id.values()):  # Iterate over the stated records, layers, batches, or graph elements.
        memberships = {  # Bind this name to an intermediate value, configuration setting, or result.
            row["split"]
            for row in split_rows  # Iterate over the stated records, layers, batches, or graph elements.
            if row["official_compound_group_id"] == compound_id
        }  # Continue or close the surrounding multiline expression or collection.
        if len(memberships) != 1:  # Evaluate this condition before executing the associated branch.
            raise ValueError(f"Compound leakage detected for {compound_id}")  # Reject invalid input or state with an explicit exception.
    return subsets, compound_by_id  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: make_loaders — see its docstring and inline comments.
def make_loaders(  # Define this callable; its indented block implements the documented operation.
    dataset: StructureDataset,  # Perform this step of the surrounding calculation or control-flow block.
    subsets: dict[str, Subset],  # Perform this step of the surrounding calculation or control-flow block.
    args: argparse.Namespace,  # Perform this step of the surrounding calculation or control-flow block.
) -> dict[str, DataLoader]:  # Continue or close the surrounding multiline expression or collection.
    """Create deterministic graph loaders for all three frozen partitions."""
    generator = torch.Generator().manual_seed(args.seed)  # Bind this name to an intermediate value, configuration setting, or result.
    return {  # Return this computed tensor, metric, object, or collection to the caller.
        split: DataLoader(  # Perform this step of the surrounding calculation or control-flow block.
            subset,  # Perform this step of the surrounding calculation or control-flow block.
            batch_size=args.batch_size,  # Bind this name to an intermediate value, configuration setting, or result.
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=0,  # Bind this name to an intermediate value, configuration setting, or result.
            collate_fn=dataset.collate_tt,  # Prepare dataset membership or batched data access for the experiment.
            drop_last=False,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        for split, subset in subsets.items()  # Iterate over the stated records, layers, batches, or graph elements.
    }  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: move_graphs — see its docstring and inline comments.
def move_graphs(  # Define this callable; its indented block implements the documented operation.
    graph: dgl.DGLGraph,  # Perform this step of the surrounding calculation or control-flow block.
    line_graph: dgl.DGLGraph,  # Perform this step of the surrounding calculation or control-flow block.
    full_graph: dgl.DGLGraph,  # Perform this step of the surrounding calculation or control-flow block.
    device: torch.device,  # Perform this step of the surrounding calculation or control-flow block.
) -> tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:  # Continue or close the surrounding multiline expression or collection.
    """Move all three original MGT graph representations to one device."""
    return graph.to(device), line_graph.to(device), full_graph.to(device)  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: target_statistics — see its docstring and inline comments.
def target_statistics(  # Define this callable; its indented block implements the documented operation.
    dataset: StructureDataset, subset: Subset  # Perform this step of the surrounding calculation or control-flow block.
) -> tuple[float, float]:  # Continue or close the surrounding multiline expression or collection.
    """Calculate normalization values from training labels only."""
    values = torch.tensor(  # Bind this name to an intermediate value, configuration setting, or result.
        [  # Continue or close the surrounding multiline expression or collection.
            float(dataset.id_prop_data[index][1])  # Perform this step of the surrounding calculation or control-flow block.
            for index in subset.indices  # Iterate over the stated records, layers, batches, or graph elements.
        ],  # Continue or close the surrounding multiline expression or collection.
        dtype=torch.float32,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    mean = float(values.mean())  # Bind this name to an intermediate value, configuration setting, or result.
    scale = float(values.std(unbiased=False))  # Bind this name to an intermediate value, configuration setting, or result.
    if not math.isfinite(scale) or scale <= 0:  # Evaluate this condition before executing the associated branch.
        raise ValueError("Training target standard deviation is invalid")
    return mean, scale  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: train_epoch — see its docstring and inline comments.
def train_epoch(  # Define this callable; its indented block implements the documented operation.
    model: Graphformer,  # Perform this step of the surrounding calculation or control-flow block.
    loader: DataLoader,  # Perform this step of the surrounding calculation or control-flow block.
    optimizer: torch.optim.Optimizer,  # Perform this step of the surrounding calculation or control-flow block.
    criterion: nn.Module,  # Perform this step of the surrounding calculation or control-flow block.
    target_mean: float,  # Perform this step of the surrounding calculation or control-flow block.
    target_scale: float,  # Perform this step of the surrounding calculation or control-flow block.
    device: torch.device,  # Perform this step of the surrounding calculation or control-flow block.
) -> float:  # Continue or close the surrounding multiline expression or collection.
    """Train the complete original Graphformer for one epoch."""
    model.train()  # Switch the model to training behaviour.
    total_loss = 0.0  # Compute or store a loss, error, residual, or regression evaluation statistic.
    total_examples = 0  # Bind this name to an intermediate value, configuration setting, or result.
    for graph, line_graph, full_graph, target, _ in loader:  # Iterate over the stated records, layers, batches, or graph elements.
        graph, line_graph, full_graph = move_graphs(  # Construct or transform graph topology, geometry, or molecular feature data.
            graph, line_graph, full_graph, device  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        target = target.to(device)  # Bind this name to an intermediate value, configuration setting, or result.
        normalized_target = (target - target_mean) / target_scale  # Bind this name to an intermediate value, configuration setting, or result.
        optimizer.zero_grad(set_to_none=True)  # Clear accumulated gradients before the next optimization update.
        output, _, _, _, _ = model(graph, line_graph, full_graph)  # Construct or transform graph topology, geometry, or molecular feature data.
        loss = criterion(output, normalized_target)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        if not torch.isfinite(loss):  # Evaluate this condition before executing the associated branch.
            raise FloatingPointError("Non-finite training loss")
        loss.backward()  # Backpropagate the loss to compute gradients for trainable parameters.
        optimizer.step()  # Advance the optimizer or learning-rate scheduler by one update.
        batch_size = target.shape[0]  # Bind this name to an intermediate value, configuration setting, or result.
        total_loss += float(loss.detach()) * batch_size  # Compute or store a loss, error, residual, or regression evaluation statistic.
        total_examples += batch_size  # Bind this name to an intermediate value, configuration setting, or result.
    return total_loss / total_examples  # Return this computed tensor, metric, object, or collection to the caller.


@torch.no_grad()  # Apply this decorator to the following class or function.
# FUNCTION: predict — see its docstring and inline comments.
def predict(  # Define this callable; its indented block implements the documented operation.
    model: Graphformer,  # Perform this step of the surrounding calculation or control-flow block.
    loader: DataLoader,  # Perform this step of the surrounding calculation or control-flow block.
    target_mean: float,  # Perform this step of the surrounding calculation or control-flow block.
    target_scale: float,  # Perform this step of the surrounding calculation or control-flow block.
    device: torch.device,  # Perform this step of the surrounding calculation or control-flow block.
) -> tuple[np.ndarray, np.ndarray, list[str]]:  # Continue or close the surrounding multiline expression or collection.
    """Predict pKD in original units for one partition."""
    model.eval()  # Switch the model to deterministic evaluation behaviour.
    observed_parts = []  # Bind this name to an intermediate value, configuration setting, or result.
    predicted_parts = []  # Bind this name to an intermediate value, configuration setting, or result.
    identifiers = []  # Bind this name to an intermediate value, configuration setting, or result.
    for graph, line_graph, full_graph, target, ids in loader:  # Iterate over the stated records, layers, batches, or graph elements.
        graph, line_graph, full_graph = move_graphs(  # Construct or transform graph topology, geometry, or molecular feature data.
            graph, line_graph, full_graph, device  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        output, _, _, _, _ = model(graph, line_graph, full_graph)  # Construct or transform graph topology, geometry, or molecular feature data.
        prediction = output.squeeze(1) * target_scale + target_mean  # Bind this name to an intermediate value, configuration setting, or result.
        observed_parts.append(target.squeeze(1).cpu().numpy())  # Perform this step of the surrounding calculation or control-flow block.
        predicted_parts.append(prediction.cpu().numpy())  # Perform this step of the surrounding calculation or control-flow block.
        identifiers.extend(ids)  # Perform this step of the surrounding calculation or control-flow block.
    return (  # Return this computed tensor, metric, object, or collection to the caller.
        np.concatenate(observed_parts),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        np.concatenate(predicted_parts),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        identifiers,  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: compound_predictions — see its docstring and inline comments.
def compound_predictions(  # Define this callable; its indented block implements the documented operation.
    observed: np.ndarray,  # Perform this step of the surrounding calculation or control-flow block.
    predicted: np.ndarray,  # Perform this step of the surrounding calculation or control-flow block.
    identifiers: list[str],  # Perform this step of the surrounding calculation or control-flow block.
    compound_by_id: dict[str, str],  # Perform this step of the surrounding calculation or control-flow block.
) -> list[dict[str, object]]:  # Continue or close the surrounding multiline expression or collection.
    """Average repeated crystallographic predictions per official compound."""
    grouped = defaultdict(list)  # Bind this name to an intermediate value, configuration setting, or result.
    for truth, estimate, complex_name in zip(  # Iterate over the stated records, layers, batches, or graph elements.
        observed, predicted, identifiers  # Perform this step of the surrounding calculation or control-flow block.
    ):  # Continue or close the surrounding multiline expression or collection.
        grouped[compound_by_id[complex_name]].append(  # Perform this step of the surrounding calculation or control-flow block.
            (float(truth), float(estimate), complex_name)  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
    rows = []  # Bind this name to an intermediate value, configuration setting, or result.
    for compound_id, values in sorted(grouped.items()):  # Iterate over the stated records, layers, batches, or graph elements.
        truths = np.array([value[0] for value in values])  # Bind this name to an intermediate value, configuration setting, or result.
        if float(truths.max() - truths.min()) > 1e-5:  # Evaluate this condition before executing the associated branch.
            raise ValueError(f"Conflicting pKD values for {compound_id}")  # Reject invalid input or state with an explicit exception.
        rows.append(  # Perform this step of the surrounding calculation or control-flow block.
            {  # Continue or close the surrounding multiline expression or collection.
                "official_compound_group_id": compound_id,
                "experimental_pKD": float(truths[0]),
                "predicted_pKD": float(
                    np.mean([value[1] for value in values])  # Invoke the relevant tensor, graph, numerical, or tabular operation.
                ),  # Continue or close the surrounding multiline expression or collection.
                "structure_count": len(values),
                "complex_names": ";".join(
                    sorted(value[2] for value in values)  # Perform this step of the surrounding calculation or control-flow block.
                ),  # Continue or close the surrounding multiline expression or collection.
            }  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
    return rows  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: write_prediction_files — see its docstring and inline comments.
def write_prediction_files(  # Define this callable; its indented block implements the documented operation.
    output_dir: Path,  # Perform this step of the surrounding calculation or control-flow block.
    observed: np.ndarray,  # Perform this step of the surrounding calculation or control-flow block.
    predicted: np.ndarray,  # Perform this step of the surrounding calculation or control-flow block.
    identifiers: list[str],  # Perform this step of the surrounding calculation or control-flow block.
    compound_rows: list[dict[str, object]],  # Perform this step of the surrounding calculation or control-flow block.
    compound_by_id: dict[str, str],  # Perform this step of the surrounding calculation or control-flow block.
) -> None:  # Continue or close the surrounding multiline expression or collection.
    """Write structure- and compound-level held-out predictions."""
    structure_path = output_dir / "test_structure_predictions.csv"
    with structure_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [  # Bind this name to an intermediate value, configuration setting, or result.
            "complex_name",
            "official_compound_group_id",
            "experimental_pKD",
            "predicted_pKD",
            "residual",
        ]  # Continue or close the surrounding multiline expression or collection.
        writer = csv.DictWriter(handle, fieldnames=fields)  # Bind this name to an intermediate value, configuration setting, or result.
        writer.writeheader()  # Perform this step of the surrounding calculation or control-flow block.
        for truth, estimate, complex_name in zip(  # Iterate over the stated records, layers, batches, or graph elements.
            observed, predicted, identifiers  # Perform this step of the surrounding calculation or control-flow block.
        ):  # Continue or close the surrounding multiline expression or collection.
            writer.writerow(  # Perform this step of the surrounding calculation or control-flow block.
                {  # Continue or close the surrounding multiline expression or collection.
                    "complex_name": complex_name,
                    "official_compound_group_id": compound_by_id[complex_name],
                    "experimental_pKD": float(truth),
                    "predicted_pKD": float(estimate),
                    "residual": float(estimate - truth),
                }  # Continue or close the surrounding multiline expression or collection.
            )  # Continue or close the surrounding multiline expression or collection.
    compound_path = output_dir / "test_compound_predictions.csv"
    with compound_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(  # Bind this name to an intermediate value, configuration setting, or result.
            handle, fieldnames=list(compound_rows[0].keys())  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        writer.writeheader()  # Perform this step of the surrounding calculation or control-flow block.
        writer.writerows(compound_rows)  # Perform this step of the surrounding calculation or control-flow block.


# FUNCTION: train — see its docstring and inline comments.
def train(args: argparse.Namespace) -> dict[str, object]:  # Define this callable; its indented block implements the documented operation.
    """Run one complete fixed-split OpenBind Graphformer experiment."""
    # Fix all supported random-number generators before data/model construction.
    seed_everything(args.seed)  # Perform this step of the surrounding calculation or control-flow block.
    device = resolve_device(args.device)  # Bind this name to an intermediate value, configuration setting, or result.
    data_root = Path(args.data_root).resolve()  # Bind this name to an intermediate value, configuration setting, or result.
    split_path = (  # Prepare dataset membership or batched data access for the experiment.
        Path(args.split_file).resolve()  # Perform this step of the surrounding calculation or control-flow block.
        if args.split_file  # Evaluate this condition before executing the associated branch.
        else data_root  # Handle the remaining case not covered by earlier conditions.
        / "splits"
        / f"{args.split_method}_seed_{args.seed}.csv"  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    # Load exact frozen structure and compound membership for this experiment.
    split_rows = read_csv(split_path)  # Prepare dataset membership or batched data access for the experiment.
    if {row["seed"] for row in split_rows} != {str(args.seed)}:
        raise ValueError("Split manifest seed does not match requested seed")
    if {row["split_method"] for row in split_rows} != {args.split_method}:
        raise ValueError("Split manifest method does not match requested method")

    dataset_args = make_dataset_args(args)  # Prepare dataset membership or batched data access for the experiment.
    processed_dir = data_root / "processed"
    processed_count = len(list(processed_dir.glob("*.bin")))
    if processed_count != 621:  # Evaluate this condition before executing the associated branch.
        raise FileNotFoundError(  # Reject invalid input or state with an explicit exception.
            "Expected 621 frozen graph files. Run "
            "`python -m utils.preprocess_openbind_mgt_graphs` first."
        )  # Continue or close the surrounding multiline expression or collection.
    # Read the already-frozen original local/line/Coulomb graph triplets.
    dataset = StructureDataset(  # Prepare dataset membership or batched data access for the experiment.
        dataset_args, process=False, random_seed=args.seed  # Prepare dataset membership or batched data access for the experiment.
    )  # Continue or close the surrounding multiline expression or collection.
    subsets, compound_by_id = build_subsets(dataset, split_rows)  # Prepare dataset membership or batched data access for the experiment.
    loaders = make_loaders(dataset, subsets, args)  # Prepare dataset membership or batched data access for the experiment.
    # Normalize labels with training statistics only to prevent information leakage.
    target_mean, target_scale = target_statistics(dataset, subsets["train"])

    with (data_root / "atom_init.json").open(
        "r", encoding="utf-8"
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        atom_initializer = json.load(handle)  # Bind this name to an intermediate value, configuration setting, or result.
    args.num_atom_fea = len(next(iter(atom_initializer.values())))  # Bind this name to an intermediate value, configuration setting, or result.
    args.num_edge_fea = 1  # Construct or transform graph topology, geometry, or molecular feature data.
    args.num_angle_fea = 1  # Construct or transform graph topology, geometry, or molecular feature data.
    args.num_clmb_fea = 1  # Bind this name to an intermediate value, configuration setting, or result.
    args.out_dims = 1  # Bind this name to an intermediate value, configuration setting, or result.
    args.residual = bool(args.residual)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    # Instantiate the complete original Graphformer after adapting only output width.
    model = Graphformer(args).to(device)  # Construct or transform graph topology, geometry, or molecular feature data.
    optimizer = torch.optim.Adam(  # Bind this name to an intermediate value, configuration setting, or result.
        model.parameters(),  # Perform this step of the surrounding calculation or control-flow block.
        lr=args.learning_rate,  # Bind this name to an intermediate value, configuration setting, or result.
        weight_decay=args.weight_decay,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Bind this name to an intermediate value, configuration setting, or result.
        optimizer,  # Perform this step of the surrounding calculation or control-flow block.
        mode="min",
        factor=0.5,  # Bind this name to an intermediate value, configuration setting, or result.
        patience=args.lr_patience,  # Bind this name to an intermediate value, configuration setting, or result.
        min_lr=args.min_lr,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    criterion = nn.HuberLoss(delta=1.0)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    output_dir = (  # Bind this name to an intermediate value, configuration setting, or result.
        Path(args.output_root).resolve()  # Perform this step of the surrounding calculation or control-flow block.
        / args.split_method  # Perform this step of the surrounding calculation or control-flow block.
        / f"seed_{args.seed}"  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    output_dir.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    checkpoint_path = output_dir / "best_openbind_mgt.pt"

    history = []  # Bind this name to an intermediate value, configuration setting, or result.
    best_validation = float("inf")
    best_epoch = 0  # Bind this name to an intermediate value, configuration setting, or result.
    epochs_without_improvement = 0  # Bind this name to an intermediate value, configuration setting, or result.
    training_start = time.perf_counter()  # Bind this name to an intermediate value, configuration setting, or result.
    # Optimize until the epoch limit or validation early-stopping criterion.
    for epoch in range(1, args.max_epochs + 1):  # Iterate over the stated records, layers, batches, or graph elements.
        train_loss = train_epoch(  # Compute or store a loss, error, residual, or regression evaluation statistic.
            model,  # Perform this step of the surrounding calculation or control-flow block.
            loaders["train"],
            optimizer,  # Perform this step of the surrounding calculation or control-flow block.
            criterion,  # Perform this step of the surrounding calculation or control-flow block.
            target_mean,  # Perform this step of the surrounding calculation or control-flow block.
            target_scale,  # Perform this step of the surrounding calculation or control-flow block.
            device,  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        val_observed, val_predicted, _ = predict(  # Bind this name to an intermediate value, configuration setting, or result.
            model,  # Perform this step of the surrounding calculation or control-flow block.
            loaders["validation"],
            target_mean,  # Perform this step of the surrounding calculation or control-flow block.
            target_scale,  # Perform this step of the surrounding calculation or control-flow block.
            device,  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        val_normalized_truth = (  # Bind this name to an intermediate value, configuration setting, or result.
            torch.from_numpy(val_observed) - target_mean  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        ) / target_scale  # Continue or close the surrounding multiline expression or collection.
        val_normalized_prediction = (  # Bind this name to an intermediate value, configuration setting, or result.
            torch.from_numpy(val_predicted) - target_mean  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        ) / target_scale  # Continue or close the surrounding multiline expression or collection.
        validation_loss = float(  # Compute or store a loss, error, residual, or regression evaluation statistic.
            criterion(  # Perform this step of the surrounding calculation or control-flow block.
                val_normalized_prediction, val_normalized_truth  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
        scheduler.step(validation_loss)  # Advance the optimizer or learning-rate scheduler by one update.
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(  # Perform this step of the surrounding calculation or control-flow block.
            {  # Continue or close the surrounding multiline expression or collection.
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
        print(  # Report progress, predictions, or metrics to the selected output/logging backend.
            f"epoch={epoch} train={train_loss:.6f} "  # Compute or store a loss, error, residual, or regression evaluation statistic.
            f"validation={validation_loss:.6f} lr={learning_rate:.2e}",  # Compute or store a loss, error, residual, or regression evaluation statistic.
            flush=True,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        if validation_loss < best_validation:  # Evaluate this condition before executing the associated branch.
            best_validation = validation_loss  # Compute or store a loss, error, residual, or regression evaluation statistic.
            best_epoch = epoch  # Bind this name to an intermediate value, configuration setting, or result.
            epochs_without_improvement = 0  # Bind this name to an intermediate value, configuration setting, or result.
            torch.save(  # Persist this result or artifact at the configured output path.
                {  # Continue or close the surrounding multiline expression or collection.
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_scale": target_scale,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation,
                },  # Continue or close the surrounding multiline expression or collection.
                checkpoint_path,  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
        else:  # Handle the remaining case not covered by earlier conditions.
            epochs_without_improvement += 1  # Bind this name to an intermediate value, configuration setting, or result.
            if epochs_without_improvement >= args.early_stopping_patience:  # Evaluate this condition before executing the associated branch.
                break  # Alter loop control for this condition.
    training_seconds = time.perf_counter() - training_start  # Bind this name to an intermediate value, configuration setting, or result.

    # Restore the single checkpoint selected without consulting test labels.
    checkpoint = torch.load(  # Bind this name to an intermediate value, configuration setting, or result.
        checkpoint_path, map_location=device, weights_only=True  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    model.load_state_dict(checkpoint["model_state_dict"])
    split_predictions = {}  # Prepare dataset membership or batched data access for the experiment.
    split_metrics = {}  # Prepare dataset membership or batched data access for the experiment.
    test_start = time.perf_counter()  # Bind this name to an intermediate value, configuration setting, or result.
    for split in ("train", "validation", "test"):
        observed, predicted, identifiers = predict(  # Bind this name to an intermediate value, configuration setting, or result.
            model,  # Perform this step of the surrounding calculation or control-flow block.
            loaders[split],  # Perform this step of the surrounding calculation or control-flow block.
            target_mean,  # Perform this step of the surrounding calculation or control-flow block.
            target_scale,  # Perform this step of the surrounding calculation or control-flow block.
            device,  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        compound_rows = compound_predictions(  # Bind this name to an intermediate value, configuration setting, or result.
            observed, predicted, identifiers, compound_by_id  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        compound_observed = np.array(  # Bind this name to an intermediate value, configuration setting, or result.
            [row["experimental_pKD"] for row in compound_rows]
        )  # Continue or close the surrounding multiline expression or collection.
        compound_predicted = np.array(  # Bind this name to an intermediate value, configuration setting, or result.
            [row["predicted_pKD"] for row in compound_rows]
        )  # Continue or close the surrounding multiline expression or collection.
        split_predictions[split] = (  # Prepare dataset membership or batched data access for the experiment.
            observed,  # Perform this step of the surrounding calculation or control-flow block.
            predicted,  # Perform this step of the surrounding calculation or control-flow block.
            identifiers,  # Perform this step of the surrounding calculation or control-flow block.
            compound_rows,  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        split_metrics[split] = {  # Prepare dataset membership or batched data access for the experiment.
            "structure_level": regression_metrics(observed, predicted),
            "compound_level": regression_metrics(
                compound_observed, compound_predicted  # Perform this step of the surrounding calculation or control-flow block.
            ),  # Continue or close the surrounding multiline expression or collection.
        }  # Continue or close the surrounding multiline expression or collection.
    test_inference_seconds = time.perf_counter() - test_start  # Bind this name to an intermediate value, configuration setting, or result.

    test_observed, test_predicted, test_ids, test_compounds = (  # Bind this name to an intermediate value, configuration setting, or result.
        split_predictions["test"]
    )  # Continue or close the surrounding multiline expression or collection.
    write_prediction_files(  # Perform this step of the surrounding calculation or control-flow block.
        output_dir,  # Perform this step of the surrounding calculation or control-flow block.
        test_observed,  # Perform this step of the surrounding calculation or control-flow block.
        test_predicted,  # Perform this step of the surrounding calculation or control-flow block.
        test_ids,  # Perform this step of the surrounding calculation or control-flow block.
        test_compounds,  # Perform this step of the surrounding calculation or control-flow block.
        compound_by_id,  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    with (output_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))  # Bind this name to an intermediate value, configuration setting, or result.
        writer.writeheader()  # Perform this step of the surrounding calculation or control-flow block.
        writer.writerows(history)  # Perform this step of the surrounding calculation or control-flow block.

    parameter_count = sum(  # Bind this name to an intermediate value, configuration setting, or result.
        parameter.numel() for parameter in model.parameters()  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    result = {  # Bind this name to an intermediate value, configuration setting, or result.
        "model": "OpenBind complete original MGT Graphformer",
        "split_method": args.split_method,
        "seed": args.seed,
        "device": str(device),
        "counts": {
            split: len(subset) for split, subset in subsets.items()  # Perform this step of the surrounding calculation or control-flow block.
        },  # Continue or close the surrounding multiline expression or collection.
        "architecture": {
            "source": "original model.graphformer.Graphformer",
            "encoder_blocks": args.num_layers,
            "coulomb_attention_layers_per_encoder": args.n_mha,
            "alignn_layers_per_encoder": args.n_alignn,
            "post_alignn_gnn_layers_per_encoder": args.n_gnn,
            "attention_heads": args.n_heads,
            "hidden_dim": args.hidden_dims,
            "embedding_dim": args.embedding_dims,
            "laplacian_positional_encoding_dim": args.num_pe_fea,
            "distance_rbf_bins": args.num_edge_bins,
            "angle_rbf_bins": args.num_angle_bins,
            "coulomb_bins": args.num_clmb_bins,
            "local_radius_angstrom": args.local_radius,
            "maximum_local_neighbors": 12,
            "residual": args.residual,
            "global_pooling": "DGL average pooling",
            "output_dimension": 1,
            "parameter_count": parameter_count,
        },  # Continue or close the surrounding multiline expression or collection.
        "training": {
            "optimizer": "Adam",
            "loss": "Huber(delta=1.0)",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "target_mean": target_mean,
            "target_scale": target_scale,
            "training_seconds": training_seconds,
        },  # Continue or close the surrounding multiline expression or collection.
        "evaluation": {
            "attention_dropout_disabled_in_eval": True,
            "test_inference_seconds_all_levels": test_inference_seconds,
            "compound_aggregation": "mean prediction across repeated structures",
        },  # Continue or close the surrounding multiline expression or collection.
        "metrics": split_metrics,
        "data": {
            "root": str(data_root),
            "split_manifest": str(split_path),
            "representation": "quality-controlled crystallographic ligand PDB",
            "graphs": "frozen original-MGT processed graph triplets",
        },  # Continue or close the surrounding multiline expression or collection.
    }  # Continue or close the surrounding multiline expression or collection.
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        json.dump(result, handle, indent=2, sort_keys=True)  # Bind this name to an intermediate value, configuration setting, or result.
        handle.write("\n")
    return result  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: main — see its docstring and inline comments.
def main() -> None:  # Define this callable; its indented block implements the documented operation.
    """Parse experiment settings and train one original MGT model."""
    parser = argparse.ArgumentParser(  # Bind this name to an intermediate value, configuration setting, or result.
        description="Train complete original MGT on OpenBind ligand PDBs."
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--split_method", choices=["random", "scaffold"], required=True
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--local_radius", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--num_pe_fea", type=int, default=10)
    parser.add_argument("--num_edge_bins", type=int, default=80)
    parser.add_argument("--num_angle_bins", type=int, default=40)
    parser.add_argument("--num_clmb_bins", type=int, default=120)
    parser.add_argument("--embedding_dims", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--n_mha", type=int, default=1)
    parser.add_argument("--n_alignn", type=int, default=3)
    parser.add_argument("--n_gnn", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--residual", type=int, choices=[0, 1], default=1)
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--output_root",
        default=str(PROJECT_ROOT / "output" / "openbind_full_mgt"),
    )  # Continue or close the surrounding multiline expression or collection.
    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.
    result = train(args)  # Bind this name to an intermediate value, configuration setting, or result.
    print(json.dumps(result, indent=2, sort_keys=True))  # Report progress, predictions, or metrics to the selected output/logging backend.


if __name__ == "__main__":
    main()  # Perform this step of the surrounding calculation or control-flow block.
