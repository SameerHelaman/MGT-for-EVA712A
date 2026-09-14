# =============================================================================
# MODULE: train_openbind_morgan_mlp.py
# PURPOSE: Morgan-fingerprint multilayer-perceptron baseline and matched evaluation pipeline.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Matched baseline history, metrics, predictions and checkpoint.
# CALCULATIONS: Morgan radius-2, 2048-bit fingerprints feed a dense regressor; training/evaluation follows the matched normalized-Huber protocol.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Train and evaluate one model-specific baseline on frozen OpenBind splits.

This standalone module contains its dataset preparation, model construction,
matched optimisation, validation, compound aggregation, metric calculation and
output-writing workflow. Functions are documented individually and operation-
level comments explain the scientific and computational stages.
"""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import argparse  # Load a standard-library, scientific, or local project dependency.
import csv  # Load a standard-library, scientific, or local project dependency.
import json  # Load a standard-library, scientific, or local project dependency.
import math  # Load a standard-library, scientific, or local project dependency.
import time  # Load a standard-library, scientific, or local project dependency.
from pathlib import Path  # Import selected classes or functions from the named dependency.

import numpy as np  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
from rdkit import Chem, DataStructs  # Import selected classes or functions from the named dependency.
from rdkit.Chem import rdFingerprintGenerator  # Import selected classes or functions from the named dependency.
from torch import nn  # Import selected classes or functions from the named dependency.
from torch.utils.data import DataLoader, Dataset, Subset  # Import selected classes or functions from the named dependency.

from train_openbind_mgt import (  # Import selected classes or functions from the named dependency.
    compound_predictions,  # Perform this step of the surrounding calculation or control-flow block.
    read_csv,  # Perform this step of the surrounding calculation or control-flow block.
    regression_metrics,  # Perform this step of the surrounding calculation or control-flow block.
    resolve_device,  # Perform this step of the surrounding calculation or control-flow block.
    seed_everything,  # Perform this step of the surrounding calculation or control-flow block.
)  # Continue or close the surrounding multiline expression or collection.
from utils.openbind_ligand_dataset import DEFAULT_DATA_ROOT  # Import selected classes or functions from the named dependency.


PROJECT_ROOT = Path(__file__).resolve().parent  # Bind this name to an intermediate value, configuration setting, or result.
MODEL_LABELS = {"morgan_mlp": "Morgan fingerprint MLP"}


# CLASS: MorganMLP — reusable model/data abstraction.
class MorganMLP(nn.Module):  # Define this reusable class and its inheritance contract.
    """The matched radius-2, 2048-bit Morgan fingerprint baseline."""

    def __init__(self) -> None:  # Define this callable; its indented block implements the documented operation.
        """Initialize this object and its required state."""
        super().__init__()  # Initialize or delegate to the parent class implementation.
        self.network = nn.Sequential(  # Store this configuration value or neural-network submodule on the instance.
            nn.Linear(2048, 128),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),  # Bind this name to an intermediate value, configuration setting, or result.
            nn.ReLU(),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
            nn.Dropout(0.4),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
            nn.Linear(128, 64),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),  # Bind this name to an intermediate value, configuration setting, or result.
            nn.ReLU(),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
            nn.Dropout(0.4),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
            nn.Linear(64, 1),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        )  # Continue or close the surrounding multiline expression or collection.
        for module in self.modules():  # Iterate over the stated records, layers, batches, or graph elements.
            if isinstance(module, nn.Linear):  # Evaluate this condition before executing the associated branch.
                nn.init.xavier_uniform_(module.weight)  # Invoke the relevant tensor, graph, numerical, or tabular operation.
                nn.init.zeros_(module.bias)  # Invoke the relevant tensor, graph, numerical, or tabular operation.

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # Define this callable; its indented block implements the documented operation.
        """Apply this module to its input tensors or graphs."""
        return self.network(features).squeeze(1)  # Return this computed tensor, metric, object, or collection to the caller.


# CLASS: OpenBindFingerprintDataset — reusable model/data abstraction.
class OpenBindFingerprintDataset(Dataset):  # Define this reusable class and its inheritance contract.
    """Create fixed Morgan fingerprints for every curated structure record."""

    def __init__(self, data_root: str | Path) -> None:  # Define this callable; its indented block implements the documented operation.
        """Initialize this object and its required state."""
        path = (  # Bind this name to an intermediate value, configuration setting, or result.
            Path(data_root).resolve()  # Perform this step of the surrounding calculation or control-flow block.
            / "curated"
            / "openbind_ligand_structures.csv"
        )  # Continue or close the surrounding multiline expression or collection.
        self.rows = read_csv(path)  # Store this configuration value or neural-network submodule on the instance.
        self.sample_ids = [row["complex_name"] for row in self.rows]
        generator = rdFingerprintGenerator.GetMorganGenerator(  # Bind this name to an intermediate value, configuration setting, or result.
            radius=2, fpSize=2048  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        features = []  # Construct or transform graph topology, geometry, or molecular feature data.
        for row in self.rows:  # Iterate over the stated records, layers, batches, or graph elements.
            molecule = Chem.MolFromSmiles(row["canonical_smiles"])
            if molecule is None:  # Evaluate this condition before executing the associated branch.
                raise ValueError(  # Reject invalid input or state with an explicit exception.
                    f"Invalid canonical SMILES for {row['complex_name']}"
                )  # Continue or close the surrounding multiline expression or collection.
            bit_vector = generator.GetFingerprint(molecule)  # Bind this name to an intermediate value, configuration setting, or result.
            values = np.zeros(2048, dtype=np.float32)  # Bind this name to an intermediate value, configuration setting, or result.
            DataStructs.ConvertToNumpyArray(bit_vector, values)  # Perform this step of the surrounding calculation or control-flow block.
            features.append(values)  # Perform this step of the surrounding calculation or control-flow block.
        self.features = torch.from_numpy(np.stack(features))  # Store this configuration value or neural-network submodule on the instance.

    def __len__(self) -> int:  # Define this callable; its indented block implements the documented operation.
        """Return the number of dataset records."""
        return len(self.rows)  # Return this computed tensor, metric, object, or collection to the caller.

    def __getitem__(self, index: int):  # Define this callable; its indented block implements the documented operation.
        """Load and return one indexed dataset record."""
        row = self.rows[index]  # Bind this name to an intermediate value, configuration setting, or result.
        return (  # Return this computed tensor, metric, object, or collection to the caller.
            self.features[index],  # Perform this step of the surrounding calculation or control-flow block.
            torch.tensor(float(row["experimental_pKD"]), dtype=torch.float32),
            row["complex_name"],
        )  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: build_subsets — see its docstring and inline comments.
def build_subsets(dataset, split_rows):  # Define this callable; its indented block implements the documented operation.
    """Apply exact frozen split membership to one baseline dataset."""
    id_to_index = {  # Bind this name to an intermediate value, configuration setting, or result.
        sample_id: index  # Perform this step of the surrounding calculation or control-flow block.
        for index, sample_id in enumerate(dataset.sample_ids)  # Iterate over the stated records, layers, batches, or graph elements.
    }  # Continue or close the surrounding multiline expression or collection.
    if set(id_to_index) != {row["complex_name"] for row in split_rows}:
        raise ValueError("Dataset and split manifest coverage differ")
    compound_by_id = {  # Bind this name to an intermediate value, configuration setting, or result.
        row["complex_name"]: row["official_compound_group_id"]
        for row in split_rows  # Iterate over the stated records, layers, batches, or graph elements.
    }  # Continue or close the surrounding multiline expression or collection.
    subsets = {  # Prepare dataset membership or batched data access for the experiment.
        split: Subset(  # Perform this step of the surrounding calculation or control-flow block.
            dataset,  # Perform this step of the surrounding calculation or control-flow block.
            [  # Continue or close the surrounding multiline expression or collection.
                id_to_index[row["complex_name"]]
                for row in split_rows  # Iterate over the stated records, layers, batches, or graph elements.
                if row["split"] == split
            ],  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
        for split in ("train", "validation", "test")
    }  # Continue or close the surrounding multiline expression or collection.
    for compound in set(compound_by_id.values()):  # Iterate over the stated records, layers, batches, or graph elements.
        memberships = {  # Bind this name to an intermediate value, configuration setting, or result.
            row["split"]
            for row in split_rows  # Iterate over the stated records, layers, batches, or graph elements.
            if row["official_compound_group_id"] == compound
        }  # Continue or close the surrounding multiline expression or collection.
        if len(memberships) != 1:  # Evaluate this condition before executing the associated branch.
            raise ValueError(f"Compound leakage detected for {compound}")  # Reject invalid input or state with an explicit exception.
    return subsets, compound_by_id  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: make_loaders — see its docstring and inline comments.
def make_loaders(dataset, subsets, args):  # Define this callable; its indented block implements the documented operation.
    """Create deterministic tensor or graph loaders."""
    loader_type = DataLoader  # Prepare dataset membership or batched data access for the experiment.
    generator = torch.Generator().manual_seed(args.seed)  # Bind this name to an intermediate value, configuration setting, or result.
    return {  # Return this computed tensor, metric, object, or collection to the caller.
        split: loader_type(  # Perform this step of the surrounding calculation or control-flow block.
            subset,  # Perform this step of the surrounding calculation or control-flow block.
            batch_size=args.batch_size,  # Bind this name to an intermediate value, configuration setting, or result.
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=0,  # Bind this name to an intermediate value, configuration setting, or result.
            drop_last=False,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        for split, subset in subsets.items()  # Iterate over the stated records, layers, batches, or graph elements.
    }  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: subset_targets — see its docstring and inline comments.
def subset_targets(dataset, subset: Subset) -> np.ndarray:  # Define this callable; its indented block implements the documented operation.
    """Read labels without constructing graphs."""
    return np.asarray(  # Return this computed tensor, metric, object, or collection to the caller.
        [  # Continue or close the surrounding multiline expression or collection.
            float(dataset.rows[index]["experimental_pKD"])
            for index in subset.indices  # Iterate over the stated records, layers, batches, or graph elements.
        ],  # Continue or close the surrounding multiline expression or collection.
        dtype=np.float32,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: build_model — see its docstring and inline comments.
def build_model(args):  # Define this callable; its indented block implements the documented operation.
    """Construct the fixed Morgan fingerprint multilayer perceptron."""
    return MorganMLP()  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: move_inputs — see its docstring and inline comments.
def move_inputs(inputs, device):  # Define this callable; its indented block implements the documented operation.
    """Move either a feature tensor or DGL graph to the accelerator."""
    return inputs.to(device)  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: train_epoch — see its docstring and inline comments.
def train_epoch(  # Define this callable; its indented block implements the documented operation.
    model,  # Perform this step of the surrounding calculation or control-flow block.
    loader,  # Perform this step of the surrounding calculation or control-flow block.
    optimizer,  # Perform this step of the surrounding calculation or control-flow block.
    criterion,  # Perform this step of the surrounding calculation or control-flow block.
    target_mean,  # Perform this step of the surrounding calculation or control-flow block.
    target_scale,  # Perform this step of the surrounding calculation or control-flow block.
    device,  # Perform this step of the surrounding calculation or control-flow block.
):  # Continue or close the surrounding multiline expression or collection.
    """Train one baseline for one epoch."""
    model.train()  # Switch the model to training behaviour.
    total = 0.0  # Bind this name to an intermediate value, configuration setting, or result.
    count = 0  # Bind this name to an intermediate value, configuration setting, or result.
    for inputs, targets, _ in loader:  # Iterate over the stated records, layers, batches, or graph elements.
        inputs = move_inputs(inputs, device)  # Bind this name to an intermediate value, configuration setting, or result.
        targets = ((targets - target_mean) / target_scale).to(device)  # Bind this name to an intermediate value, configuration setting, or result.
        optimizer.zero_grad(set_to_none=True)  # Clear accumulated gradients before the next optimization update.
        predictions = model(inputs)  # Create or apply a trainable neural-network component.
        loss = criterion(predictions, targets)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        if not torch.isfinite(loss):  # Evaluate this condition before executing the associated branch.
            raise FloatingPointError("Non-finite training loss")
        loss.backward()  # Backpropagate the loss to compute gradients for trainable parameters.
        optimizer.step()  # Advance the optimizer or learning-rate scheduler by one update.
        total += float(loss.detach()) * targets.shape[0]  # Compute or store a loss, error, residual, or regression evaluation statistic.
        count += targets.shape[0]  # Bind this name to an intermediate value, configuration setting, or result.
    return total / count  # Return this computed tensor, metric, object, or collection to the caller.


@torch.no_grad()  # Apply this decorator to the following class or function.
# FUNCTION: predict — see its docstring and inline comments.
def predict(model, loader, target_mean, target_scale, device):  # Define this callable; its indented block implements the documented operation.
    """Predict original-scale pKD for one partition."""
    model.eval()  # Switch the model to deterministic evaluation behaviour.
    truths = []  # Bind this name to an intermediate value, configuration setting, or result.
    predictions = []  # Bind this name to an intermediate value, configuration setting, or result.
    identifiers = []  # Bind this name to an intermediate value, configuration setting, or result.
    for inputs, targets, ids in loader:  # Iterate over the stated records, layers, batches, or graph elements.
        outputs = model(move_inputs(inputs, device))  # Create or apply a trainable neural-network component.
        outputs = outputs * target_scale + target_mean  # Bind this name to an intermediate value, configuration setting, or result.
        truths.append(targets.numpy())  # Perform this step of the surrounding calculation or control-flow block.
        predictions.append(outputs.cpu().numpy())  # Perform this step of the surrounding calculation or control-flow block.
        identifiers.extend(list(ids))  # Perform this step of the surrounding calculation or control-flow block.
    return (  # Return this computed tensor, metric, object, or collection to the caller.
        np.concatenate(truths),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        np.concatenate(predictions),  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        identifiers,  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: validation_loss — see its docstring and inline comments.
def validation_loss(  # Define this callable; its indented block implements the documented operation.
    model, loader, criterion, target_mean, target_scale, device  # Perform this step of the surrounding calculation or control-flow block.
):  # Continue or close the surrounding multiline expression or collection.
    """Calculate sample-weighted normalized validation Huber loss."""
    model.eval()  # Switch the model to deterministic evaluation behaviour.
    total = 0.0  # Bind this name to an intermediate value, configuration setting, or result.
    count = 0  # Bind this name to an intermediate value, configuration setting, or result.
    with torch.no_grad():  # Enter a managed context so resources and graph state are cleaned up safely.
        for inputs, targets, _ in loader:  # Iterate over the stated records, layers, batches, or graph elements.
            targets = ((targets - target_mean) / target_scale).to(device)  # Bind this name to an intermediate value, configuration setting, or result.
            outputs = model(move_inputs(inputs, device))  # Create or apply a trainable neural-network component.
            loss = criterion(outputs, targets)  # Compute or store a loss, error, residual, or regression evaluation statistic.
            total += float(loss) * targets.shape[0]  # Compute or store a loss, error, residual, or regression evaluation statistic.
            count += targets.shape[0]  # Bind this name to an intermediate value, configuration setting, or result.
    return total / count  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: write_outputs — see its docstring and inline comments.
def write_outputs(  # Define this callable; its indented block implements the documented operation.
    output_dir,  # Perform this step of the surrounding calculation or control-flow block.
    history,  # Perform this step of the surrounding calculation or control-flow block.
    test_values,  # Perform this step of the surrounding calculation or control-flow block.
    compound_rows,  # Perform this step of the surrounding calculation or control-flow block.
    compound_by_id,  # Perform this step of the surrounding calculation or control-flow block.
):  # Continue or close the surrounding multiline expression or collection.
    """Write histories and both held-out prediction levels."""
    with (output_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))  # Bind this name to an intermediate value, configuration setting, or result.
        writer.writeheader()  # Perform this step of the surrounding calculation or control-flow block.
        writer.writerows(history)  # Perform this step of the surrounding calculation or control-flow block.
    observed, predicted, identifiers = test_values  # Bind this name to an intermediate value, configuration setting, or result.
    with (output_dir / "test_structure_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
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
    with (output_dir / "test_compound_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        writer = csv.DictWriter(  # Bind this name to an intermediate value, configuration setting, or result.
            handle, fieldnames=list(compound_rows[0])  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        writer.writeheader()  # Perform this step of the surrounding calculation or control-flow block.
        writer.writerows(compound_rows)  # Perform this step of the surrounding calculation or control-flow block.


# FUNCTION: architecture_report — see its docstring and inline comments.
def architecture_report(args, parameter_count):  # Define this callable; its indented block implements the documented operation.
    """Record the exact fingerprint and MLP configuration."""
    return {  # Return this computed tensor, metric, object, or collection to the caller.
        "parameter_count": parameter_count,
        "fingerprint": "Morgan radius 2, 2048 bits",
        "network": "2048-128-64-1 MLP",
        "coordinates": False,
        "crystallographic_coordinates": False,
        "distance_rbf": False,
        "angular_processing": False,
        "coulomb_attention": False,
    }  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: train — see its docstring and inline comments.
def train(args):  # Define this callable; its indented block implements the documented operation.
    """Run one complete matched OpenBind baseline experiment."""
    # Fix all supported RNGs so initialization, dropout and shuffling reproduce.
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
    # Load the immutable compound-aware membership manifest.
    split_rows = read_csv(split_path)  # Prepare dataset membership or batched data access for the experiment.
    if {row["split_method"] for row in split_rows} != {
        args.split_method  # Perform this step of the surrounding calculation or control-flow block.
    }:  # Continue or close the surrounding multiline expression or collection.
        raise ValueError("Split method mismatch")
    if {row["seed"] for row in split_rows} != {str(args.seed)}:
        raise ValueError("Split seed mismatch")
    # Select either fixed fingerprints or the requested controlled graph view.
    dataset = OpenBindFingerprintDataset(data_root)  # Prepare dataset membership or batched data access for the experiment.
    subsets, compound_by_id = build_subsets(dataset, split_rows)  # Prepare dataset membership or batched data access for the experiment.
    loaders = make_loaders(dataset, subsets, args)  # Prepare dataset membership or batched data access for the experiment.
    # Estimate normalization from training labels only.
    training_targets = subset_targets(dataset, subsets["train"])
    target_mean = float(training_targets.mean())  # Bind this name to an intermediate value, configuration setting, or result.
    target_scale = float(training_targets.std(ddof=0))  # Bind this name to an intermediate value, configuration setting, or result.
    # Construct exactly one architecture from the controlled ablation series.
    model = build_model(args).to(device)  # Create or apply a trainable neural-network component.
    parameter_count = sum(  # Bind this name to an intermediate value, configuration setting, or result.
        parameter.numel() for parameter in model.parameters()  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    optimizer = torch.optim.Adam(  # Bind this name to an intermediate value, configuration setting, or result.
        model.parameters(),  # Perform this step of the surrounding calculation or control-flow block.
        lr=args.learning_rate,  # Bind this name to an intermediate value, configuration setting, or result.
        weight_decay=args.weight_decay,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    criterion = nn.HuberLoss(delta=1.0)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Bind this name to an intermediate value, configuration setting, or result.
        optimizer,  # Perform this step of the surrounding calculation or control-flow block.
        mode="min",
        factor=0.5,  # Bind this name to an intermediate value, configuration setting, or result.
        patience=args.lr_patience,  # Bind this name to an intermediate value, configuration setting, or result.
        min_lr=args.min_lr,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    output_dir = (  # Bind this name to an intermediate value, configuration setting, or result.
        Path(args.output_root).resolve()  # Perform this step of the surrounding calculation or control-flow block.
        / args.model  # Perform this step of the surrounding calculation or control-flow block.
        / args.split_method  # Perform this step of the surrounding calculation or control-flow block.
        / f"seed_{args.seed}"  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    output_dir.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    checkpoint_path = output_dir / f"best_{args.model}.pt"  # Create or apply a trainable neural-network component.
    best_validation = math.inf  # Bind this name to an intermediate value, configuration setting, or result.
    best_epoch = 0  # Bind this name to an intermediate value, configuration setting, or result.
    stale_epochs = 0  # Bind this name to an intermediate value, configuration setting, or result.
    history = []  # Bind this name to an intermediate value, configuration setting, or result.
    start = time.perf_counter()  # Bind this name to an intermediate value, configuration setting, or result.
    # Train with validation-based checkpointing and patience-based stopping.
    for epoch in range(1, args.max_epochs + 1):  # Iterate over the stated records, layers, batches, or graph elements.
        training_loss = train_epoch(  # Compute or store a loss, error, residual, or regression evaluation statistic.
            model,  # Perform this step of the surrounding calculation or control-flow block.
            loaders["train"],
            optimizer,  # Perform this step of the surrounding calculation or control-flow block.
            criterion,  # Perform this step of the surrounding calculation or control-flow block.
            target_mean,  # Perform this step of the surrounding calculation or control-flow block.
            target_scale,  # Perform this step of the surrounding calculation or control-flow block.
            device,  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        current_validation = validation_loss(  # Compute or store a loss, error, residual, or regression evaluation statistic.
            model,  # Perform this step of the surrounding calculation or control-flow block.
            loaders["validation"],
            criterion,  # Perform this step of the surrounding calculation or control-flow block.
            target_mean,  # Perform this step of the surrounding calculation or control-flow block.
            target_scale,  # Perform this step of the surrounding calculation or control-flow block.
            device,  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        scheduler.step(current_validation)  # Advance the optimizer or learning-rate scheduler by one update.
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(  # Perform this step of the surrounding calculation or control-flow block.
            {  # Continue or close the surrounding multiline expression or collection.
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": current_validation,
                "learning_rate": learning_rate,
            }  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
        print(  # Report progress, predictions, or metrics to the selected output/logging backend.
            f"epoch={epoch} train={training_loss:.6f} "  # Compute or store a loss, error, residual, or regression evaluation statistic.
            f"validation={current_validation:.6f} "  # Bind this name to an intermediate value, configuration setting, or result.
            f"lr={learning_rate:.2e}",  # Bind this name to an intermediate value, configuration setting, or result.
            flush=True,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        if current_validation < best_validation:  # Evaluate this condition before executing the associated branch.
            best_validation = current_validation  # Bind this name to an intermediate value, configuration setting, or result.
            best_epoch = epoch  # Bind this name to an intermediate value, configuration setting, or result.
            stale_epochs = 0  # Bind this name to an intermediate value, configuration setting, or result.
            torch.save(model.state_dict(), checkpoint_path)  # Persist this result or artifact at the configured output path.
        else:  # Handle the remaining case not covered by earlier conditions.
            stale_epochs += 1  # Bind this name to an intermediate value, configuration setting, or result.
            if stale_epochs >= args.early_stopping_patience:  # Evaluate this condition before executing the associated branch.
                break  # Alter loop control for this condition.
    training_seconds = time.perf_counter() - start  # Bind this name to an intermediate value, configuration setting, or result.
    # Restore the lowest-validation-loss state before reporting any metrics.
    model.load_state_dict(  # Perform this step of the surrounding calculation or control-flow block.
        torch.load(checkpoint_path, map_location=device, weights_only=True)  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    metrics = {}  # Bind this name to an intermediate value, configuration setting, or result.
    saved_predictions = {}  # Bind this name to an intermediate value, configuration setting, or result.
    inference_start = time.perf_counter()  # Bind this name to an intermediate value, configuration setting, or result.
    # Apply the same metric and compound-aggregation code to every partition.
    for split in ("train", "validation", "test"):
        observed, predicted, identifiers = predict(  # Bind this name to an intermediate value, configuration setting, or result.
            model,  # Perform this step of the surrounding calculation or control-flow block.
            loaders[split],  # Perform this step of the surrounding calculation or control-flow block.
            target_mean,  # Perform this step of the surrounding calculation or control-flow block.
            target_scale,  # Perform this step of the surrounding calculation or control-flow block.
            device,  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        compounds = compound_predictions(  # Bind this name to an intermediate value, configuration setting, or result.
            observed, predicted, identifiers, compound_by_id  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
        compound_observed = np.asarray(  # Bind this name to an intermediate value, configuration setting, or result.
            [row["experimental_pKD"] for row in compounds]
        )  # Continue or close the surrounding multiline expression or collection.
        compound_predicted = np.asarray(  # Bind this name to an intermediate value, configuration setting, or result.
            [row["predicted_pKD"] for row in compounds]
        )  # Continue or close the surrounding multiline expression or collection.
        metrics[split] = {  # Prepare dataset membership or batched data access for the experiment.
            "structure_level": regression_metrics(observed, predicted),
            "compound_level": regression_metrics(
                compound_observed, compound_predicted  # Perform this step of the surrounding calculation or control-flow block.
            ),  # Continue or close the surrounding multiline expression or collection.
        }  # Continue or close the surrounding multiline expression or collection.
        saved_predictions[split] = (  # Prepare dataset membership or batched data access for the experiment.
            (observed, predicted, identifiers),  # Continue or close the surrounding multiline expression or collection.
            compounds,  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.
    inference_seconds = time.perf_counter() - inference_start  # Bind this name to an intermediate value, configuration setting, or result.
    write_outputs(  # Perform this step of the surrounding calculation or control-flow block.
        output_dir,  # Perform this step of the surrounding calculation or control-flow block.
        history,  # Perform this step of the surrounding calculation or control-flow block.
        saved_predictions["test"][0],
        saved_predictions["test"][1],
        compound_by_id,  # Perform this step of the surrounding calculation or control-flow block.
    )  # Continue or close the surrounding multiline expression or collection.
    result = {  # Bind this name to an intermediate value, configuration setting, or result.
        "model": MODEL_LABELS[args.model],
        "model_key": args.model,
        "split_method": args.split_method,
        "seed": args.seed,
        "device": str(device),
        "counts": {
            split: len(subset) for split, subset in subsets.items()  # Perform this step of the surrounding calculation or control-flow block.
        },  # Continue or close the surrounding multiline expression or collection.
        "architecture": architecture_report(args, parameter_count),
        "training": {
            "optimizer": "Adam",
            "loss": "Huber(delta=1.0)",
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "early_stopping_patience": args.early_stopping_patience,
            "training_seconds": training_seconds,
        },  # Continue or close the surrounding multiline expression or collection.
        "evaluation": {
            "compound_aggregation": "mean prediction across repeated structures",
            "inference_seconds_all_partitions": inference_seconds,
        },  # Continue or close the surrounding multiline expression or collection.
        "metrics": metrics,
        "data": {
            "root": str(data_root),
            "split_manifest": str(split_path),
            "coordinate_source": (
                "OpenBind ligand_ref.sdf crystallographic pose"
                if args.model in {"3d_gnn", "3d_alignn"}
                else None  # Handle the remaining case not covered by earlier conditions.
            ),  # Continue or close the surrounding multiline expression or collection.
        },  # Continue or close the surrounding multiline expression or collection.
    }  # Continue or close the surrounding multiline expression or collection.
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        json.dump(result, handle, indent=2, sort_keys=True)  # Bind this name to an intermediate value, configuration setting, or result.
        handle.write("\n")
    return result  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: build_parser — see its docstring and inline comments.
def build_parser(  # Define this callable; its indented block implements the documented operation.
    fixed_model: str | None = None,  # Create or apply a trainable neural-network component.
    description: str | None = None,  # Bind this name to an intermediate value, configuration setting, or result.
) -> argparse.ArgumentParser:  # Continue or close the surrounding multiline expression or collection.
    """Build the common CLI, optionally fixing it to one model architecture."""
    if fixed_model is not None and fixed_model not in MODEL_LABELS:  # Evaluate this condition before executing the associated branch.
        raise ValueError(f"Unknown baseline model: {fixed_model}")  # Reject invalid input or state with an explicit exception.
    parser = argparse.ArgumentParser(  # Bind this name to an intermediate value, configuration setting, or result.
        description=description or "Train a matched OpenBind ligand baseline."
    )  # Continue or close the surrounding multiline expression or collection.
    if fixed_model is None:  # Evaluate this condition before executing the associated branch.
        parser.add_argument(  # Register this command-line option, including its type, default, or help text.
            "--model",
            choices=list(MODEL_LABELS),  # Create or apply a trainable neural-network component.
            required=True,  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
    else:  # Handle the remaining case not covered by earlier conditions.
        parser.set_defaults(model=fixed_model)  # Create or apply a trainable neural-network component.
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
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--spatial_cutoff", type=float, default=5.0)
    parser.add_argument("--max_neighbors", type=int, default=32)
    parser.add_argument("--distance_bins", type=int, default=40)
    parser.add_argument("--angle_bins", type=int, default=40)
    parser.add_argument(  # Register this command-line option, including its type, default, or help text.
        "--output_root",
        default=str(PROJECT_ROOT / "output" / "openbind_baselines"),
    )  # Continue or close the surrounding multiline expression or collection.
    return parser  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: run_cli — see its docstring and inline comments.
def run_cli(  # Define this callable; its indented block implements the documented operation.
    fixed_model: str | None = None,  # Create or apply a trainable neural-network component.
    description: str | None = None,  # Bind this name to an intermediate value, configuration setting, or result.
) -> None:  # Continue or close the surrounding multiline expression or collection.
    """Parse arguments, run one baseline and print its complete result record."""
    parser = build_parser(fixed_model=fixed_model, description=description)  # Create or apply a trainable neural-network component.
    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.
    result = train(args)  # Bind this name to an intermediate value, configuration setting, or result.
    print(json.dumps(result, indent=2, sort_keys=True))  # Report progress, predictions, or metrics to the selected output/logging backend.


# FUNCTION: main — see its docstring and inline comments.
def main() -> None:  # Define this callable; its indented block implements the documented operation.
    """Run the fixed Morgan fingerprint model; no model selector is required."""
    run_cli(fixed_model="morgan_mlp")


if __name__ == "__main__":
    main()  # Perform this step of the surrounding calculation or control-flow block.
