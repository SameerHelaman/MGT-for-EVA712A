"""Train matched OpenBind fingerprint, 2D GNN, 3D GNN, or 3D ALIGNN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from model.ligand_3d_alignn import Ligand3DALIGNN
from model.ligand_3d_gnn import Ligand3DGNN
from model.ligand_gnn import Ligand2DGNN
from train_openbind_mgt import (
    compound_predictions,
    read_csv,
    regression_metrics,
    resolve_device,
    seed_everything,
)
from utils.molecular_features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from utils.openbind_ligand_dataset import (
    DEFAULT_DATA_ROOT,
    OpenBindGraphDataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_LABELS = {
    "morgan_mlp": "Morgan fingerprint MLP",
    "2d_gnn": "2D GNN",
    "3d_gnn": "Crystallographic 3D distance GNN",
    "3d_alignn": "Crystallographic 3D ALIGNN",
}


class MorganMLP(nn.Module):
    """The matched radius-2, 2048-bit Morgan fingerprint baseline."""

    def __init__(self) -> None:
        """Initialize this object and its required state."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply this module to its input tensors or graphs."""
        return self.network(features).squeeze(1)


class OpenBindFingerprintDataset(Dataset):
    """Create fixed Morgan fingerprints for every curated structure record."""

    def __init__(self, data_root: str | Path) -> None:
        """Initialize this object and its required state."""
        path = (
            Path(data_root).resolve()
            / "curated"
            / "openbind_ligand_structures.csv"
        )
        self.rows = read_csv(path)
        self.sample_ids = [row["complex_name"] for row in self.rows]
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )
        features = []
        for row in self.rows:
            molecule = Chem.MolFromSmiles(row["canonical_smiles"])
            if molecule is None:
                raise ValueError(
                    f"Invalid canonical SMILES for {row['complex_name']}"
                )
            bit_vector = generator.GetFingerprint(molecule)
            values = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bit_vector, values)
            features.append(values)
        self.features = torch.from_numpy(np.stack(features))

    def __len__(self) -> int:
        """Return the number of dataset records."""
        return len(self.rows)

    def __getitem__(self, index: int):
        """Load and return one indexed dataset record."""
        row = self.rows[index]
        return (
            self.features[index],
            torch.tensor(float(row["experimental_pKD"]), dtype=torch.float32),
            row["complex_name"],
        )


def build_subsets(dataset, split_rows):
    """Apply exact frozen split membership to one baseline dataset."""
    id_to_index = {
        sample_id: index
        for index, sample_id in enumerate(dataset.sample_ids)
    }
    if set(id_to_index) != {row["complex_name"] for row in split_rows}:
        raise ValueError("Dataset and split manifest coverage differ")
    compound_by_id = {
        row["complex_name"]: row["official_compound_group_id"]
        for row in split_rows
    }
    subsets = {
        split: Subset(
            dataset,
            [
                id_to_index[row["complex_name"]]
                for row in split_rows
                if row["split"] == split
            ],
        )
        for split in ("train", "validation", "test")
    }
    for compound in set(compound_by_id.values()):
        memberships = {
            row["split"]
            for row in split_rows
            if row["official_compound_group_id"] == compound
        }
        if len(memberships) != 1:
            raise ValueError(f"Compound leakage detected for {compound}")
    return subsets, compound_by_id


def make_loaders(dataset, subsets, args):
    """Create deterministic tensor or graph loaders."""
    loader_type = (
        DataLoader if args.model == "morgan_mlp" else GraphDataLoader
    )
    generator = torch.Generator().manual_seed(args.seed)
    return {
        split: loader_type(
            subset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=0,
            drop_last=False,
        )
        for split, subset in subsets.items()
    }


def subset_targets(dataset, subset: Subset) -> np.ndarray:
    """Read labels without constructing graphs."""
    return np.asarray(
        [
            float(dataset.rows[index]["experimental_pKD"])
            for index in subset.indices
        ],
        dtype=np.float32,
    )


def build_model(args):
    """Construct exactly one controlled baseline architecture."""
    if args.model == "morgan_mlp":
        return MorganMLP()
    common = {
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
    }
    if args.model == "2d_gnn":
        return Ligand2DGNN(**common)
    if args.model == "3d_gnn":
        return Ligand3DGNN(
            **common,
            distance_bins=args.distance_bins,
            spatial_cutoff=args.spatial_cutoff,
        )
    return Ligand3DALIGNN(
        **common,
        distance_bins=args.distance_bins,
        angle_bins=args.angle_bins,
        spatial_cutoff=args.spatial_cutoff,
    )


def move_inputs(inputs, device):
    """Move either a feature tensor or DGL graph to the accelerator."""
    return inputs.to(device)


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    target_mean,
    target_scale,
    device,
):
    """Train one baseline for one epoch."""
    model.train()
    total = 0.0
    count = 0
    for inputs, targets, _ in loader:
        inputs = move_inputs(inputs, device)
        targets = ((targets - target_mean) / target_scale).to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * targets.shape[0]
        count += targets.shape[0]
    return total / count


@torch.no_grad()
def predict(model, loader, target_mean, target_scale, device):
    """Predict original-scale pKD for one partition."""
    model.eval()
    truths = []
    predictions = []
    identifiers = []
    for inputs, targets, ids in loader:
        outputs = model(move_inputs(inputs, device))
        outputs = outputs * target_scale + target_mean
        truths.append(targets.numpy())
        predictions.append(outputs.cpu().numpy())
        identifiers.extend(list(ids))
    return (
        np.concatenate(truths),
        np.concatenate(predictions),
        identifiers,
    )


def validation_loss(
    model, loader, criterion, target_mean, target_scale, device
):
    """Calculate sample-weighted normalized validation Huber loss."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            targets = ((targets - target_mean) / target_scale).to(device)
            outputs = model(move_inputs(inputs, device))
            loss = criterion(outputs, targets)
            total += float(loss) * targets.shape[0]
            count += targets.shape[0]
    return total / count


def write_outputs(
    output_dir,
    history,
    test_values,
    compound_rows,
    compound_by_id,
):
    """Write histories and both held-out prediction levels."""
    with (output_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    observed, predicted, identifiers = test_values
    with (output_dir / "test_structure_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "complex_name",
            "official_compound_group_id",
            "experimental_pKD",
            "predicted_pKD",
            "residual",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for truth, estimate, complex_name in zip(
            observed, predicted, identifiers
        ):
            writer.writerow(
                {
                    "complex_name": complex_name,
                    "official_compound_group_id": compound_by_id[complex_name],
                    "experimental_pKD": float(truth),
                    "predicted_pKD": float(estimate),
                    "residual": float(estimate - truth),
                }
            )
    with (output_dir / "test_compound_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(compound_rows[0])
        )
        writer.writeheader()
        writer.writerows(compound_rows)


def architecture_report(args, parameter_count):
    """Record the controlled information included in each model."""
    report = {
        "parameter_count": parameter_count,
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "coordinates": args.model in {"3d_gnn", "3d_alignn"},
        "crystallographic_coordinates": args.model
        in {"3d_gnn", "3d_alignn"},
        "distance_rbf": args.model in {"3d_gnn", "3d_alignn"},
        "angular_processing": args.model == "3d_alignn",
        "coulomb_attention": False,
    }
    if args.model == "morgan_mlp":
        report.update(
            {
                "fingerprint": "Morgan radius 2, 2048 bits",
                "network": "2048-128-64-1 MLP",
            }
        )
    else:
        report.update(
            {
                "hidden_dim": args.hidden_dim,
                "embedding_dim": args.embedding_dim,
                "layers": args.num_layers,
                "spatial_cutoff_angstrom": (
                    args.spatial_cutoff
                    if args.model != "2d_gnn"
                    else None
                ),
                "maximum_spatial_neighbors": (
                    args.max_neighbors
                    if args.model != "2d_gnn"
                    else None
                ),
                "distance_rbf_bins": (
                    args.distance_bins
                    if args.model != "2d_gnn"
                    else None
                ),
                "angle_rbf_bins": (
                    args.angle_bins
                    if args.model == "3d_alignn"
                    else None
                ),
            }
        )
    return report


def train(args):
    """Run one complete matched OpenBind baseline experiment."""
    # Fix all supported RNGs so initialization, dropout and shuffling reproduce.
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data_root = Path(args.data_root).resolve()
    split_path = (
        Path(args.split_file).resolve()
        if args.split_file
        else data_root
        / "splits"
        / f"{args.split_method}_seed_{args.seed}.csv"
    )
    # Load the immutable compound-aware membership manifest.
    split_rows = read_csv(split_path)
    if {row["split_method"] for row in split_rows} != {
        args.split_method
    }:
        raise ValueError("Split method mismatch")
    if {row["seed"] for row in split_rows} != {str(args.seed)}:
        raise ValueError("Split seed mismatch")
    # Select either fixed fingerprints or the requested controlled graph view.
    dataset = (
        OpenBindFingerprintDataset(data_root)
        if args.model == "morgan_mlp"
        else OpenBindGraphDataset(
            data_root=data_root,
            mode=args.model,
            spatial_cutoff=args.spatial_cutoff,
            max_neighbors=args.max_neighbors,
        )
    )
    subsets, compound_by_id = build_subsets(dataset, split_rows)
    loaders = make_loaders(dataset, subsets, args)
    # Estimate normalization from training labels only.
    training_targets = subset_targets(dataset, subsets["train"])
    target_mean = float(training_targets.mean())
    target_scale = float(training_targets.std(ddof=0))
    # Construct exactly one architecture from the controlled ablation series.
    model = build_model(args).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    output_dir = (
        Path(args.output_root).resolve()
        / args.model
        / args.split_method
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"best_{args.model}.pt"
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    start = time.perf_counter()
    # Train with validation-based checkpointing and patience-based stopping.
    for epoch in range(1, args.max_epochs + 1):
        training_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            target_mean,
            target_scale,
            device,
        )
        current_validation = validation_loss(
            model,
            loaders["validation"],
            criterion,
            target_mean,
            target_scale,
            device,
        )
        scheduler.step(current_validation)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": current_validation,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"epoch={epoch} train={training_loss:.6f} "
            f"validation={current_validation:.6f} "
            f"lr={learning_rate:.2e}",
            flush=True,
        )
        if current_validation < best_validation:
            best_validation = current_validation
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                break
    training_seconds = time.perf_counter() - start
    # Restore the lowest-validation-loss state before reporting any metrics.
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    metrics = {}
    saved_predictions = {}
    inference_start = time.perf_counter()
    # Apply the same metric and compound-aggregation code to every partition.
    for split in ("train", "validation", "test"):
        observed, predicted, identifiers = predict(
            model,
            loaders[split],
            target_mean,
            target_scale,
            device,
        )
        compounds = compound_predictions(
            observed, predicted, identifiers, compound_by_id
        )
        compound_observed = np.asarray(
            [row["experimental_pKD"] for row in compounds]
        )
        compound_predicted = np.asarray(
            [row["predicted_pKD"] for row in compounds]
        )
        metrics[split] = {
            "structure_level": regression_metrics(observed, predicted),
            "compound_level": regression_metrics(
                compound_observed, compound_predicted
            ),
        }
        saved_predictions[split] = (
            (observed, predicted, identifiers),
            compounds,
        )
    inference_seconds = time.perf_counter() - inference_start
    write_outputs(
        output_dir,
        history,
        saved_predictions["test"][0],
        saved_predictions["test"][1],
        compound_by_id,
    )
    result = {
        "model": MODEL_LABELS[args.model],
        "model_key": args.model,
        "split_method": args.split_method,
        "seed": args.seed,
        "device": str(device),
        "counts": {
            split: len(subset) for split, subset in subsets.items()
        },
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
        },
        "evaluation": {
            "compound_aggregation": "mean prediction across repeated structures",
            "inference_seconds_all_partitions": inference_seconds,
        },
        "metrics": metrics,
        "data": {
            "root": str(data_root),
            "split_manifest": str(split_path),
            "coordinate_source": (
                "OpenBind ligand_ref.sdf crystallographic pose"
                if args.model in {"3d_gnn", "3d_alignn"}
                else None
            ),
        },
    }
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    """Parse one matched baseline experiment and print its complete result record."""
    parser = argparse.ArgumentParser(
        description="Train a matched OpenBind ligand baseline."
    )
    parser.add_argument(
        "--model",
        choices=["morgan_mlp", "2d_gnn", "3d_gnn", "3d_alignn"],
        required=True,
    )
    parser.add_argument(
        "--split_method", choices=["random", "scaffold"], required=True
    )
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
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
    parser.add_argument(
        "--output_root",
        default=str(PROJECT_ROOT / "output" / "openbind_baselines"),
    )
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
