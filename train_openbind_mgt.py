"""Train the complete original MGT Graphformer on OpenBind ligand PDBs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import dgl
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Subset

from model.graphformer import Graphformer
from utils.datasets import StructureDataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "OpenBind_EV-A71_2A" / "experiment_a_ligand_mgt"
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch, CUDA and DGL."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV manifest into dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic Torch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def regression_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Calculate standard affinity-regression metrics."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - observed
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0
        else float("nan")
    )
    pearson = (
        float(pearsonr(observed, predicted).statistic)
        if len(observed) > 1
        and np.std(observed) > 0
        and np.std(predicted) > 0
        else float("nan")
    )
    spearman = (
        float(spearmanr(observed, predicted).statistic)
        if len(observed) > 1
        and np.std(observed) > 0
        and np.std(predicted) > 0
        else float("nan")
    )
    nonzero = np.abs(observed) > np.finfo(np.float64).eps
    mape = (
        float(np.mean(np.abs(residual[nonzero] / observed[nonzero])) * 100)
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "count": int(len(observed)),
        "mae": mae,
        "mape_percent": mape,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2,
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


def make_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    """Provide the exact fields required by original StructureDataset."""
    return argparse.Namespace(
        root=str(Path(args.data_root).resolve()),
        max_nei_num=args.max_neighbors,
        num_pe_fea=args.num_pe_fea,
        local_radius=args.local_radius,
        periodic=False,
    )


def build_subsets(
    dataset: StructureDataset, split_rows: list[dict[str, str]]
) -> tuple[dict[str, Subset], dict[str, str]]:
    """Map frozen manifest membership onto original StructureDataset indices."""
    id_to_index = {
        row[0]: index for index, row in enumerate(dataset.id_prop_data)
    }
    if len(id_to_index) != len(dataset):
        raise ValueError("Duplicate complex names in id_prop.csv")
    expected_ids = set(id_to_index)
    manifest_ids = {row["complex_name"] for row in split_rows}
    if expected_ids != manifest_ids:
        raise ValueError("Split manifest and id_prop.csv do not have equal coverage")
    compound_by_id = {
        row["complex_name"]: row["official_compound_group_id"]
        for row in split_rows
    }
    subsets = {}
    # Evaluate all partitions identically; test remains untouched during selection.
    for split in ("train", "validation", "test"):
        members = [
            id_to_index[row["complex_name"]]
            for row in split_rows
            if row["split"] == split
        ]
        if not members:
            raise ValueError(f"Frozen split {split} is empty")
        subsets[split] = Subset(dataset, members)
    for compound_id in set(compound_by_id.values()):
        memberships = {
            row["split"]
            for row in split_rows
            if row["official_compound_group_id"] == compound_id
        }
        if len(memberships) != 1:
            raise ValueError(f"Compound leakage detected for {compound_id}")
    return subsets, compound_by_id


def make_loaders(
    dataset: StructureDataset,
    subsets: dict[str, Subset],
    args: argparse.Namespace,
) -> dict[str, DataLoader]:
    """Create deterministic graph loaders for all three frozen partitions."""
    generator = torch.Generator().manual_seed(args.seed)
    return {
        split: DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=0,
            collate_fn=dataset.collate_tt,
            drop_last=False,
        )
        for split, subset in subsets.items()
    }


def move_graphs(
    graph: dgl.DGLGraph,
    line_graph: dgl.DGLGraph,
    full_graph: dgl.DGLGraph,
    device: torch.device,
) -> tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
    """Move all three original MGT graph representations to one device."""
    return graph.to(device), line_graph.to(device), full_graph.to(device)


def target_statistics(
    dataset: StructureDataset, subset: Subset
) -> tuple[float, float]:
    """Calculate normalization values from training labels only."""
    values = torch.tensor(
        [
            float(dataset.id_prop_data[index][1])
            for index in subset.indices
        ],
        dtype=torch.float32,
    )
    mean = float(values.mean())
    scale = float(values.std(unbiased=False))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Training target standard deviation is invalid")
    return mean, scale


def train_epoch(
    model: Graphformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    target_mean: float,
    target_scale: float,
    device: torch.device,
) -> float:
    """Train the complete original Graphformer for one epoch."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for graph, line_graph, full_graph, target, _ in loader:
        graph, line_graph, full_graph = move_graphs(
            graph, line_graph, full_graph, device
        )
        target = target.to(device)
        normalized_target = (target - target_mean) / target_scale
        optimizer.zero_grad(set_to_none=True)
        output, _, _, _, _ = model(graph, line_graph, full_graph)
        loss = criterion(output, normalized_target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        batch_size = target.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
    return total_loss / total_examples


@torch.no_grad()
def predict(
    model: Graphformer,
    loader: DataLoader,
    target_mean: float,
    target_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Predict pKD in original units for one partition."""
    model.eval()
    observed_parts = []
    predicted_parts = []
    identifiers = []
    for graph, line_graph, full_graph, target, ids in loader:
        graph, line_graph, full_graph = move_graphs(
            graph, line_graph, full_graph, device
        )
        output, _, _, _, _ = model(graph, line_graph, full_graph)
        prediction = output.squeeze(1) * target_scale + target_mean
        observed_parts.append(target.squeeze(1).cpu().numpy())
        predicted_parts.append(prediction.cpu().numpy())
        identifiers.extend(ids)
    return (
        np.concatenate(observed_parts),
        np.concatenate(predicted_parts),
        identifiers,
    )


def compound_predictions(
    observed: np.ndarray,
    predicted: np.ndarray,
    identifiers: list[str],
    compound_by_id: dict[str, str],
) -> list[dict[str, object]]:
    """Average repeated crystallographic predictions per official compound."""
    grouped = defaultdict(list)
    for truth, estimate, complex_name in zip(
        observed, predicted, identifiers
    ):
        grouped[compound_by_id[complex_name]].append(
            (float(truth), float(estimate), complex_name)
        )
    rows = []
    for compound_id, values in sorted(grouped.items()):
        truths = np.array([value[0] for value in values])
        if float(truths.max() - truths.min()) > 1e-5:
            raise ValueError(f"Conflicting pKD values for {compound_id}")
        rows.append(
            {
                "official_compound_group_id": compound_id,
                "experimental_pKD": float(truths[0]),
                "predicted_pKD": float(
                    np.mean([value[1] for value in values])
                ),
                "structure_count": len(values),
                "complex_names": ";".join(
                    sorted(value[2] for value in values)
                ),
            }
        )
    return rows


def write_prediction_files(
    output_dir: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    identifiers: list[str],
    compound_rows: list[dict[str, object]],
    compound_by_id: dict[str, str],
) -> None:
    """Write structure- and compound-level held-out predictions."""
    structure_path = output_dir / "test_structure_predictions.csv"
    with structure_path.open("w", encoding="utf-8", newline="") as handle:
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
    compound_path = output_dir / "test_compound_predictions.csv"
    with compound_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(compound_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(compound_rows)


def train(args: argparse.Namespace) -> dict[str, object]:
    """Run one complete fixed-split OpenBind Graphformer experiment."""
    # Fix all supported random-number generators before data/model construction.
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
    # Load exact frozen structure and compound membership for this experiment.
    split_rows = read_csv(split_path)
    if {row["seed"] for row in split_rows} != {str(args.seed)}:
        raise ValueError("Split manifest seed does not match requested seed")
    if {row["split_method"] for row in split_rows} != {args.split_method}:
        raise ValueError("Split manifest method does not match requested method")

    dataset_args = make_dataset_args(args)
    processed_dir = data_root / "processed"
    processed_count = len(list(processed_dir.glob("*.bin")))
    if processed_count != 621:
        raise FileNotFoundError(
            "Expected 621 frozen graph files. Run "
            "`python -m utils.preprocess_openbind_mgt_graphs` first."
        )
    # Read the already-frozen original local/line/Coulomb graph triplets.
    dataset = StructureDataset(
        dataset_args, process=False, random_seed=args.seed
    )
    subsets, compound_by_id = build_subsets(dataset, split_rows)
    loaders = make_loaders(dataset, subsets, args)
    # Normalize labels with training statistics only to prevent information leakage.
    target_mean, target_scale = target_statistics(dataset, subsets["train"])

    with (data_root / "atom_init.json").open(
        "r", encoding="utf-8"
    ) as handle:
        atom_initializer = json.load(handle)
    args.num_atom_fea = len(next(iter(atom_initializer.values())))
    args.num_edge_fea = 1
    args.num_angle_fea = 1
    args.num_clmb_fea = 1
    args.out_dims = 1
    args.residual = bool(args.residual)
    # Instantiate the complete original Graphformer after adapting only output width.
    model = Graphformer(args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    criterion = nn.HuberLoss(delta=1.0)
    output_dir = (
        Path(args.output_root).resolve()
        / args.split_method
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_openbind_mgt.pt"

    history = []
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()
    # Optimize until the epoch limit or validation early-stopping criterion.
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            target_mean,
            target_scale,
            device,
        )
        val_observed, val_predicted, _ = predict(
            model,
            loaders["validation"],
            target_mean,
            target_scale,
            device,
        )
        val_normalized_truth = (
            torch.from_numpy(val_observed) - target_mean
        ) / target_scale
        val_normalized_prediction = (
            torch.from_numpy(val_predicted) - target_mean
        ) / target_scale
        validation_loss = float(
            criterion(
                val_normalized_prediction, val_normalized_truth
            )
        )
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"epoch={epoch} train={train_loss:.6f} "
            f"validation={validation_loss:.6f} lr={learning_rate:.2e}",
            flush=True,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_scale": target_scale,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                break
    training_seconds = time.perf_counter() - training_start

    # Restore the single checkpoint selected without consulting test labels.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    split_predictions = {}
    split_metrics = {}
    test_start = time.perf_counter()
    for split in ("train", "validation", "test"):
        observed, predicted, identifiers = predict(
            model,
            loaders[split],
            target_mean,
            target_scale,
            device,
        )
        compound_rows = compound_predictions(
            observed, predicted, identifiers, compound_by_id
        )
        compound_observed = np.array(
            [row["experimental_pKD"] for row in compound_rows]
        )
        compound_predicted = np.array(
            [row["predicted_pKD"] for row in compound_rows]
        )
        split_predictions[split] = (
            observed,
            predicted,
            identifiers,
            compound_rows,
        )
        split_metrics[split] = {
            "structure_level": regression_metrics(observed, predicted),
            "compound_level": regression_metrics(
                compound_observed, compound_predicted
            ),
        }
    test_inference_seconds = time.perf_counter() - test_start

    test_observed, test_predicted, test_ids, test_compounds = (
        split_predictions["test"]
    )
    write_prediction_files(
        output_dir,
        test_observed,
        test_predicted,
        test_ids,
        test_compounds,
        compound_by_id,
    )
    with (output_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    result = {
        "model": "OpenBind complete original MGT Graphformer",
        "split_method": args.split_method,
        "seed": args.seed,
        "device": str(device),
        "counts": {
            split: len(subset) for split, subset in subsets.items()
        },
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
        },
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
        },
        "evaluation": {
            "attention_dropout_disabled_in_eval": True,
            "test_inference_seconds_all_levels": test_inference_seconds,
            "compound_aggregation": "mean prediction across repeated structures",
        },
        "metrics": split_metrics,
        "data": {
            "root": str(data_root),
            "split_manifest": str(split_path),
            "representation": "quality-controlled crystallographic ligand PDB",
            "graphs": "frozen original-MGT processed graph triplets",
        },
    }
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    """Parse experiment settings and train one original MGT model."""
    parser = argparse.ArgumentParser(
        description="Train complete original MGT on OpenBind ligand PDBs."
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
    parser.add_argument(
        "--output_root",
        default=str(PROJECT_ROOT / "output" / "openbind_full_mgt"),
    )
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
