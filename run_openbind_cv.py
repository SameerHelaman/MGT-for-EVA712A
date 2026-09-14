# =============================================================================
# MODULE: run_openbind_cv.py
# PURPOSE: Run and aggregate random or scaffold five-fold CV for all model configurations.
# LIBRARIES: Standard-library process/file modules orchestrate existing trainers;
#            NumPy/Pandas/SciPy calculate pooled out-of-fold regression statistics.
# OUTPUT: Isolated fold directories, command logs, per-fold summaries, pooled
#         predictions and method-specific cv_summary.json/csv output directories.
# CALCULATIONS: Existing model trainers remain unchanged; this runner combines
#               each compound's single outer-test prediction into pooled metrics.
# =============================================================================
"""Run all seven OpenBind models with random or scaffold five-fold cross-validation."""

from __future__ import annotations  # Postpone evaluation of modern type annotations.

import argparse  # Define model, fold, device and resume command-line options.
import csv  # Write compact fold-level result tables for dissertation analysis.
import json  # Read model metrics and write reproducible run/summary metadata.
import subprocess  # Invoke each existing trainer in an isolated Python process.
import sys  # Reuse the active environment's Python interpreter.
import time  # Record experiment wall-clock durations and completion times.
from pathlib import Path  # Resolve repository, manifest and output locations safely.

import numpy as np  # Calculate pooled numerical regression errors.
import pandas as pd  # Combine per-fold compound prediction tables.
from scipy.stats import pearsonr, spearmanr  # Calculate linear and rank correlations.

from utils.generate_scaffold_cv import DEFAULT_DATA_ROOT, generate as generate_scaffold_cv  # Reuse audited scaffold grouping.
from utils.generate_random_cv import generate as generate_random_cv  # Reuse audited compound-grouped random stratification.

PROJECT_ROOT = Path(__file__).resolve().parent  # Locate all trainer scripts relative to this runner.
DEFAULT_CV_DIRS = {"scaffold": DEFAULT_DATA_ROOT / "cv", "random": DEFAULT_DATA_ROOT / "cv_random"}  # Keep each CV design separate.
DEFAULT_OUTPUT_ROOTS = {"scaffold": PROJECT_ROOT / "output" / "openbind_scaffold_cv", "random": PROJECT_ROOT / "output" / "openbind_random_cv"}  # Isolate results by scientific design.
MODEL_ORDER = (  # Freeze a scientifically interpretable controlled model order.
    "morgan_mlp",  # Fixed 2D Morgan-fingerprint multilayer perceptron baseline.
    "2d_gnn",  # Learned atom-bond connectivity graph baseline.
    "3d_gnn",  # Crystallographic distance graph neural network.
    "3d_alignn",  # Crystallographic distance-plus-angle ALIGNN model.
    "adapted_mgt",  # Unmasked ligand MGT using the original Graphformer encoder.
    "masked_alignn",  # ALIGNN after training-fold-only atom masking pretraining.
    "masked_mgt",  # Adapted ligand MGT after training-fold-only masking pretraining.
)  # Finish the complete seven-configuration experiment list.
MODEL_LABELS = {  # Map command-line keys to dissertation-facing names.
    "morgan_mlp": "Morgan MLP",  # Label the fixed fingerprint baseline.
    "2d_gnn": "2D GNN",  # Label the learned connectivity baseline.
    "3d_gnn": "3D distance GNN",  # Label the distance-only geometric model.
    "3d_alignn": "3D ALIGNN",  # Label the unmasked angular model.
    "adapted_mgt": "Adapted ligand MGT",  # Distinguish the matched trainer from original-script reproduction.
    "masked_alignn": "Masked ALIGNN",  # Label the pretrained angular model.
    "masked_mgt": "Masked adapted ligand MGT",  # Label the pretrained matched MGT.
}  # Finish the reporting-name mapping.


def fold_manifest(cv_dir: Path, method: str, fold: int, seed: int, folds: int) -> Path:  # Centralize immutable manifest naming.
    """Return the trainer-compatible manifest path for one outer fold."""
    return cv_dir / f"{method}_cv{folds}_fold_{fold}_seed_{seed}.csv"  # Match either generator's deterministic filename.


def model_output_dir(output_root: Path, method: str, fold: int, model: str, seed: int) -> Path:  # Resolve one trainer's final artifact directory.
    """Return the directory containing metrics.json for one model/fold."""
    fold_root = output_root / f"fold_{fold}"  # Give every outer experiment an isolated root.
    if model in {"morgan_mlp", "2d_gnn", "3d_gnn", "3d_alignn"}:  # Handle standard matched baseline path conventions.
        return fold_root / "unmasked" / model / method / f"seed_{seed}"  # Mirror the baseline trainer's nested output layout.
    if model == "adapted_mgt":  # Handle the MGT trainer, which does not add a model-name directory.
        return fold_root / "unmasked" / "adapted_mgt" / method / f"seed_{seed}"  # Isolate unmasked MGT artifacts.
    if model == "masked_alignn":  # Handle the masking script's ALIGNN-specific nested directory.
        return fold_root / "masked_alignn" / "3d_alignn" / method / f"seed_{seed}"  # Locate masked ALIGNN metrics.
    if model == "masked_mgt":  # Handle the masking script's MGT output layout.
        return fold_root / "masked_mgt" / method / f"seed_{seed}"  # Locate masked MGT metrics.
    raise ValueError(f"Unknown model: {model}")  # Reject unsupported model keys before launching a process.


def trainer_command(args: argparse.Namespace, model: str, fold: int, manifest: Path) -> list[str]:  # Build one exact experiment command.
    """Construct the existing trainer CLI for one model and outer fold."""
    fold_root = args.output_root / f"fold_{fold}"  # Resolve this fold's isolated result root.
    common = [  # Define options shared by every matched supervised trainer.
        "--split_method", args.cv_method,  # Tell trainers to enforce the selected manifest-method metadata.
        "--split_file", str(manifest),  # Supply exact fold-specific train/validation/test membership.
        "--data_root", str(args.data_root),  # Point every trainer to the same curated structures/graphs.
        "--seed", str(args.seed),  # Use a paired initialization/shuffling seed across architectures.
        "--device", args.device,  # Select automatic, CPU or CUDA execution.
        "--batch_size", str(args.batch_size),  # Use the requested matched mini-batch size.
        "--max_epochs", str(args.max_epochs),  # Apply the same supervised epoch ceiling.
        "--early_stopping_patience", str(args.early_stopping_patience),  # Select checkpoints only from validation loss.
    ]  # Finish the common supervised options.
    if model == "morgan_mlp":  # Select the fixed fingerprint entry point.
        return [sys.executable, str(PROJECT_ROOT / "train_openbind_morgan_mlp.py"), *common, "--output_root", str(fold_root / "unmasked")]  # Build its full CLI.
    if model == "2d_gnn":  # Select the learned two-dimensional graph entry point.
        return [sys.executable, str(PROJECT_ROOT / "train_openbind_2d_gnn.py"), *common, "--output_root", str(fold_root / "unmasked")]  # Build its full CLI.
    if model == "3d_gnn":  # Select the crystallographic distance model entry point.
        return [sys.executable, str(PROJECT_ROOT / "train_openbind_3d_gnn.py"), *common, "--output_root", str(fold_root / "unmasked")]  # Build its full CLI.
    if model == "3d_alignn":  # Select the crystallographic angular model entry point.
        return [sys.executable, str(PROJECT_ROOT / "train_openbind_3d_alignn.py"), *common, "--output_root", str(fold_root / "unmasked")]  # Build its full CLI.
    if model == "adapted_mgt":  # Select the matched ligand MGT entry point.
        return [sys.executable, str(PROJECT_ROOT / "train_openbind_mgt.py"), *common, "--output_root", str(fold_root / "unmasked" / "adapted_mgt")]  # Build its full CLI.
    masking_common = [  # Add pretraining options required only by masked configurations.
        *common,  # Retain the exact supervised fold, seed and optimization limits.
        "--pretrain_epochs", str(args.pretrain_epochs),  # Set the atom-reconstruction epoch count.
        "--mask_rate", str(args.mask_rate),  # Set the fraction of training atoms masked per batch.
    ]  # Finish the masking-specific common options.
    if model == "masked_alignn":  # Select masking followed by ALIGNN fine-tuning.
        return [sys.executable, str(PROJECT_ROOT / "train_openbind_masked.py"), "--model", "3d_alignn", *masking_common, "--output_root", str(fold_root / "masked_alignn")]  # Build its full CLI.
    if model == "masked_mgt":  # Select masking followed by adapted-MGT fine-tuning.
        return [sys.executable, str(PROJECT_ROOT / "train_openbind_masked.py"), "--model", "mgt", *masking_common, "--output_root", str(fold_root / "masked_mgt")]  # Build its full CLI.
    raise ValueError(f"Unknown model: {model}")  # Reject unsupported keys rather than launching the wrong script.


def pooled_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:  # Calculate out-of-fold compound metrics.
    """Return the same principal regression statistics used by matched trainers."""
    residual = predicted - observed  # Calculate signed prediction errors in pKD units.
    mse = float(np.mean(residual**2))  # Average squared errors across all out-of-fold compounds.
    mae = float(np.mean(np.abs(residual)))  # Average absolute errors across all out-of-fold compounds.
    denominator = float(np.sum((observed - observed.mean()) ** 2))  # Calculate total target variation for R-squared.
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan")  # Compare residual and total variation.
    pearson = float(pearsonr(observed, predicted).statistic) if len(observed) > 1 else float("nan")  # Calculate linear association.
    spearman = float(spearmanr(observed, predicted).statistic) if len(observed) > 1 else float("nan")  # Calculate rank association.
    return {  # Return JSON-serializable compound-level statistics.
        "n_compounds": int(len(observed)),  # Report complete out-of-fold sample size.
        "mae": mae,  # Report mean absolute error in pKD.
        "mse": mse,  # Report mean squared error in squared pKD.
        "rmse": float(np.sqrt(mse)),  # Convert MSE back to pKD units.
        "r2": r2,  # Report variance explained relative to the pooled target mean.
        "pearson_r": pearson,  # Report linear correlation.
        "spearman_r": spearman,  # Report rank correlation.
    }  # Finish the pooled metric record.


def summarize(output_root: Path, method: str, models: list[str], folds_to_use: list[int], seed: int) -> dict[str, object]:  # Aggregate completed fold artifacts.
    """Create fold-level and pooled out-of-fold summaries for available complete models."""
    fold_rows: list[dict[str, object]] = []  # Collect one metric record per completed model/fold pair.
    pooled: dict[str, dict[str, object]] = {}  # Collect one pooled result record per model with all requested folds.
    for model in models:  # Aggregate configurations in controlled hierarchy order.
        prediction_frames: list[pd.DataFrame] = []  # Collect outer-test compound predictions for pooled evaluation.
        complete = True  # Assume all requested folds exist until a missing artifact is found.
        for fold in folds_to_use:  # Read every requested outer test fold.
            directory = model_output_dir(output_root, method, fold, model, seed)  # Resolve this model/fold artifact directory.
            metrics_path = directory / "metrics.json"  # Locate the trainer's complete metric record.
            predictions_path = directory / "test_compound_predictions.csv"  # Locate compound-aggregated outer-test predictions.
            if not metrics_path.exists() or not predictions_path.exists():  # Detect interrupted or not-yet-run experiments.
                complete = False  # Prevent misleading pooled metrics from incomplete outer coverage.
                continue  # Retain available fold rows but skip missing artifacts.
            payload = json.loads(metrics_path.read_text())  # Load exact saved model and evaluation metadata.
            metric = payload["metrics"]["test"]["compound_level"]  # Select the comparable compound-level outer-test result.
            fold_rows.append({  # Create a compact fold-level summary row.
                "model_key": model,  # Preserve the command-line configuration key.
                "model": MODEL_LABELS[model],  # Provide a dissertation-facing label.
                "outer_fold": fold,  # Identify which compound/scaffold group was held out.
                "test_compounds": metric.get("count", metric.get("n_compounds")),  # Record evaluated compound count.
                "mae": metric["mae"],  # Record fold compound MAE.
                "rmse": metric["rmse"],  # Record fold compound RMSE.
                "r2": metric["r2"],  # Record fold compound R-squared.
                "pearson_r": metric["pearson_r"],  # Record fold Pearson correlation.
                "spearman_r": metric["spearman_r"],  # Record fold Spearman correlation.
                "metrics_path": str(metrics_path.resolve()),  # Preserve provenance to the full result.
            })  # Finish this fold result row.
            frame = pd.read_csv(predictions_path)  # Load each compound's single held-out prediction.
            frame["outer_fold"] = fold  # Add fold provenance before concatenation.
            prediction_frames.append(frame)  # Retain this fold for pooled analysis.
        if complete and len(prediction_frames) == len(folds_to_use):  # Pool only complete requested outer coverage.
            combined = pd.concat(prediction_frames, ignore_index=True)  # Concatenate non-overlapping outer-test compounds.
            id_column = "official_compound_group_id"  # Use the frozen compound grouping identifier.
            if combined[id_column].duplicated().any():  # Each compound must have exactly one out-of-fold prediction.
                raise AssertionError(f"Duplicate out-of-fold compound predictions for {model}")  # Reject leakage or aggregation errors.
            observed = combined["experimental_pKD"].to_numpy(dtype=float)  # Extract pooled experimental targets.
            predicted = combined["predicted_pKD"].to_numpy(dtype=float)  # Extract pooled held-out predictions.
            model_metrics = pooled_metrics(observed, predicted)  # Calculate pooled statistics across all test folds.
            model_fold_rows = [row for row in fold_rows if row["model_key"] == model]  # Select this model's fold results.
            for metric_name in ("mae", "rmse", "r2", "pearson_r", "spearman_r"):  # Summarize variation across outer folds.
                values = np.asarray([float(row[metric_name]) for row in model_fold_rows])  # Form the five-fold metric vector.
                model_metrics[f"fold_mean_{metric_name}"] = float(values.mean())  # Report the arithmetic fold mean.
                model_metrics[f"fold_sd_{metric_name}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0  # Report sample standard deviation.
            prediction_output = output_root / "summary" / f"{model}_out_of_fold_compound_predictions.csv"  # Define the pooled prediction artifact.
            prediction_output.parent.mkdir(parents=True, exist_ok=True)  # Create the summary directory once required.
            combined.sort_values(id_column).to_csv(prediction_output, index=False)  # Persist auditable out-of-fold predictions.
            model_metrics["prediction_file"] = str(prediction_output.resolve())  # Link metrics to exact pooled predictions.
            pooled[model] = model_metrics  # Store this model's complete pooled result.
    summary_dir = output_root / "summary"  # Resolve the shared summary artifact directory.
    summary_dir.mkdir(parents=True, exist_ok=True)  # Create it even when only partial folds are complete.
    fold_csv = summary_dir / "fold_metrics.csv"  # Define the tidy fold-level result table.
    if fold_rows:  # Write a table only when at least one run has completed.
        with fold_csv.open("w", encoding="utf-8", newline="") as handle:  # Open with stable CSV newline handling.
            writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))  # Freeze columns from the complete first row.
            writer.writeheader()  # Write descriptive headers.
            writer.writerows(fold_rows)  # Write all available fold results.
    summary = {  # Build the machine-readable cross-validation summary.
        "design": f"five-fold {method} cross-validation with fold-specific validation",  # State evaluation design.
        "seed": seed,  # Record paired experiment seed.
        "requested_folds": folds_to_use,  # Record which folds were aggregated.
        "requested_models": models,  # Record which configurations were requested.
        "fold_metrics_file": str(fold_csv.resolve()),  # Link the tidy fold result table.
        "pooled_out_of_fold_metrics": pooled,  # Store complete pooled metrics and fold uncertainty.
    }  # Finish the summary object.
    (summary_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")  # Persist reproducible summary JSON.
    return summary  # Return the summary for command-line display and tests.


def parse_models(values: list[str]) -> list[str]:  # Normalize the flexible --models argument.
    """Expand `all` or validate a duplicate-free ordered model list."""
    if values == ["all"]:  # Recognize the default complete experiment request.
        return list(MODEL_ORDER)  # Expand to all seven configurations in scientific order.
    if "all" in values:  # Prevent ambiguous mixtures such as `all 3d_gnn`.
        raise ValueError("Use --models all by itself, or list individual models")  # Require one unambiguous selection style.
    unknown = set(values) - set(MODEL_ORDER)  # Find unsupported model keys.
    if unknown:  # Reject misspellings before expensive training begins.
        raise ValueError(f"Unknown models: {sorted(unknown)}")  # Report exact invalid values.
    return [model for model in MODEL_ORDER if model in values]  # Return unique models in controlled hierarchy order.


def main() -> None:  # Define the complete cross-validation command-line workflow.
    """Generate folds, run selected models and aggregate available out-of-fold results."""
    parser = argparse.ArgumentParser(description=__doc__)  # Create a self-documenting scientific experiment CLI.
    parser.add_argument("--cv_method", choices=["random", "scaffold"], default="scaffold")  # Select interpolation or unseen-framework evaluation.
    parser.add_argument("--models", nargs="+", default=["all"])  # Run all seven models or an explicit subset.
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))  # Run all five outer folds or selected indices.
    parser.add_argument("--n_folds", type=int, default=5)  # Fix the outer cross-validation fold count.
    parser.add_argument("--seed", type=int, default=123)  # Use the established paired experiment seed.
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)  # Locate curated structures and processed graphs.
    parser.add_argument("--cv_dir", type=Path, default=None)  # Locate or create frozen fold manifests.
    parser.add_argument("--output_root", type=Path, default=None)  # Isolate all CV checkpoints and reports.
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")  # Select execution hardware.
    parser.add_argument("--batch_size", type=int, default=32)  # Apply a common supervised mini-batch size.
    parser.add_argument("--max_epochs", type=int, default=200)  # Apply the established supervised epoch ceiling.
    parser.add_argument("--early_stopping_patience", type=int, default=10)  # Stop using validation loss only.
    parser.add_argument("--pretrain_epochs", type=int, default=30)  # Apply the established masking pretraining duration.
    parser.add_argument("--mask_rate", type=float, default=0.2)  # Apply the established atom masking fraction.
    parser.add_argument("--resume", action="store_true")  # Skip runs whose metrics and predictions already exist.
    parser.add_argument("--dry_run", action="store_true")  # Print exact commands without launching training.
    parser.add_argument("--prepare_only", action="store_true")  # Generate/audit folds without launching any model.
    args = parser.parse_args()  # Parse all experiment settings.
    args.data_root = args.data_root.resolve()  # Normalize dataset paths for logs and child processes.
    args.cv_dir = (args.cv_dir or DEFAULT_CV_DIRS[args.cv_method]).resolve()  # Select and normalize method-specific manifests.
    args.output_root = (args.output_root or DEFAULT_OUTPUT_ROOTS[args.cv_method]).resolve()  # Select and normalize method-specific results.
    models = parse_models(args.models)  # Expand and validate requested model configurations.
    if any(fold < 0 or fold >= args.n_folds for fold in args.folds):  # Ensure every requested fold index exists.
        raise ValueError(f"Fold indices must be between 0 and {args.n_folds - 1}")  # Reject invalid outer folds.
    source = args.data_root / "splits" / f"{args.cv_method}_seed_{args.seed}.csv"  # Select the corresponding audited source manifest.
    generator = generate_scaffold_cv if args.cv_method == "scaffold" else generate_random_cv  # Select grouping semantics before allocation.
    metadata = generator(args.data_root, source, args.cv_dir, args.n_folds, args.seed)  # Regenerate deterministic manifests and checksums.
    if args.prepare_only:  # Allow users to inspect membership before expensive computation.
        print(json.dumps(metadata, indent=2, sort_keys=True))  # Display the complete fold audit.
        return  # Stop before launching trainers.
    run_log: list[dict[str, object]] = []  # Record every launched, skipped or dry-run command.
    for fold in args.folds:  # Execute one complete outer experiment at a time.
        manifest = fold_manifest(args.cv_dir, args.cv_method, fold, args.seed, args.n_folds)  # Resolve exact frozen membership for this fold.
        for model in models:  # Run every requested representation on identical fold membership.
            output_dir = model_output_dir(args.output_root, args.cv_method, fold, model, args.seed)  # Resolve expected model artifacts.
            metrics_path = output_dir / "metrics.json"  # Identify completion using the trainer's final metric file.
            predictions_path = output_dir / "test_compound_predictions.csv"  # Require held-out compound predictions too.
            command = trainer_command(args, model, fold, manifest)  # Construct an auditable child-process command.
            status = "pending"  # Initialize this run's status before resume/dry-run decisions.
            start = time.perf_counter()  # Start a monotonic wall-clock timer.
            if args.resume and metrics_path.exists() and predictions_path.exists():  # Reuse only complete fold artifacts.
                status = "skipped_complete"  # Record that no training was repeated.
            elif args.dry_run:  # Avoid mutation while displaying the full planned experiment.
                status = "dry_run"  # Record that the command was validated but not executed.
                print(" ".join(command), flush=True)  # Print a copy-pasteable shell command.
            else:  # Launch the real existing trainer for this model/fold pair.
                print(f"cv_fold={fold} model={model} status=running", flush=True)  # Provide concise progress before a potentially long run.
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)  # Execute and stop immediately if the trainer fails.
                status = "completed"  # Mark successful child-process completion.
            elapsed = time.perf_counter() - start  # Calculate orchestration time for this run.
            run_log.append({  # Store exact provenance for later audit.
                "outer_fold": fold,  # Record the held-out outer fold.
                "model": model,  # Record model configuration key.
                "status": status,  # Record completed, skipped or dry-run state.
                "elapsed_seconds": elapsed,  # Record wall-clock orchestration duration.
                "manifest": str(manifest),  # Link exact train/validation/test membership.
                "command": command,  # Preserve every CLI token without shell quoting ambiguity.
                "metrics_path": str(metrics_path),  # Link expected final metrics.
            })  # Finish this run-log record.
            args.output_root.mkdir(parents=True, exist_ok=True)  # Ensure the root exists before updating progress metadata.
            (args.output_root / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")  # Persist progress after every run.
    summary = summarize(args.output_root, args.cv_method, models, args.folds, args.seed)  # Aggregate all completed requested results.
    print(json.dumps(summary, indent=2, sort_keys=True))  # Display final or partial CV results.


if __name__ == "__main__":  # Run orchestration only when this file is executed directly.
    main()  # Generate folds, run models and summarize out-of-fold predictions.
