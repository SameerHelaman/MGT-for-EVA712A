# =============================================================================
# MODULE: utils/preprocess_openbind_mgt_graphs.py
# PURPOSE: Builds and serialises original-MGT DGL graph triplets for every curated ligand.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Serialised DGL .bin graph triplets, manifest CSV and summary JSON.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Freeze original MGT graph triplets for OpenBind ligand PDBs."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

import argparse  # Load a standard-library, scientific, or local project dependency.
import csv  # Load a standard-library, scientific, or local project dependency.
import hashlib  # Load a standard-library, scientific, or local project dependency.
import json  # Load a standard-library, scientific, or local project dependency.
from pathlib import Path  # Import selected classes or functions from the named dependency.
from types import SimpleNamespace  # Import selected classes or functions from the named dependency.

import dgl  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.

from utils.datasets import StructureDataset  # Import selected classes or functions from the named dependency.


# FUNCTION: sha256_file — see its docstring and inline comments.
def sha256_file(path: Path) -> str:  # Define this callable; its indented block implements the documented operation.
    """Calculate the SHA-256 checksum of one processed graph file."""
    digest = hashlib.sha256()  # Bind this name to an intermediate value, configuration setting, or result.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)  # Perform this step of the surrounding calculation or control-flow block.
    return digest.hexdigest()  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: main — see its docstring and inline comments.
def main() -> None:  # Define this callable; its indented block implements the documented operation.
    """Construct each original graph triplet once and freeze it on disk."""
    project_root = Path(__file__).resolve().parents[1]  # Bind this name to an intermediate value, configuration setting, or result.
    default_root = (  # Bind this name to an intermediate value, configuration setting, or result.
        project_root / "OpenBind_EV-A71_2A" / "experiment_a_ligand_mgt"
    )  # Continue or close the surrounding multiline expression or collection.
    parser = argparse.ArgumentParser(  # Bind this name to an intermediate value, configuration setting, or result.
        description="Freeze original MGT graphs for OpenBind ligand PDBs."
    )  # Continue or close the surrounding multiline expression or collection.
    parser.add_argument("--data_root", type=Path, default=default_root)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local_radius", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--num_pe_fea", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.
    data_root = args.data_root.resolve()  # Bind this name to an intermediate value, configuration setting, or result.
    dataset_args = SimpleNamespace(  # Prepare dataset membership or batched data access for the experiment.
        root=str(data_root),  # Bind this name to an intermediate value, configuration setting, or result.
        max_nei_num=args.max_neighbors,  # Bind this name to an intermediate value, configuration setting, or result.
        num_pe_fea=args.num_pe_fea,  # Bind this name to an intermediate value, configuration setting, or result.
        local_radius=args.local_radius,  # Bind this name to an intermediate value, configuration setting, or result.
        periodic=False,  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    dataset = StructureDataset(  # Prepare dataset membership or batched data access for the experiment.
        dataset_args, process=True, random_seed=args.seed  # Prepare dataset membership or batched data access for the experiment.
    )  # Continue or close the surrounding multiline expression or collection.
    processed_dir = data_root / "processed"
    report_dir = data_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    report_dir.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.
    rows = []  # Bind this name to an intermediate value, configuration setting, or result.
    generated = 0  # Bind this name to an intermediate value, configuration setting, or result.
    reused = 0  # Bind this name to an intermediate value, configuration setting, or result.
    for index in range(len(dataset)):  # Iterate over the stated records, layers, batches, or graph elements.
        complex_name = dataset.id_prop_data[index][0]  # Prepare dataset membership or batched data access for the experiment.
        destination = processed_dir / f"{complex_name}.bin"  # Bind this name to an intermediate value, configuration setting, or result.
        if destination.exists() and not args.overwrite:  # Evaluate this condition before executing the associated branch.
            graphs, _ = dgl.load_graphs(str(destination))  # Construct or transform graph topology, geometry, or molecular feature data.
            if len(graphs) != 3:  # Evaluate this condition before executing the associated branch.
                raise ValueError(f"Invalid graph count in {destination}")  # Reject invalid input or state with an explicit exception.
            graph, line_graph, full_graph = graphs  # Construct or transform graph topology, geometry, or molecular feature data.
            reused += 1  # Bind this name to an intermediate value, configuration setting, or result.
        else:  # Handle the remaining case not covered by earlier conditions.
            graph, line_graph, full_graph, _, loaded_id = dataset[index]  # Construct or transform graph topology, geometry, or molecular feature data.
            if loaded_id != complex_name:  # Evaluate this condition before executing the associated branch.
                raise ValueError("Dataset identifier changed during processing")
            if not (  # Evaluate this condition before executing the associated branch.
                torch.isfinite(graph.ndata["node_feats"]).all()
                and torch.isfinite(graph.ndata["pes"]).all()
                and torch.isfinite(graph.edata["edge_feats"]).all()
                and torch.isfinite(line_graph.edata["angle_feats"]).all()
                and torch.isfinite(full_graph.edata["fc_feats"]).all()
            ):  # Continue or close the surrounding multiline expression or collection.
                raise ValueError(f"Non-finite graph features for {complex_name}")  # Reject invalid input or state with an explicit exception.
            dgl.save_graphs(  # Invoke the relevant tensor, graph, numerical, or tabular operation.
                str(destination), [graph, line_graph, full_graph]  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
            generated += 1  # Bind this name to an intermediate value, configuration setting, or result.
        rows.append(  # Perform this step of the surrounding calculation or control-flow block.
            {  # Continue or close the surrounding multiline expression or collection.
                "complex_name": complex_name,
                "graph_file": f"processed/{destination.name}",
                "sha256": sha256_file(destination),
                "nodes": graph.num_nodes(),
                "local_edges": graph.num_edges(),
                "line_nodes": line_graph.num_nodes(),
                "line_edges": line_graph.num_edges(),
                "coulomb_nodes": full_graph.num_nodes(),
                "coulomb_edges": full_graph.num_edges(),
            }  # Continue or close the surrounding multiline expression or collection.
        )  # Continue or close the surrounding multiline expression or collection.
    manifest_path = report_dir / "processed_graph_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))  # Bind this name to an intermediate value, configuration setting, or result.
        writer.writeheader()  # Perform this step of the surrounding calculation or control-flow block.
        writer.writerows(rows)  # Perform this step of the surrounding calculation or control-flow block.
    summary = {  # Bind this name to an intermediate value, configuration setting, or result.
        "dataset": "OpenBind EV-A71 2A Experiment A",
        "graph_constructor": "original utils.datasets.StructureDataset",
        "graph_count": len(rows),
        "generated_this_run": generated,
        "reused_this_run": reused,
        "seed": args.seed,
        "local_radius_angstrom": args.local_radius,
        "maximum_local_neighbors": args.max_neighbors,
        "laplacian_positional_encoding_dimension": args.num_pe_fea,
        "manifest": str(manifest_path.relative_to(data_root)),
        "manifest_sha256": sha256_file(manifest_path),
    }  # Continue or close the surrounding multiline expression or collection.
    with (report_dir / "processed_graph_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:  # Continue or close the surrounding multiline expression or collection.
        json.dump(summary, handle, indent=2, sort_keys=True)  # Bind this name to an intermediate value, configuration setting, or result.
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))  # Report progress, predictions, or metrics to the selected output/logging backend.


if __name__ == "__main__":
    main()  # Perform this step of the surrounding calculation or control-flow block.
