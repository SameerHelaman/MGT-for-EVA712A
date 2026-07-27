"""Freeze original MGT graph triplets for OpenBind ligand PDBs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import dgl
import torch

from utils.datasets import StructureDataset


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of one processed graph file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Construct each original graph triplet once and freeze it on disk."""
    project_root = Path(__file__).resolve().parents[1]
    default_root = (
        project_root / "OpenBind_EV-A71_2A" / "experiment_a_ligand_mgt"
    )
    parser = argparse.ArgumentParser(
        description="Freeze original MGT graphs for OpenBind ligand PDBs."
    )
    parser.add_argument("--data_root", type=Path, default=default_root)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local_radius", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--num_pe_fea", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    dataset_args = SimpleNamespace(
        root=str(data_root),
        max_nei_num=args.max_neighbors,
        num_pe_fea=args.num_pe_fea,
        local_radius=args.local_radius,
        periodic=False,
    )
    dataset = StructureDataset(
        dataset_args, process=True, random_seed=args.seed
    )
    processed_dir = data_root / "processed"
    report_dir = data_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    generated = 0
    reused = 0
    for index in range(len(dataset)):
        complex_name = dataset.id_prop_data[index][0]
        destination = processed_dir / f"{complex_name}.bin"
        if destination.exists() and not args.overwrite:
            graphs, _ = dgl.load_graphs(str(destination))
            if len(graphs) != 3:
                raise ValueError(f"Invalid graph count in {destination}")
            graph, line_graph, full_graph = graphs
            reused += 1
        else:
            graph, line_graph, full_graph, _, loaded_id = dataset[index]
            if loaded_id != complex_name:
                raise ValueError("Dataset identifier changed during processing")
            if not (
                torch.isfinite(graph.ndata["node_feats"]).all()
                and torch.isfinite(graph.ndata["pes"]).all()
                and torch.isfinite(graph.edata["edge_feats"]).all()
                and torch.isfinite(line_graph.edata["angle_feats"]).all()
                and torch.isfinite(full_graph.edata["fc_feats"]).all()
            ):
                raise ValueError(f"Non-finite graph features for {complex_name}")
            dgl.save_graphs(
                str(destination), [graph, line_graph, full_graph]
            )
            generated += 1
        rows.append(
            {
                "complex_name": complex_name,
                "graph_file": f"processed/{destination.name}",
                "sha256": sha256_file(destination),
                "nodes": graph.num_nodes(),
                "local_edges": graph.num_edges(),
                "line_nodes": line_graph.num_nodes(),
                "line_edges": line_graph.num_edges(),
                "coulomb_nodes": full_graph.num_nodes(),
                "coulomb_edges": full_graph.num_edges(),
            }
        )
    manifest_path = report_dir / "processed_graph_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
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
    }
    with (report_dir / "processed_graph_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
