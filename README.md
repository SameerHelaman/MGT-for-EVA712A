# Molecular Graph Transformer for OpenBind single-target affinity prediction

This repository adapts the original Molecular Graph Transformer (MGT) implementation to predict experimental **pKD** for crystallographic ligands from the OpenBind EV-A71/CVA16 2A protease release. It contains the original MGT code, a quality-controlled OpenBind dataset pipeline, controlled 2D/3D baselines, the complete original Graphformer, and an atom-masking pretraining experiment.

The central scientific question is:

> Do learned 2D chemistry, crystallographic distances, bond angles, and full Coulomb-attention MGT components improve ligand-affinity prediction and generalisation to unseen chemical scaffolds?

The main result is that crystallographic geometry is useful. Without pretraining, the 3D distance GNN gives the strongest scaffold generalisation. With 20% atom-masking pretraining, complete MGT improves substantially and becomes numerically best, although its advantage over the much smaller 3D distance GNN is too small to establish from one seed.

## Contents

- [Dataset](#dataset)
- [Scientific design](#scientific-design)
- [Models](#models)
- [Results](#results)
- [Repository tree and call map](#repository-tree-and-call-map)
- [Original code and modifications](#original-code-and-modifications)
- [Installation](#installation)
- [Reproducing the dataset](#reproducing-the-dataset)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Output files](#output-files)
- [Implementation walkthrough](#implementation-walkthrough)
- [Limitations](#limitations)

## Dataset

The experiments use the OpenBind Structure–Affinity Data Release for EV-A71/CVA16 2A protease. The public release contains 925 crystallographic binding events from 699 compounds, with affinity values available for a subset. Affinity is measured as KD using the Creoptix WAVEsystem and represented here as:

```text
pKD = -log10(KD in molar units)
```

This project is a **ligand-only, single-target** experiment. It uses the crystallographic coordinates in each `ligand_ref.sdf`; it does not provide the protein structure or sequence to the model. Predictions therefore represent single-target ligand structure–activity learning, not a target-agnostic physical protein–ligand scoring function.

### Curation rules

A crystallographic record is retained only when:

1. `experimental_pKD` is present.
2. The complex is present in the official affinity reference.
3. The ligand is not covalent.
4. The structure is not marked as a suspected artefact.
5. The reference pose passes the supplied PoseBusters validity flag.
6. Metadata and reference-SDF canonical molecular identities agree.

The frozen curated dataset contains:

| Quantity | Count |
|---|---:|
| Source metadata rows | 925 |
| Curated crystallographic ligand structures | 621 |
| Excluded structures | 304 |
| Benchmark compound groups | 474 |
| Train structures | 435 |
| Validation structures | 93 |
| Test structures | 93 |

Some compounds have multiple crystallographic structures. Every structure for the same `official_compound_group_id` is assigned to the same partition. Test predictions are reported at both structure level and compound level; compound predictions are the mean across repeated crystal structures.

### Frozen splits

Two 70/15/15 manifests are provided with seed 123:

- Random compound-group split: `splits/random_seed_123.csv`
- Bemis–Murcko scaffold split: `splits/scaffold_seed_123.csv`

Both have 435/93/93 structures. Compound leakage is zero for both; scaffold leakage is zero for the scaffold split. Checksums and the exact rules are stored in `OpenBind_EV-A71_2A/experiment_a_ligand_mgt/reports/dataset_metadata.json`.

The scaffold split is the primary generalisation test because its test compounds contain scaffolds absent from training. The random split mainly measures interpolation among related chemistry.

## Scientific design

The controlled ablation adds one representation component at a time:

```text
Morgan fingerprint MLP
        │ fixed 2D substructure representation
        ▼
2D GNN
        │ learned atom–bond message passing
        ▼
Crystallographic 3D distance GNN
        │ spatial neighbours + distance RBF expansion
        ▼
Crystallographic 3D ALIGNN
        │ line graph + bond-angle cosine RBF expansion
        ▼
Complete original MGT Graphformer
        │ Coulomb graph attention + Laplacian PE
        │ + ALIGNN + local GNN + FFN/residual/norm
        ▼
Masked complete MGT
          20% atom-feature reconstruction before affinity fine-tuning
```

Unless noted otherwise, matched experiments use:

- Seed: 123
- Batch size: 32
- Optimizer: Adam
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- Loss: Huber, delta 1.0
- Train-only target normalization
- Maximum epochs: 200
- Early-stopping patience: 10
- Best validation-loss checkpoint for testing
- Metrics: MAE, MSE, RMSE, R², Pearson r, Spearman r and MAPE

## Models

### Morgan fingerprint MLP

- Morgan fingerprint radius 2, 2048 bits.
- MLP: `2048 → 128 → 64 → 1`.
- Batch normalization, ReLU and dropout.
- No learned graph, coordinates, angles or Coulomb attention.
- Parameters: 270,977.

Purpose: fixed-representation chemical baseline.

### 2D GNN

- Atoms are nodes.
- Chemical bonds are edges.
- 152-dimensional RDKit atom features.
- 12-dimensional RDKit bond features.
- Three original `EdgeGatedGraphConv` layers.
- No coordinates, spatial neighbours, distance RBFs, line graph or Coulomb graph.
- Parameters: 4,094,849.

Purpose: test whether learned molecular graphs outperform Morgan fingerprints.

### Crystallographic 3D distance GNN

- Same atom and bond chemistry as the 2D GNN.
- Coordinates loaded from OpenBind `ligand_ref.sdf` crystal poses.
- Chemical-bond and spatial-neighbour edge union.
- 5 Å spatial cutoff and up to 32 spatial neighbours per atom.
- 40-bin distance RBF expansion using the original MGT `RBFExpansion`.
- Three original `EdgeGatedGraphConv` layers.
- No line graph, angular ALIGNN processing or Coulomb attention.
- Parameters: 4,099,969.

Purpose: isolate the value of experimental 3D distances.

### Crystallographic 3D ALIGNN

- Same chemistry, spatial graph and distances as the 3D distance GNN.
- DGL line graph: local graph edges become line-graph nodes.
- Bond angles represented by cosine values in `[-1, 1]`.
- 40-bin angle RBF expansion.
- Three original `model.alignn.ALIGNNLayer` updates.
- No Coulomb attention.
- Parameters: 8,118,529.

Purpose: isolate the value of angular processing beyond distances.

### Complete original MGT Graphformer

Uses `model.graphformer.Graphformer` with:

- Original atom embedding.
- Local distance graph.
- Line graph and original ALIGNN updates.
- Fully connected wider graph with `ZiZj / rij` Coulomb features.
- Original multi-head graph attention from `model/transformer.py`.
- Post-ALIGNN `EdgeGatedGraphConv` layers.
- Two-layer feed-forward block.
- Encoder residual connection and layer normalization.
- Deterministic 10-dimensional Laplacian positional encoding.
- Global average pooling and scalar pKD head.
- Parameters: 13,702,241.

### Masked-pretrained ALIGNN and MGT

`utils.masker.MaskAtom` zeros 20% of atom feature vectors. The encoder and a temporary linear decoder reconstruct the original atom feature vectors with mean squared error for 30 epochs. Only the training partition is used. The decoder is discarded, and the pretrained encoder is fine-tuned on pKD.

The original `pre-training.py` is Graphformer-specific and applies a sigmoid output before `BCEWithLogitsLoss`, which would apply the sigmoid logic twice. The matched experiment therefore reuses the original `MaskAtom` transform but uses a single, consistent MSE reconstruction objective for both ALIGNN and MGT.

## Results

All values below are compound-level test metrics for seed 123.

### Unmasked model comparison

| Model | Split | MAE ↓ | RMSE ↓ | R² ↑ | Pearson ↑ | Spearman ↑ |
|---|---|---:|---:|---:|---:|---:|
| Morgan MLP | Random | 0.416 | 0.534 | 0.523 | 0.732 | 0.619 |
| 2D GNN | Random | 0.395 | 0.502 | 0.577 | 0.771 | 0.697 |
| 3D distance GNN | Random | 0.388 | 0.488 | 0.602 | 0.789 | **0.752** |
| 3D ALIGNN | Random | 0.379 | **0.474** | **0.624** | **0.802** | 0.737 |
| Complete MGT | Random | **0.367** | 0.479 | 0.616 | 0.787 | 0.693 |
| Morgan MLP | Scaffold | 0.572 | 0.716 | -0.007 | 0.503 | 0.269 |
| 2D GNN | Scaffold | 0.519 | 0.674 | 0.110 | 0.489 | 0.307 |
| **3D distance GNN** | Scaffold | **0.515** | **0.655** | **0.159** | **0.547** | **0.360** |
| 3D ALIGNN | Scaffold | 0.545 | 0.691 | 0.063 | 0.473 | 0.290 |
| Complete MGT | Scaffold | 0.578 | 0.757 | -0.125 | 0.354 | 0.132 |

The controlled random progression is:

```text
Morgan → 2D GNN → 3D distance GNN → 3D ALIGNN
RMSE  0.534     0.502             0.488       0.474
```

Distances provide a consistent improvement on both splits. Angles improve random interpolation but overfit unseen scaffolds. Unmasked MGT does not justify its extra capacity on the scaffold test.

### Effect of atom masking

| Model | Split | Masking | MAE ↓ | RMSE ↓ | R² ↑ | Pearson ↑ | Spearman ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| ALIGNN | Random | No | 0.379 | 0.474 | 0.624 | 0.802 | **0.737** |
| ALIGNN | Random | Yes | **0.377** | **0.467** | **0.635** | **0.805** | 0.726 |
| MGT | Random | No | **0.367** | 0.479 | 0.616 | 0.787 | **0.693** |
| MGT | Random | Yes | 0.368 | **0.459** | **0.648** | **0.812** | 0.685 |
| ALIGNN | Scaffold | No | **0.545** | **0.691** | **0.063** | **0.473** | 0.290 |
| ALIGNN | Scaffold | Yes | 0.559 | 0.716 | -0.005 | 0.468 | **0.293** |
| MGT | Scaffold | No | 0.578 | 0.757 | -0.125 | 0.354 | 0.132 |
| MGT | Scaffold | Yes | **0.513** | **0.651** | **0.170** | **0.526** | **0.354** |

Masking improves MGT substantially, especially on the scaffold split. It does not improve ALIGNN scaffold generalisation.

### Final ranking for scaffold generalisation

| Model | Scaffold RMSE ↓ | Scaffold R² ↑ |
|---|---:|---:|
| **Masked complete MGT** | **0.651** | **0.170** |
| 3D distance GNN | 0.655 | 0.159 |
| 2D GNN | 0.674 | 0.110 |
| Unmasked ALIGNN | 0.691 | 0.063 |
| Masked ALIGNN | 0.716 | -0.005 |
| Morgan MLP | 0.716 | -0.007 |
| Unmasked complete MGT | 0.757 | -0.125 |

Masked MGT is numerically best, but the RMSE advantage over the 3D distance GNN is only 0.004 pKD. Multiple seeds and uncertainty estimates are required before claiming a statistically meaningful difference. The distance GNN remains the best efficiency–performance choice.

## Repository tree and call map

Generated `.bin`, `.pt`, `.ckpt`, structure and cache files are abbreviated.

```text
MGT/
├── README.md                         # This scientific and reproducibility guide
├── model/
│   ├── alignn.py                     # Original ALIGNN and edge-gated graph layers
│   ├── transformer.py                # Original Coulomb multi-head graph attention
│   ├── graphformer.py                # Original complete MGT encoder and regressor
│   ├── ligand_gnn.py                 # Controlled ligand-only 2D GNN
│   ├── ligand_3d_gnn.py              # Controlled crystallographic distance GNN
│   └── ligand_3d_alignn.py           # Controlled crystallographic ALIGNN
├── modules/
│   └── modules.py                    # Original MLPLayer and RBFExpansion utilities
├── utils/
│   ├── datasets.py                   # Original MGT StructureDataset + OpenBind compatibility
│   ├── masker.py                     # Original random atom-feature masking transform
│   ├── molecular_features.py         # RDKit atom and bond feature definitions
│   ├── openbind_ligand_dataset.py    # SDF → controlled 2D/3D DGL graphs
│   ├── audit_openbind_structure_files.py
│   │                                  # Audit 925 metadata rows against structure files
│   ├── prepare_openbind_ligand_mgt.py # Curate labels/poses and freeze random/scaffold splits
│   ├── preprocess_openbind_mgt_graphs.py
│   │                                  # Build 621 original-MGT graph triplets on disk
│   ├── prepare_openbind_original_entrypoints.py
│   │                                  # Create leak-free roots for training.py/testing.py
│   └── make_atom_init.py              # Original atom-initializer generation utility
├── train_openbind_baseline.py        # Morgan, 2D GNN, 3D GNN and 3D ALIGNN trainer
├── train_openbind_mgt.py             # Matched complete-MGT trainer
├── train_openbind_masked.py          # Mask pretraining → matched ALIGNN/MGT fine-tuning
├── pre-training.py                   # Original MGT masking-pretraining entry point
├── training.py                       # Original Fabric MGT training entry point
├── testing.py                        # Original Fabric MGT testing entry point
├── run.py                            # Original checkpoint inference entry point
├── OpenBind_EV-A71_2A/
│   ├── OpenBind_EV-A71_2A/           # Downloaded OpenBind release and structure folders
│   ├── experiment_a_ligand_mgt/
│   │   ├── atom_init.json            # 90-dimensional original MGT atom features
│   │   ├── id_prop.csv               # Structure ID,pKD for StructureDataset
│   │   ├── raw/                      # Curated ligand-only crystallographic PDB files
│   │   ├── curated/                  # Included compounds/structures and exclusions
│   │   ├── splits/                   # Frozen random and scaffold manifests
│   │   ├── processed/                # Frozen original-MGT DGL graph triplets
│   │   └── reports/                  # Dataset, checksum and graph audit metadata
│   └── original_entrypoints_random_seed_123/
│       ├── train_validation/         # 528-structure root for original training.py
│       └── test/                     # Untouched 93-structure root for testing.py
└── output/
    ├── openbind_baselines/           # Four unmasked baseline result trees
    ├── openbind_full_mgt/            # Matched unmasked complete-MGT results
    ├── openbind_masked_3d_alignn/    # Masked ALIGNN results
    ├── openbind_masked_mgt/          # Masked complete-MGT results
    └── openbind_original_entrypoints/# Original training.py/testing.py run
```

### Runtime call graph

```text
train_openbind_baseline.py
├── MorganMLP + RDKit Morgan generator
├── OpenBindGraphDataset
│   ├── molecular_features.atom_features
│   └── molecular_features.bond_features
├── Ligand2DGNN
│   └── model.alignn.EdgeGatedGraphConv
├── Ligand3DGNN
│   ├── modules.RBFExpansion
│   └── model.alignn.EdgeGatedGraphConv
└── Ligand3DALIGNN
    ├── utils.datasets.compute_bond_cosines
    ├── modules.RBFExpansion
    └── model.alignn.ALIGNNLayer

train_openbind_mgt.py
├── utils.datasets.StructureDataset
│   ├── processed/*.bin
│   └── deterministic Laplacian PE
└── model.graphformer.Graphformer
    ├── model.transformer.multiheaded
    ├── model.alignn.ALIGNNLayer
    ├── model.alignn.EdgeGatedGraphConv
    └── modules.{MLPLayer,RBFExpansion}

train_openbind_masked.py
├── utils.masker.MaskAtom
├── ALIGNN encoder or Graphformer encoder
├── temporary atom-feature reconstruction decoder
└── unchanged matched trainer after decoder removal
```

### Source file reference

| File | Responsibility | Called by |
|---|---|---|
| `model/alignn.py` | Original edge-gated convolution and angle–edge–atom update | Graphformer and all learned GNN baselines |
| `model/transformer.py` | Original wider-graph multi-head attention | `graphformer.py` |
| `model/graphformer.py` | Complete MGT composition and regression head | Original scripts, `train_openbind_mgt.py`, masked MGT |
| `model/ligand_gnn.py` | 2D atom–bond model | `train_openbind_baseline.py` |
| `model/ligand_3d_gnn.py` | Distance-RBF spatial model | `train_openbind_baseline.py` |
| `model/ligand_3d_alignn.py` | Line-graph angular model using original ALIGNNLayer | Baseline and masked trainers |
| `modules/modules.py` | MLP blocks and Gaussian RBF expansion | Graphformer and 3D baselines |
| `utils/datasets.py` | Original structure-to-three-graph conversion and batching | Original scripts and matched MGT |
| `utils/masker.py` | Randomly zero selected node feature vectors and retain labels | Original and matched masking workflows |
| `utils/molecular_features.py` | Exact RDKit feature vocabularies and encoders | `openbind_ligand_dataset.py` |
| `utils/openbind_ligand_dataset.py` | Load reference SDFs and create controlled graphs | Baseline trainer |
| `utils/audit_openbind_structure_files.py` | Resolve metadata-to-structure mapping and checksums | Dataset preparation stage |
| `utils/prepare_openbind_ligand_mgt.py` | Curation, PDB export, grouping and split generation | Run once before experiments |
| `utils/preprocess_openbind_mgt_graphs.py` | Store Graphformer graph triplets as `.bin` | Run once before MGT training |
| `utils/prepare_openbind_original_entrypoints.py` | Separate development and test roots | Original-script reproduction |
| `train_openbind_baseline.py` | Shared matched training/evaluation for four baselines | CLI entry point |
| `train_openbind_mgt.py` | Matched full-MGT training/evaluation | CLI and masked wrapper |
| `train_openbind_masked.py` | Shared mask-pretrain/fine-tune experiment | CLI entry point |
| `pre-training.py` | Original Fabric masking experiment | Original CLI; not used for matched results |
| `training.py` | Original Fabric supervised training | Original CLI reproduction |
| `testing.py` | Original Fabric MAE evaluation | Original CLI reproduction |
| `run.py` | Original inference on unlabeled structures | Original CLI |

## Original code and modifications

### Source annotation convention

Every Python module, class, function and method has a docstring. Meaningful tensor, graph, data-selection, normalization, masking, optimization, checkpoint and evaluation operations have adjacent inline comments. Multiline signatures, closing delimiters and repetitive `argparse` declarations are documented as logical groups so comments do not obscure executable code. The annotations describe behaviour only; they do not alter the original mathematical operators.

The original scientific operators remain in their original files:

- `model/alignn.py`: `ALIGNNLayer` and `EdgeGatedGraphConv`.
- `model/transformer.py`: multi-head graph attention.
- `model/graphformer.py`: full encoder ordering, feed-forward block, residual, normalization and pooling.
- `modules/modules.py`: MLP and RBF calculations.
- `utils/datasets.compute_bond_cosines`: angular calculation.

Only two compatibility corrections were made inside original files:

1. **Evaluation dropout correction** in `model/transformer.py`: functional dropout now receives `training=self.training`, so attention dropout is disabled during validation/testing and remains active during training.
2. **OpenBind structure compatibility** in `utils/datasets.py`: non-periodic ligand PDBs can fall back to RDKit parsing; Laplacian positional-encoding signs are generated deterministically per structure and seed.

New files add:

- RDKit atom/bond chemistry for controlled baselines.
- Crystallographic SDF loading and spatial edges.
- Scalar pKD prediction heads.
- Frozen manifest selection and compound leakage checks.
- Train-only target normalization.
- Standard regression metrics and compound aggregation.
- OpenBind curation, split and graph preprocessing.
- Matched atom-masking pretraining.

Protein sequence fusion, protein embeddings, atom masking during ordinary supervised baselines, and unrelated material-property outputs are not used.

## Installation

The completed runs used Linux, Python 3.11 and CUDA. Verified package versions in the working environment were:

| Library | Version | Role |
|---|---|---|
| PyTorch | 2.11.0+cu128 | Tensor operations, optimization and neural networks |
| DGL | 2.4.0+cu124 | Molecular, line and Coulomb graphs |
| RDKit | 2025.09.5 | Molecule parsing, canonical identity, scaffolds and fingerprints |
| Lightning | 2.6.5 | Original Fabric training/testing scripts |
| NumPy | 2.4.6 | Numerical arrays |
| SciPy | 1.17.1 | Pearson and Spearman statistics |
| pandas | 3.0.3 | Original atom-initializer utility |
| pymatgen | installed in `mgt` | Original CIF/PDB/XYZ molecule loading |

Create and activate an environment:

```bash
conda create -n mgt python=3.11 -y
conda activate mgt
```

Install PyTorch appropriate for the machine’s CUDA driver. For CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining packages:

```bash
conda install -c conda-forge rdkit pymatgen lightning pandas scipy numpy -y
```

Install a DGL build compatible with the installed PyTorch/CUDA combination. The experiments used DGL `2.4.0+cu124`; follow the DGL installation selector if that exact build is unavailable for the local platform.

Verify the environment:

```bash
python - <<'PY'
import torch, dgl, rdkit, lightning, numpy, scipy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("dgl", dgl.__version__)
print("rdkit", rdkit.__version__)
print("lightning", lightning.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
PY
```

## Reproducing the dataset

Run commands from the repository root.

### 1. Audit the downloaded release

```bash
python utils/audit_openbind_structure_files.py \
  --dataset_root OpenBind_EV-A71_2A/OpenBind_EV-A71_2A
```

This maps metadata records to structure directories, audits reference SDF/PDB availability and writes structure manifests/checksums.

### 2. Curate ligand structures and create splits

```bash
python utils/prepare_openbind_ligand_mgt.py \
  --dataset_root OpenBind_EV-A71_2A/OpenBind_EV-A71_2A \
  --benchmark_reference ../EV-A71_2A_benchmark/affinity/reference/fragalysis_compound_reference.csv \
  --output_root OpenBind_EV-A71_2A/experiment_a_ligand_mgt
```

This writes:

- `curated/openbind_ligand_structures.csv`
- `curated/openbind_compounds.csv`
- `curated/excluded_records.csv`
- ligand-only crystallographic PDBs under `raw/`
- `id_prop.csv`
- random and scaffold split manifests
- dataset metadata and checksums

### 3. Build frozen original-MGT graphs

```bash
python -m utils.preprocess_openbind_mgt_graphs \
  --data_root OpenBind_EV-A71_2A/experiment_a_ligand_mgt \
  --seed 123
```

This creates one `.bin` file per curated structure in `processed/`. Each file stores local, line and Coulomb graphs produced by the original `StructureDataset` graph constructor.

## Reproducing the experiments

All commands default to seed 123 and use CUDA automatically when available.

### Four controlled baselines

```bash
python train_openbind_baseline.py --model morgan_mlp --split_method random --device auto
python train_openbind_baseline.py --model morgan_mlp --split_method scaffold --device auto

python train_openbind_baseline.py --model 2d_gnn --split_method random --device auto
python train_openbind_baseline.py --model 2d_gnn --split_method scaffold --device auto

python train_openbind_baseline.py --model 3d_gnn --split_method random --device auto
python train_openbind_baseline.py --model 3d_gnn --split_method scaffold --device auto

python train_openbind_baseline.py --model 3d_alignn --split_method random --device auto
python train_openbind_baseline.py --model 3d_alignn --split_method scaffold --device auto
```

### Matched complete MGT

```bash
python train_openbind_mgt.py --split_method random --seed 123 --device auto
python train_openbind_mgt.py --split_method scaffold --seed 123 --device auto
```

These commands use the complete original `Graphformer`, frozen partitions, matched target normalization, early stopping and complete metrics.

### Masked-pretrained ALIGNN and MGT

```bash
python train_openbind_masked.py \
  --model 3d_alignn \
  --split_method random \
  --pretrain_epochs 30 \
  --mask_rate 0.2 \
  --device auto

python train_openbind_masked.py \
  --model 3d_alignn \
  --split_method scaffold \
  --pretrain_epochs 30 \
  --mask_rate 0.2 \
  --device auto

python train_openbind_masked.py \
  --model mgt \
  --split_method random \
  --pretrain_epochs 30 \
  --mask_rate 0.2 \
  --device auto

python train_openbind_masked.py \
  --model mgt \
  --split_method scaffold \
  --pretrain_epochs 30 \
  --mask_rate 0.2 \
  --device auto
```

### Original `training.py` and `testing.py`

Prepare non-overlapping roots:

```bash
python utils/prepare_openbind_original_entrypoints.py
```

Train for 100 epochs with the original Fabric workflow:

```bash
python training.py \
  --root OpenBind_EV-A71_2A/original_entrypoints_random_seed_123/train_validation \
  --model_path output/openbind_original_entrypoints/random/seed_123/checkpoints \
  --save_dir output/openbind_original_entrypoints/random/seed_123/logs \
  --run_name training_py \
  --n_devices 1 \
  --accelerator cuda \
  --process 0 \
  --periodic 0 \
  --out_dims 1 \
  --out_names pKD \
  --epochs 100 \
  --batch_size 2 \
  --n_cum 8 \
  --train_split 0.8 \
  --val_split 0.2 \
  --num_layers 1 \
  --n_mha 1 \
  --n_alignn 3 \
  --n_gnn 3
```

Test the lowest-validation checkpoint on the untouched test root:

```bash
python testing.py \
  --root OpenBind_EV-A71_2A/original_entrypoints_random_seed_123/test \
  --model_path output/openbind_original_entrypoints/random/seed_123/checkpoints \
  --model_name lowest.ckpt \
  --save_dir output/openbind_original_entrypoints/random/seed_123/logs \
  --run_name testing_py \
  --n_devices 1 \
  --accelerator cuda \
  --process 0 \
  --periodic 0 \
  --out_dims 1 \
  --out_names pKD \
  --num_layers 1 \
  --n_mha 1 \
  --n_alignn 3 \
  --n_gnn 3
```

The completed original run trained in approximately 24.3 minutes and produced compound-level random-test RMSE 0.475 and R² 0.622. Its internal train/validation division is made by the original `random_split` without a supplied generator, so that internal membership is not deterministic across fresh executions. The independent test root remains frozen.

## Output files

Every matched experiment directory contains:

```text
best_*.pt                       # Best validation-loss model weights
history.csv                     # Epoch, training loss, validation loss and LR
test_structure_predictions.csv # One row per crystal structure
test_compound_predictions.csv  # Mean prediction per official compound group
metrics.json                    # Config, counts, architecture and all metrics
```

Masked `metrics.json` files additionally contain:

```json
{
  "masked_pretraining": {
    "transform": "original utils.masker.MaskAtom",
    "mask_rate": 0.2,
    "epochs": 30,
    "objective": "MSE atom-feature reconstruction",
    "training_partition_only": true,
    "history": []
  }
}
```

Result locations:

- `output/openbind_baselines/<model>/<split>/seed_123/`
- `output/openbind_full_mgt/<split>/seed_123/`
- `output/openbind_masked_3d_alignn/3d_alignn/<split>/seed_123/`
- `output/openbind_masked_mgt/<split>/seed_123/`
- `output/openbind_original_entrypoints/random/seed_123/`

## Implementation walkthrough

### Data flow

1. The audit script resolves each metadata row to the official structure directory.
2. The preparation script applies quality rules, reads `ligand_ref.sdf`, preserves its coordinates and writes ligand-only PDB files.
3. Canonical SMILES identify compounds; Bemis–Murcko SMILES define scaffold groups.
4. Compound groups are allocated to frozen train/validation/test manifests.
5. Controlled baselines read the reference SDF directly and construct only the graph information required by that ablation.
6. Complete MGT reads frozen `.bin` triplets created by the original graph constructor.
7. Targets are normalized with training labels only.
8. The best validation checkpoint is evaluated once on train, validation and held-out test partitions.
9. Repeated crystallographic structures are averaged to produce compound-level predictions.

### Controlled graph construction

`OpenBindGraphDataset.__getitem__` loads a fresh RDKit molecule from the structure’s `ligand_ref.sdf`. In 2D mode it creates directed chemical-bond edges only. In 3D modes it adds directed spatial edges within the cutoff, retains chemical-bond feature vectors where bonds exist, supplies zero bond chemistry for non-bonded spatial edges, and stores distance and displacement tensors.

### Original MGT graph construction

`StructureDataset._construct_graph` creates:

- `G`: local neighbours, atom features, distances, displacement vectors and Laplacian PE.
- `LG`: the line graph of `G`, with angle cosine on every valid edge pair.
- `FG`: the wider graph with Coulomb edge feature `ZiZj / rij`.

`Graphformer.forward` embeds these features, applies the original encoder stack, averages atom representations and predicts one scalar.

### Masking flow

1. `MaskAtom` samples at least one node and approximately 20% of each batched graph’s nodes.
2. It copies the original selected feature vectors into a node subgraph.
3. It replaces those vectors with zeros in the encoder input.
4. ALIGNN or MGT generates contextual atom representations.
5. A temporary decoder reconstructs the held-out features.
6. After 30 epochs, the decoder is discarded.
7. The encoder weights initialize normal supervised affinity fine-tuning.

## Limitations

- Results currently use one seed. Run at least five seeds and report mean ± standard deviation or confidence intervals.
- The masked-MGT versus distance-GNN scaffold RMSE difference is only 0.004 pKD and is not established statistically.
- The dataset is small relative to the 8.1M ALIGNN and 13.7M MGT models.
- Multiple crystal structures can weight compounds with repeated structures more heavily during structure-level training, although compound identities never cross partitions.
- Ligand-only models cannot explicitly represent protein–ligand contacts, water networks, protonation coupling or receptor conformational change.
- Crystal poses are experimental inputs; performance is not equivalent to prediction from SMILES alone or from docked/generated poses.
- The original Fabric training script has different loss, batching and internal split behaviour from the matched trainer; its result is a compatibility reproduction, not a controlled replacement for the matched comparison.

## Final conclusion

Crystallographic distances robustly improve single-target affinity prediction over fixed fingerprints and 2D learned graphs. Angular ALIGNN processing improves random-split interpolation but does not improve unseen-scaffold performance. Complete MGT overfits when trained only on affinity labels, but atom-masking pretraining substantially improves its scaffold result from RMSE 0.757/R² -0.125 to RMSE 0.651/R² 0.170. Masked MGT is numerically strongest, while the crystallographic 3D distance GNN provides nearly identical scaffold performance with less than one-third of the parameters.

## Data license and acknowledgement

The OpenBind dataset is released under CC0 1.0 Universal. Credit the OpenBind Consortium and the EV-A71/CVA16 2A protease structure–affinity release when publishing results derived from this repository.
