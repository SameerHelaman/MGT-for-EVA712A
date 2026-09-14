# OpenBind ligand-affinity prediction with MGT

This project compares seven molecular models for predicting the binding affinity
of ligands in the OpenBind EV-A71 2A dataset. It uses Morgan fingerprints,
molecular graphs and crystallographic ligand geometry, with optional
atom-feature masking pretraining.

The target is **experimental pKD**, `pKD = −log10(KD in mol/L)`, not pKa.
Only ligand information is used; protein coordinates and sequences are not inputs.

This README explains how to install the dependencies, prepare the raw data,
run the experiments and produce the tables and figures.

This is a **code-only repository**. Raw data, prepared datasets, split files,
trained checkpoints and generated tables/figures are not included. Download
the source files and run the steps below to recreate them. The three notebooks
are published without saved outputs; run them after their required preparation
or training steps.

## Contents

- [Installation](#installation)
- [Download the project and raw data](#download-the-project-and-raw-data)
- [Prepare the dataset](#prepare-the-dataset)
- [Create the splits](#create-the-splits)
- [Run training and testing](#run-training-and-testing)
- [Models and training procedure](#models-and-training-procedure)
- [Output files](#output-files)
- [Generate tables and figures](#generate-tables-and-figures)
- [Project structure and file guide](#project-structure-and-file-guide)
- [Legacy files and practical notes](#legacy-files-and-practical-notes)
- [AI acknowledgement](#ai-acknowledgement)

## Installation

Use a Linux NVIDIA GPU node, not a CPU-only or login node. The project was run
on `chegpu004` with Python 3.11, an NVIDIA RTX 5080, PyTorch 2.11.0+cu128 and
DGL 2.4.0+cu124. A new machine needs a suitable NVIDIA driver and an available
GPU allocation. Follow your institution's instructions for allocating a GPU;
this repository does not supply a scheduler job script.

Follow the sections in order to create a new environment, download the code and
raw data, prepare the dataset, and run training, testing and analysis.

### Create the GPU environment

Install or initialize Conda/Miniforge first. The commands below target Linux
x86_64, Python 3.11 and CUDA 12.8-capable NVIDIA hardware; the PyTorch wheel also
requires glibc 2.28 or newer.

```bash
conda create -n mgt-openbind-gpu -c conda-forge \
  python=3.11.15 pip git git-lfs curl unzip -y
conda activate mgt-openbind-gpu

conda install -c dglteam/label/th24_cu124 -c conda-forge \
  "dgl=2.4.0.th24.cu124=py311_0" -y

python -m pip install "torch==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  numpy==2.4.6 scipy==1.17.1 pandas==3.0.3 rdkit==2025.9.5 \
  pymatgen==2026.5.4 pymatgen-core==2026.4.16 monty==2026.5.18 \
  matplotlib==3.10.9 networkx==3.6.1 packaging==26.2 psutil==7.2.2 \
  pydantic==2.12.5 PyYAML==6.0.3 requests==2.34.2 tqdm==4.68.4 \
  Pillow==12.3.0 setuptools==78.1.0 \
  jupyterlab==4.6.3 ipykernel==7.3.0 nbconvert==7.17.1 adjustText==1.4.0

export DGLBACKEND=pytorch
```


| Libraries | Purpose |
|---|---|
| PyTorch, DGL | Neural networks, GPU computation and graph message passing. |
| RDKit | SDF parsing, atom/bond features, fingerprints and scaffold identification. |
| pymatgen, Monty | Structure loading and MGT graph construction. |
| NumPy, pandas, SciPy | Arrays, data manifests and evaluation statistics. |
| Matplotlib, Pillow, adjustText | Figures, molecular illustrations and plot labels. |
| JupyterLab, ipykernel, nbconvert | Running and exporting notebooks. |


### Check the GPU installation

Run these commands inside the activated `mgt-openbind-gpu` environment on the GPU node:

```bash
nvidia-smi

python - <<'PY'
import torch
import dgl
import dgl.function as fn
import numpy, pandas, scipy, rdkit, pymatgen, matplotlib

assert torch.cuda.is_available(), "CUDA is unavailable; check your GPU allocation."
g = dgl.graph(([0, 1], [1, 0])).to("cuda")
x = torch.randn(2, 8, device="cuda", requires_grad=True)
g.ndata["x"] = x
g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
g.ndata["h"].square().mean().backward()
torch.cuda.synchronize()
assert x.grad is not None
print("PyTorch:", torch.__version__, "| DGL:", dgl.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU graph forward/backward check passed.")
PY
```

Stop if this fails. A successful PyTorch import alone does not establish that
DGL can run on the GPU. All training commands below explicitly use
`--device cuda`.


## Download the project and raw data

### Clone the code and official affinity reference

With the environment above activated, run the following from a directory where
you want a new project workspace. The project directory is deliberately named
`MGT`, matching the relative paths used by the scripts and notebooks.

```bash
mkdir -p mgt-openbind-workspace
cd mgt-openbind-workspace

git clone --depth 1 https://github.com/SameerHelaman/MGT-for-EVA712A.git MGT
git clone https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark.git EV-A71_2A_benchmark
git -C EV-A71_2A_benchmark checkout --detach 86e5c12da518d749c33cfa9dcb6ae8eae1b804f9

cd MGT

printf '%s\n' \
  '1ff937a952a9e11f9783c9de362b738a96e55e8b06ead2e41828ef90d8538fc8  ../EV-A71_2A_benchmark/affinity/reference/fragalysis_compound_reference.csv' \
  | sha256sum --check - || exit 1
```

The benchmark checkout is pinned to the revision used for this project. Its
reference CSV is needed to select and verify the affinity labels; the raw
structure archive alone is not sufficient. Keep this sibling-directory layout
and run the remaining shell commands from `MGT/`.

### 1. Download and extract the raw release

The source archive is the [OpenBind release on Zenodo](https://doi.org/10.5281/zenodo.20026661).

```bash
mkdir -p downloads OpenBind_EV-A71_2A

curl --fail --location --retry 3 \
  https://zenodo.org/api/records/20026661/files/OpenBind_EV-A71_2A.zip/content \
  --output downloads/OpenBind_EV-A71_2A.zip

printf '%s\n' \
  '860a4979d0ba9decaa2bfaa933c1d217  downloads/OpenBind_EV-A71_2A.zip' \
  | md5sum --check - || exit 1

unzip -o downloads/OpenBind_EV-A71_2A.zip -d OpenBind_EV-A71_2A
```

This produces `OpenBind_EV-A71_2A/OpenBind_EV-A71_2A/`, containing
`EV-A71_2A_metadata.csv` and `structures/`. The repeated directory name is
intentional: the controlled graph loader expects reference SDFs there.
Extraction with `-o` replaces files at that location, so use it for a fresh
rebuild, not merely to inspect an existing dataset.

### 2. Obtain the MGT atom-feature reference

`atom_init.json` is required. It supplies fixed 90-dimensional elemental
features, not learned affinity values. It cannot be derived from the affinity
CSV alone.

```bash
mkdir -p examples/example_data

curl --fail --location --retry 3 \
  https://raw.githubusercontent.com/MolecularGraphTransformer/MGT/main/examples/example_data/atom_init.json \
  --output examples/example_data/atom_init.json

printf '%s\n' \
  '93d7b2c2381f8dd9a465f4428e2d3fb6f72b3eadf3c6752591e7924b835809cd  examples/example_data/atom_init.json' \
  | sha256sum --check - || exit 1
```

## Prepare the dataset

Define the paths once. If starting a new terminal, activate the environment,
return to `MGT/`, and define them again.

```bash
OPENBIND_RAW_ROOT="OpenBind_EV-A71_2A/OpenBind_EV-A71_2A"
OPENBIND_DATA_ROOT="OpenBind_EV-A71_2A/experiment_a_ligand_mgt"
OPENBIND_REFERENCE="../EV-A71_2A_benchmark/affinity/reference/fragalysis_compound_reference.csv"
```

### 1. Audit the raw structure files

```bash
python -m utils.audit_openbind_structure_files \
  --dataset_root "$OPENBIND_RAW_ROOT"
```

This maps metadata records to structure files and records their availability
and checksums in `$OPENBIND_RAW_ROOT/reports/`. Run it before curation.

### 2. Curate the ligand dataset

```bash
python -m utils.prepare_openbind_ligand_mgt \
  --dataset_root "$OPENBIND_RAW_ROOT" \
  --benchmark_reference "$OPENBIND_REFERENCE" \
  --output_root "$OPENBIND_DATA_ROOT"
```

This checks affinity/reference agreement and structure-quality rules, reads
`ligand_ref.sdf`, and exports ligand-only PDB files without regenerating or
minimizing their coordinates. It also creates compound identities, scaffold
groups and the classic train/validation/test splits.

For the specified release, the expected preparation is **925 source records →
621 retained structures representing 474 compounds**, with 304 excluded
records. Check the reports if your counts differ.

| Generated file or directory, under `$OPENBIND_DATA_ROOT` | Contents |
|---|---|
| `curated/openbind_ligand_structures.csv` | Retained structure records, identities, labels and source paths. |
| `curated/openbind_compounds.csv` | Compound-level records. |
| `curated/excluded_records.csv` | Excluded records and reasons. |
| `raw/` | Coordinate-preserving, ligand-only PDBs used by MGT. |
| `id_prop.csv` | Structure identifiers and pKD targets for the MGT dataset. |
| `atom_init.json` | Copy of the fixed MGT elemental-feature reference. |
| `splits/` | Classic random and scaffold partitions. |
| `reports/dataset_metadata.json` | Preparation rules, counts and provenance. |

### 3. Build the MGT graph cache

```bash
python -m utils.preprocess_openbind_mgt_graphs \
  --data_root "$OPENBIND_DATA_ROOT" \
  --seed 123 --local_radius 8.0 --max_neighbors 12 --num_pe_fea 10 \
  --overwrite
```

This writes one `.bin` per structure under `processed/`. Each contains the
local graph, its line graph and the wider Coulomb-feature graph. Expect 621
cache files; inspect `reports/processed_graph_summary.json` and
`reports/processed_graph_manifest.csv`.

`--overwrite` rebuilds the cache and is intentional for a from-scratch run.
Without it, existing files can be reused even if graph settings have changed.
The Morgan and controlled GNN trainers use the original reference SDFs; only
the MGT variants use this three-graph cache.

## Create the splits

### Classic random and scaffold splits

The curation command already creates:

```text
splits/random_seed_123.csv
splits/scaffold_seed_123.csv
```

Each divides the 621 structure rows into 435 training, 93 validation and
93 test rows. Repeated structures of a compound remain together. The scaffold
split additionally separates Bemis–Murcko scaffold groups.

### Random and scaffold five-fold CV

```bash
python -m utils.generate_random_cv \
  --data_root "$OPENBIND_DATA_ROOT" \
  --source "$OPENBIND_DATA_ROOT/splits/random_seed_123.csv" \
  --output_dir "$OPENBIND_DATA_ROOT/cv_random" \
  --folds 5 --seed 123

python -m utils.generate_scaffold_cv \
  --data_root "$OPENBIND_DATA_ROOT" \
  --source "$OPENBIND_DATA_ROOT/splits/scaffold_seed_123.csv" \
  --output_dir "$OPENBIND_DATA_ROOT/cv" \
  --folds 5 --seed 123
```

These use all curated records in their source manifests, not just the classic
training subset. Random CV groups by compound and balances affinity; scaffold
CV keeps scaffold groups separate. Each outer fold has its own training,
validation and test membership.

The directories contain `cv_metadata.json`, an overall assignment CSV and
five trainer-ready manifests:

```text
cv_random/random_grouped_5fold_seed_123.csv
cv_random/random_cv5_fold_0_seed_123.csv       # folds 0 through 4
cv/scaffold_grouped_5fold_seed_123.csv
cv/scaffold_cv5_fold_0_seed_123.csv            # folds 0 through 4
```

Read `cv_metadata.json` for counts and leakage checks. Keep seed 123:
preparation filenames and notebooks are written around that experiment.

## Run training and testing

The training scripts select the best validation checkpoint and then evaluate
the held-out test partition automatically. There is no separate `testing.py`
step for these experiments.

### 1. Random five-fold CV: all seven models

```bash
python run_openbind_cv.py \
  --cv_method random --models all --folds 0 1 2 3 4 \
  --data_root "$OPENBIND_DATA_ROOT" \
  --cv_dir "$OPENBIND_DATA_ROOT/cv_random" \
  --output_root output/openbind_random_cv \
  --seed 123 --device cuda --batch_size 32 \
  --max_epochs 200 --early_stopping_patience 10 \
  --pretrain_epochs 30 --mask_rate 0.2
```

### 2. Scaffold five-fold CV: all seven models

```bash
python run_openbind_cv.py \
  --cv_method scaffold --models all --folds 0 1 2 3 4 \
  --data_root "$OPENBIND_DATA_ROOT" \
  --cv_dir "$OPENBIND_DATA_ROOT/cv" \
  --output_root output/openbind_scaffold_cv \
  --seed 123 --device cuda --batch_size 32 \
  --max_epochs 200 --early_stopping_patience 10 \
  --pretrain_epochs 30 --mask_rate 0.2
```

Together these commands perform 70 supervised fits: seven configurations × two
CV methods × five folds. The two masked configurations also perform a
training-fold-only pretraining stage before each supervised fit. The runner
launches the models sequentially using the active Python environment and writes
summaries after the requested runs.

To run a subset, replace `--models all` with space-separated model keys, for
example `--models 3d_gnn adapted_mgt masked_mgt`. To select one outer fold,
use `--folds 0`. Such a subset is not the complete five-fold experiment.

Add `--resume` to the same command to skip runs that already have
`metrics.json` and `test_compound_predictions.csv`. This is a completed-run
skip, not an optimizer/checkpoint restart. It does not verify that previous
settings match, so use a new `--output_root` when changing an experiment.

### 3. Classic train/validation/test experiments

These are separate experiments using `splits/*_seed_123.csv`, not prerequisites
for CV. The following loop runs all five unmasked models on both classic splits:

```bash
for OPENBIND_SPLIT in random scaffold; do
  for OPENBIND_TRAINER in \
    train_openbind_morgan_mlp.py \
    train_openbind_2d_gnn.py \
    train_openbind_3d_gnn.py \
    train_openbind_3d_alignn.py \
    train_openbind_mgt.py; do
    python "$OPENBIND_TRAINER" \
      --data_root "$OPENBIND_DATA_ROOT" \
      --split_method "$OPENBIND_SPLIT" --seed 123 --device cuda \
      --batch_size 32 --max_epochs 200 --early_stopping_patience 10 \
      || exit 1
  done
done
```

Run both masked configurations on both classic splits:

```bash
for OPENBIND_SPLIT in random scaffold; do
  python train_openbind_masked.py \
    --model 3d_alignn --split_method "$OPENBIND_SPLIT" \
    --data_root "$OPENBIND_DATA_ROOT" --seed 123 --device cuda \
    --pretrain_epochs 30 --mask_rate 0.2 \
    --batch_size 32 --max_epochs 200 --early_stopping_patience 10 \
    || exit 1

  python train_openbind_masked.py \
    --model mgt --split_method "$OPENBIND_SPLIT" \
    --data_root "$OPENBIND_DATA_ROOT" --seed 123 --device cuda \
    --pretrain_epochs 30 --mask_rate 0.2 --max_neighbors 12 \
    --batch_size 32 --max_epochs 200 --early_stopping_patience 10 \
    || exit 1
done
```

Each masked command performs **pretraining → supervised fine-tuning → testing**.
Do not run the unmasked trainer afterwards to continue it; that starts a new
model. The scripts use separate default output directories, listed below.

## Models and training procedure

| CV key | Input and implementation |
|---|---|
| `morgan_mlp` | Radius-2, 2,048-bit Morgan fingerprint; MLP with widths 2,048 → 128 → 64 → 1. |
| `2d_gnn` | RDKit atom features and chemical-bond edges; learned graph message passing. |
| `3d_gnn` | Chemical bonds plus spatial neighbours within 5 Å; distance-RBF features. |
| `3d_alignn` | The 3D graph plus a line graph encoding angles; coupled angle–edge–atom updates. |
| `adapted_mgt` | Graphformer using local, line and wider graphs; adapted to non-periodic ligands and scalar pKD prediction. |
| `masked_alignn` | The 3D ALIGNN encoder after atom-feature reconstruction pretraining. |
| `masked_mgt` | The adapted MGT encoder after atom-feature reconstruction pretraining. |

The controlled GNNs use 152-dimensional atom and 12-dimensional bond features.
Their 3D graphs select up to 32 spatial neighbours per atom in addition to
retaining chemical bonds. MGT uses a different representation: 90-dimensional
elemental features, 10-dimensional Laplacian positional encoding, an 8 Å local
graph capped at 12 neighbours, angular updates and wider-graph attention.
These models are therefore not all single-variable architectural ablations.

MGT's wider edges carry `Zi × Zj / rij`, using atomic numbers and distances.
These are **Coulomb-inspired descriptors, not physical electrostatic energies
or learned partial charges**. Geometry is encoded through graph features,
not an image of the molecule.

### Training and evaluation

Targets are normalized using the training structures' mean and population
standard deviation only. All main trainers use Adam (`lr=1e-4`,
`weight_decay=1e-5`), Huber loss (`delta=1`), batch size 32 and at most
200 epochs. Validation loss controls learning-rate reduction (factor 0.5,
patience 5), early stopping (patience 10), and checkpoint selection.

The best validation checkpoint is restored before evaluation. Predictions are
converted back to pKD and averaged across repeated structures of each compound.
Metrics are MAE, MSE, RMSE, R², Pearson and Spearman correlations. Complete CV
pools one outer-test prediction per compound across five folds.

### Masked pretraining

The masked trainer reuses `utils.masker.MaskAtom` to zero approximately 20% of
the batched atoms' complete feature vectors. A temporary linear decoder
reconstructs them from 512-dimensional encoder states using masked-node MSE
for 30 epochs, with training-partition structures only.

Connectivity, distances, angles and MGT positional/Coulomb information remain
available: this is feature masking, not atom removal. The decoder is discarded
and all encoder weights are fine-tuned for affinity. Pretraining history is
stored in `metrics.json`, not a separate reusable pretraining checkpoint.

## Output files

Every completed main training run writes:

| File | Contents |
|---|---|
| `best_*.pt` | Best validation checkpoint. MGT uses `best_openbind_mgt.pt`. |
| `history.csv` | Epoch-level training loss, validation loss and learning rate. |
| `metrics.json` | Arguments, counts, model details, normalization and evaluation metrics. Masked runs also include pretraining history. |
| `test_structure_predictions.csv` | One held-out prediction per structure. |
| `test_compound_predictions.csv` | Structure predictions averaged per compound. |

Classic results use these directories, with `<split>` equal to `random` or
`scaffold`:

```text
output/openbind_baselines/<model>/<split>/seed_123/
output/openbind_full_mgt/<split>/seed_123/
output/openbind_masked_3d_alignn/3d_alignn/<split>/seed_123/
output/openbind_masked_mgt/<split>/seed_123/
```

CV results are separated by method and outer fold:

```text
output/openbind_random_cv/             # also output/openbind_scaffold_cv/
├── run_log.json
├── fold_0/                           # through fold_4/
│   ├── unmasked/<model>/<method>/seed_123/
│   ├── masked_alignn/3d_alignn/<method>/seed_123/
│   └── masked_mgt/<method>/seed_123/
└── summary/
    ├── fold_metrics.csv
    ├── cv_summary.json
    └── <model>_out_of_fold_compound_predictions.csv
```

`fold_metrics.csv` contains the individual fold scores. `cv_summary.json`
contains aggregate summaries, and each out-of-fold CSV contains the pooled
compound predictions. Check that all five folds are present before using them
as complete CV results.

Retain `metrics.json` alongside a checkpoint: the controlled trainers save a
model state dictionary, while the MGT checkpoint also includes its scaler.
The legacy `testing.py` and `run.py` use a different checkpoint workflow;
they are not general inference commands for these `.pt` files.

## Generate tables and figures

Run the notebooks after completing both five-fold CV commands. They read saved
predictions and metrics; they do not train the models.

### Start Jupyter on the GPU node

From `MGT/`, activate the environment created during installation and start Jupyter:

```bash
conda activate mgt-openbind-gpu

python -m ipykernel install --user --name mgt-practical \
  --display-name "Python (MGT practical)"

python -m jupyter lab notebooks --no-browser --ip=127.0.0.1
```

Connect through your institution's approved SSH tunnel or remote Jupyter
service, using the URL/token printed by Jupyter. Select **Python (MGT practical)**
as the notebook kernel. Confirm `sys.executable` points to the intended Conda
environment.

### Run the notebooks in this order

The reporting workflow uses the three notebooks below. Each has concise
code-purpose markdown and comments, with shared imports collected near the top.
Run the cells in order so the setup and data-loading cells run before plotting.

| Notebook | What to do and what it produces |
|---|---|
| [openbind_results_analysis.ipynb](notebooks/openbind_results_analysis.ipynb) | Open after all CV fits finish. Run its analysis cells in order to generate tables and figures in `output/dissertation_analysis/`. |
| [openbind_results_and_discussion.ipynb](notebooks/openbind_results_and_discussion.ipynb) | Open after the analysis notebook. Displays the generated artifacts with written interpretation; it does not replace the analysis step. |
| [openbind_methodology_figures.ipynb](notebooks/openbind_methodology_figures.ipynb) | Generates workflow/model diagrams using the prepared data and splits. Most outputs go to `output/methodology_figures/`; see the compound-view exception below. |

The analysis notebook begins with a `%pip install adjustText` setup cell before
the shared imports. The package is
already installed by the setup above: skip that cell, or use **Save As** to
create `openbind_results_analysis.local.ipynb`, comment out that line in the
copy, then run all cells. Renaming the original notebooks is otherwise
unnecessary.

The written discussion, selected-model labels and some captions contain
project-specific values. Review them after new experiments; rerunning the
notebooks does not automatically rewrite all narrative claims.

### Main generated tables

These are written under `output/dissertation_analysis/tables/`:

| Filename | Contents |
|---|---|
| `table_random_cv_performance.csv` | Model comparison for random CV. |
| `table_scaffold_cv_performance.csv` | Model comparison for scaffold CV. |
| `table_all_cv_model_scores.csv` | Combined model scores for both CV designs. |
| `table_all_cv_fold_scores.csv` | Individual model/fold scores. |
| `table_random_cv_best_model_per_fold.csv` | Lowest-test-RMSE model in each random outer fold. |
| `table_scaffold_cv_best_model_per_fold.csv` | Lowest-test-RMSE model in each scaffold outer fold. |

The last two tables are **post-hoc comparisons of test results**, not a
validation-based procedure for choosing a model. Analysis figures go to
`output/dissertation_analysis/figures/`; `artifact_manifest.csv` records the
main saved artifacts.

### If you change input or output paths

The commands above use the notebook defaults, so no path edits are needed for
that layout. For renamed experiment directories, change the following settings
in a working copy of the notebooks:

| Notebook/location | Settings to change |
|---|---|
| Analysis: initial configuration cell | `RESULT_ROOTS["random"]`, `RESULT_ROOTS["scaffold"]`, `DATA_ROOT` and, if needed, `ANALYSIS_ROOT`, `FIGURE_ROOT`, `TABLE_ROOT`. |
| Analysis: both later cells beginning `def find_project_root()` | Update their hard-coded root-search paths, `CV_PATHS` and `TABLE_ROOT`. In the first, `CV_PATHS` points to summary **directories**; in the second, to **fold_metrics.csv files**. Changing only the initial cell is insufficient. |
| Discussion: initial configuration cell | Update both the search for `output/dissertation_analysis` and the assigned `ANALYSIS_ROOT`. |
| Methodology: initial configuration cell | Update `DATA_ROOT` and `OUTPUT_ROOT`. |
| Methodology: compound-level 3D-view cell | Check `MGT_ROOT`, `EXPERIMENT_ROOT`, `SOURCE_DATASET_ROOT` and `CURATED_PATH`. This cell resolves paths separately. |

Keep the analysis output under the repository root because artifact paths are
recorded relative to it. A different dataset, seed or fold count also requires
reviewing the notebooks' fixed filenames/counts; changing the output directory
alone does not generalize them.

The compound-level view saves `figure_2_7_compound_level_3d_model_view.png`
and `.pdf`. Its default is `output/dissertation_analysis/figures/`, separate
from the other methodology figures. To put it with them, set
`FIGURE_ROOT = OUTPUT_ROOT` before that cell in your working copy. This figure
is not added to the main methodology artifact list.

## Project structure and file guide

```text
your-workspace/
├── EV-A71_2A_benchmark/                 official affinity reference
└── MGT/
    ├── README.md
    ├── train_openbind_*.py             individual training entry points
    ├── run_openbind_cv.py              all-model CV orchestration
    ├── run_scaffold_cv.py              scaffold-default wrapper
    ├── model/                         neural-network architectures
    ├── modules/                       shared neural-network layers
    ├── utils/                         data, features, graphs, splits and masks
    ├── notebooks/                     analysis and figure generation
    ├── examples/example_data/atom_init.json
    ├── OpenBind_EV-A71_2A/
    │   ├── OpenBind_EV-A71_2A/         extracted metadata and structures
    │   └── experiment_a_ligand_mgt/   generated training dataset
    │       ├── atom_init.json
    │       ├── id_prop.csv
    │       ├── raw/
    │       ├── processed/
    │       ├── curated/
    │       ├── reports/
    │       ├── splits/
    │       ├── cv_random/
    │       └── cv/
    └── output/                        checkpoints, predictions and figures
```

### What each active Python file does

| File | Responsibility and connection to the workflow |
|---|---|
| [run_openbind_cv.py](run_openbind_cv.py) | Creates/loads the requested CV manifests, launches individual trainers and aggregates outer-test compound predictions. |
| [run_scaffold_cv.py](run_scaffold_cv.py) | Convenience entry point for the same workflow with scaffold CV as the default. |
| [train_openbind_morgan_mlp.py](train_openbind_morgan_mlp.py) | Builds Morgan fingerprints and trains/evaluates the fingerprint MLP. |
| [train_openbind_2d_gnn.py](train_openbind_2d_gnn.py) | Trains/evaluates the atom–bond graph model using the shared ligand dataset. |
| [train_openbind_3d_gnn.py](train_openbind_3d_gnn.py) | Trains/evaluates the crystallographic distance model. |
| [train_openbind_3d_alignn.py](train_openbind_3d_alignn.py) | Trains/evaluates the distance-and-angle model; also supplies the supervised stage used by masked ALIGNN. |
| [train_openbind_mgt.py](train_openbind_mgt.py) | Loads cached graph triplets and trains/evaluates Graphformer; also supplies the supervised stage used by masked MGT. |
| [train_openbind_masked.py](train_openbind_masked.py) | Applies training-partition masking, trains a reconstruction decoder, then invokes ALIGNN or MGT affinity fine-tuning/evaluation. |
| [utils/audit_openbind_structure_files.py](utils/audit_openbind_structure_files.py) | Resolves raw structure-file locations and writes the audit manifest/checksums. |
| [utils/prepare_openbind_ligand_mgt.py](utils/prepare_openbind_ligand_mgt.py) | Filters records, checks official labels, exports ligand PDBs and creates curated tables/classic splits. |
| [utils/preprocess_openbind_mgt_graphs.py](utils/preprocess_openbind_mgt_graphs.py) | Builds and saves the MGT three-graph cache with provenance reports. |
| [utils/generate_random_cv.py](utils/generate_random_cv.py) | Creates compound-grouped random outer folds and per-fold train/validation/test manifests. |
| [utils/generate_scaffold_cv.py](utils/generate_scaffold_cv.py) | Creates scaffold-disjoint outer folds and their manifests; CV uses achiral scaffold grouping. |
| [utils/molecular_features.py](utils/molecular_features.py) | Defines RDKit atom/bond feature vocabularies and encoding functions. |
| [utils/openbind_ligand_dataset.py](utils/openbind_ligand_dataset.py) | Loads reference SDFs and constructs the controlled 2D/3D graphs used by the GNN trainers. |
| [utils/datasets.py](utils/datasets.py) | Supplies MGT structure loading, graph construction/batching, positional encoding and angle-cosine calculations. |
| [utils/masker.py](utils/masker.py) | Implements the original `MaskAtom` transform reused for atom-feature masking. |
| [model/ligand_gnn.py](model/ligand_gnn.py) | Defines the 2D ligand graph network. |
| [model/ligand_3d_gnn.py](model/ligand_3d_gnn.py) | Defines the distance-based ligand graph network. |
| [model/ligand_3d_alignn.py](model/ligand_3d_alignn.py) | Defines the ligand ALIGNN encoder using the shared angular layers. |
| [model/alignn.py](model/alignn.py) | Implements `ALIGNNLayer` and `EdgeGatedGraphConv`, reused by the graph models. |
| [model/transformer.py](model/transformer.py) | Implements wider-graph multi-head attention used by Graphformer. |
| [model/graphformer.py](model/graphformer.py) | Combines MGT attention, angular/local updates, feed-forward processing, pooling and prediction. |
| [modules/modules.py](modules/modules.py) | Supplies shared MLP and radial-basis expansion layers. |
| `model/__init__.py`, `modules/__init__.py`, `utils/__init__.py` | Package initialization and imports; retain these with their directories. |

In short, preparation creates the data/manifests; the CV runner passes each
manifest to a trainer; the trainer loads its dataset and model implementation;
the notebooks read the saved predictions and summaries. The `model/`,
`modules/` and `utils/` files are imported by the entry points, not run
individually in sequence.

## Legacy files and practical notes

These files are retained for reference or optional upstream reproduction.
They are **not required steps** for the main commands above.

| File/resource | Status |
|---|---|
| [pre-training.py](pre-training.py) | Original Fabric pretraining workflow; not the matched training-fold masking experiment. |
| [training.py](training.py), [testing.py](testing.py), [run.py](run.py) | Original supervised training, testing and inference entry points. They use different settings/checkpoint conventions; do not substitute them for the matched trainers. |
| [utils/prepare_openbind_original_entrypoints.py](utils/prepare_openbind_original_entrypoints.py) | Prepares separate data roots for the original entry points only. |
| [utils/make_atom_init.py](utils/make_atom_init.py), `utils/atomic_properties.xlsx` | Optional elemental-feature generation utility and source workbook. Not needed when using the verified `atom_init.json`. |
| `__pycache__/`, `*.pyc` | Automatically generated Python caches; not source or data dependencies. |


### Sources

The implementation builds on the
[original Molecular Graph Transformer](https://github.com/MolecularGraphTransformer/MGT).
Raw structures come from the
[OpenBind data release](https://doi.org/10.5281/zenodo.20026661);
the official labels/reference are in the
[EV-A71 2A benchmark repository](https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark).
Use those projects' acknowledgements and licence terms when redistributing
their code or data.

## AI acknowledgement

OpenAI Codex was used to assist with debugging and code annotation. All
AI-assisted outputs were reviewed and verified by the author before being used
in this project.
