# CycleVI

CycleVI is a deep generative model that isolates cell cycle variation in single-cell RNA-seq data. It learns a disentangled latent space where two dimensions capture circular cell cycle position (z_cycle) and the remaining dimensions capture everything else (z_other) — so downstream analyses are not confounded by cell cycle.

> **Preprint:** [CycleVI: Isolating cell cycle variation with an interpretable deep generative model](https://doi.org/10.1101/2025.11.04.686009)

---

## Installation

```bash
pip install cyclevi
```

To run without installing (from the repository root):

```bash
python -m cyclevi
```

**Dependencies:** PyTorch, scvi-tools, anndata, scanpy, scikit-learn, click, numpy, pandas.

---

## Quickstart

One command runs the full pipeline — load data, compute phase initialization, train the model, save outputs:

```bash
cyclevi run --input data.h5ad --output results/
```

Outputs written to `results/`:

| File | Contents |
|---|---|
| `model/` | Saved model (reload with `CycleVI.load`) |
| `latent_cycle.csv` | 2D circular cell cycle coordinates (`z_cycle_x`, `z_cycle_y`) |
| `latent_other.csv` | Non-cycling latent dimensions (`z_1` … `z_N`) |
| `cycle_angles.csv` | Inferred cell cycle angle per cell (radians, −π to π) |

---

## Step-by-step workflow

For more control, or to reuse the prepared file across multiple runs:

```bash
# Step 1: compute phase initialization, save a prepared file
cyclevi prepare --input data.h5ad --output prepared.h5ad

# Step 2: train (layer and column names are read from the file automatically)
cyclevi train --input prepared.h5ad --output results/
```

The prepared file stores all settings internally, so `cyclevi train` needs no extra flags.

---

## Supported input formats

All commands that accept `--input` support the following formats:

| Format | Description |
|---|---|
| `.h5ad` | AnnData HDF5 file |
| `.h5` | 10x Genomics HDF5 file (Cell Ranger output) |
| `.loom` | Loom file (e.g. from velocyto) |
| `.mtx` | MatrixMarket sparse matrix (cells as rows by default) |
| directory | 10x Genomics MTX folder (`matrix.mtx`, `barcodes.tsv`, `features.tsv`) or Single Cell Expression Atlas folder (`*.mtx`, `*.mtx_cols`, `*.mtx_rows`) — detected automatically |
| `.csv` / `.tsv` | Delimited text (cells as rows, genes as columns by default) |

For `.mtx` and `.csv`/`.tsv` files where genes are rows, add `--transpose`.

---

## Commands

### `cyclevi run`

End-to-end pipeline in one step.

```bash
cyclevi run --input data.h5ad --output results/
```

Accepts all options from `prepare` and `train` combined (see below).

---

### `cyclevi prepare`

Compute phase initialization and save a prepared `.h5ad` file.

CycleVI infers cell cycle position itself — this step provides the initial phase guesses the model needs to start training. It scores cells using known S and G2/M marker genes, derives a continuous angle from the scores, and quantile-transforms it so cells are spread uniformly around the circle.

```bash
cyclevi prepare --input data.h5ad --output prepared.h5ad
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--input` | — | Input file or directory |
| `--output` | — | Path to save the prepared `.h5ad` file |
| `--gene-id-type` | `auto` | Gene identifier type: `auto` detects from `var_names`, or set `ensembl` / `symbol` explicitly. Ignored when `--s-genes` and `--g2m-genes` are provided. |
| `--s-genes` | — | Text file with S-phase marker genes, one per line. Overrides the bundled gene list (must be paired with `--g2m-genes`) |
| `--g2m-genes` | — | Text file with G2/M-phase marker genes, one per line. Must be provided together with `--s-genes` |
| `--var-names` | `gene_symbols` | For 10x MTX directories: use `gene_symbols` or `gene_ids` as `var_names` |
| `--transpose` | off | Transpose matrix after loading (for MTX/CSV with genes as rows) |
| `--counts-layer` | `counts` | Layer name for storing a copy of raw counts |
| `--phase-key` | `phase` | `adata.obs` key for discrete phase labels (G1 / S / G2M) |
| `--angle-key` | `cycle_angle` | `adata.obs` key for the raw arctan2 angle |
| `--uniform-angle-key` | `cycle_angle_uniform` | `adata.obs` key for the quantile-transformed angle (used for training) |

By default, CycleVI uses the bundled Regev-lab marker genes (Ensembl IDs or gene symbols, detected automatically). To use your own lists — for example if your data comes from a non-human organism — provide plain-text files with one gene name per line:

```bash
cyclevi prepare --input data.h5ad --output prepared.h5ad \
  --s-genes my_s_genes.txt \
  --g2m-genes my_g2m_genes.txt
```

Both flags must be given together; gene names must match your dataset's `var_names`.

**Columns added to `adata.obs`:**

| Column | Description |
|---|---|
| `phase` | Initial phase label: G1, S, or G2M |
| `S_score` | Continuous S-phase score |
| `G2M_score` | Continuous G2/M-phase score |
| `cycle_angle` | Raw angle: `arctan2(G2M_score, S_score)` in [0, 2π] |
| `cycle_angle_uniform` | Quantile-transformed angle, uniformly distributed in [0, 2π] |

The prepared file stores these column names in `adata.uns["cyclevi"]` so that `cyclevi train` picks them up automatically.

---

### `cyclevi train`

Train a CycleVI model from a prepared `.h5ad` file.

```bash
cyclevi train --input prepared.h5ad --output results/
```

If the file was created by `cyclevi prepare`, no additional flags are needed — the layer and column names are read from the file automatically.

**Options:**

| Option | Default | Description |
|---|---|---|
| `--input` | — | Prepared `.h5ad` file |
| `--output` | — | Output directory |
| `--batch-key` | `None` | `adata.obs` column for experimental batch |
| `--labels-key` | `None` | `adata.obs` column for cell type labels |
| `--cycle-label-key` | auto | `adata.obs` column for phase labels — read from file if prepared with `cyclevi prepare` |
| `--cycle-angle-key` | auto | `adata.obs` column for cycle angle — read from file if prepared with `cyclevi prepare` |
| `--layer` | auto | AnnData layer with raw counts — read from file if prepared with `cyclevi prepare` |
| `--n-latent` | `10` | Total latent dimensions (first 2 are always z_cycle) |
| `--n-hidden` | `128` | Hidden units per encoder/decoder layer |
| `--n-layers` | `1` | Number of encoder/decoder layers |
| `--n-epochs` | `400` | Training epochs |
| `--batch-size` | `128` | Mini-batch size |
| `--lr` | `1e-3` | Learning rate |

**Outputs saved to `--output`:**

| File | Contents |
|---|---|
| `model/` | Saved model (reload with `CycleVI.load`) |
| `latent_cycle.csv` | 2D circular cell cycle coordinates (`z_cycle_x`, `z_cycle_y`) |
| `latent_other.csv` | Non-cycling latent dimensions (`z_1` … `z_N`) |
| `cycle_angles.csv` | Inferred cell cycle angle per cell (radians, −π to π) |

---

### `cyclevi extract`

Extract latent representations from a previously saved model without retraining.

```bash
cyclevi extract \
  --input data.h5ad \
  --model results/model \
  --output results/
```

Writes the same three CSV files as `train`.

| Option | Description |
|---|---|
| `--input` | Input `.h5ad` file |
| `--model` | Directory of a saved CycleVI model |
| `--output` | Directory to write output CSVs |

---

## Python API

```python
import anndata as ad
from cyclevi import CycleVI
from cyclevi.prepare import load_adata, compute_phase_initialization

# Load and prepare data
adata = load_adata("data.h5ad")
compute_phase_initialization(adata)
# To use custom marker genes instead of the bundled lists:
# compute_phase_initialization(adata, s_genes_file="s.txt", g2m_genes_file="g2m.txt")

# Train
CycleVI.setup_anndata(
    adata,
    layer="counts",
    cycle_initiation_label_key="phase",
    cycle_initiation_angle_key="cycle_angle_uniform",
)
model = CycleVI(adata)
model.train(max_epochs=400)

# Extract latent representations
import numpy as np
z = model.get_latent_representation(adata)
z_cycle   = z[:, :2]          # circular cell cycle coordinates
z_other   = z[:, 2:]          # non-cycling dimensions
angles    = np.arctan2(z[:, 1], z[:, 0])  # cell cycle angle
```

For a full walkthrough, see [`Tutorial.ipynb`](Tutorial.ipynb) or [`Tutorial_colab.ipynb`](Tutorial_colab.ipynb) (Google Colab ready).

---

## Feedback

For questions and comments, contact [Gustavo S. Jeuken](mailto:g.stolfjeuken@vu.nl) or open an issue on [GitHub](https://github.com/jeuken/CycleVI/issues).

## License

BSD 3-Clause License

## Citation

If you use CycleVI in a publication, please cite:

> CycleVI: Isolating cell cycle variation with an interpretable deep generative model
>
> Pia Mozdzanowski, Marcel Tarbier, Gustavo S. Jeuken
>
> bioRxiv 2025.11.04.686009; doi: https://doi.org/10.1101/2025.11.04.686009
