# CycleVI

CycleVI is a deep generative model that isolates cell cycle variation in single-cell RNA-seq data. It learns a disentangled latent space where two dimensions capture the cell's **circular position in the cell cycle** (`z_cycle`) and the remaining dimensions capture **everything else** (`z_other`) — so your downstream analyses (clustering, UMAP, integration) are no longer confounded by the cell cycle.

In practice, CycleVI gives you two things for every cell:

- a **cell cycle angle** — a continuous coordinate describing where the cell sits in the cycle, and
- a **cell cycle-free embedding** — a representation you can use in place of PCA for everything downstream.

> **Publication:** [CycleVI: Isolating cell cycle variation with an interpretable deep generative model](https://doi.org/10.1093/bioinformatics/btag372)

---

## Installation

```bash
pip install cyclevi
```

This installs the `cyclevi` command-line tool and the Python package. To run from a clone of the repository without installing:

```bash
python -m cyclevi
```

**Dependencies** (installed automatically): PyTorch, scvi-tools, anndata, scanpy, scikit-learn, scipy, click, numpy, pandas. A GPU is optional but strongly recommended — see [Runtime](#runtime) below.

---

## What your input data needs

Before running anything, make sure your data meets these expectations:

- **Raw counts**, not normalized or log-transformed values. CycleVI models counts directly; the matrix should contain integer-like UMI/read counts.
- **Cells as rows, genes as columns** (the standard AnnData orientation). If your matrix is the other way round, add `--transpose`.
- **Gene names as `var_names`** — either gene symbols (e.g. `MKI67`) or Ensembl IDs (e.g. `ENSG00000148773`) work automatically, and CycleVI detects which one you and matches the correct built-in cell cycle marker list. Human data works out of the box; for other organisms, or other gene ID schemes, supply your own marker lists (see [`prepare`](#cyclevi-prepare)).

Standard quality-control filtering (removing low-count cells and rarely-expressed genes) before running CycleVI is recommended, just as for any scRNA-seq workflow.

See [Supported input formats](#supported-input-formats) for the file types you can load.

---

## Quickstart

One command runs the full pipeline — load data, compute the cell cycle initialization, train the model, and save outputs:

```bash
cyclevi run --input data.h5ad --output results/
```

That's all most users need. The four files written to `results/` are described next.

---

## Understanding the output

Every command that produces results writes these files to the output directory:

| File | What it contains | How to use it |
|---|---|---|
| `cycle_angles.csv` | One angle per cell, in radians (−π to π). The cell's inferred position on the cell cycle. | This is the **main result**. Treat it like a cell cycle "pseudotime" — sort cells by it, color a UMAP by it, or test genes for cycle-dependent expression. |
| `latent_other.csv` | A per-cell embedding (`z_1` … `z_N`) with the cell cycle **removed**. | Use this **in place of PCA** for clustering, UMAP, and integration, so the cell cycle no longer drives your results. |
| `latent_cycle.csv` | The 2D coordinates (`z_cycle_x`, `z_cycle_y`) that the angle is computed from. | Plot `z_cycle_y` against `z_cycle_x` to see cells arranged on a circle (the cell cycle); the angle in `cycle_angles.csv` is `arctan2(z_cycle_y, z_cycle_x)`. |
| `model/` | The trained model. | Reload later with `CycleVI.load` or `cyclevi extract` to get representations for new cells without retraining. |

Each CSV is indexed by cell barcode, so you can join it straight back onto your AnnData with `adata.obs`.

---

## Which command should I use?

- **Just want results?** Use [`cyclevi run`](#cyclevi-run) — it does everything in one step.
- **Want to inspect the cell cycle initialization, or reuse it across several training runs?** Split the work with [`cyclevi prepare`](#cyclevi-prepare) then [`cyclevi train`](#cyclevi-train).
- **Already have a trained model and want representations for more cells?** Use [`cyclevi extract`](#cyclevi-extract).

---

## Step-by-step workflow

For more control, or to reuse the prepared file across multiple training runs:

```bash
# Step 1: compute the cell cycle initialization, save a prepared file
cyclevi prepare --input data.h5ad --output prepared.h5ad

# Step 2: train (layer and column names are read from the file automatically)
cyclevi train --input prepared.h5ad --output results/
```

The prepared file records all the settings it needs internally, so `cyclevi train` requires no extra flags.

---

## Runtime

CycleVI trains a neural network and benefits greatly from a GPU. It automatically uses an NVIDIA (CUDA) or Apple Silicon (MPS) GPU if one is available, and prints which device it is using when training starts. On CPU, training still works but is considerably slower — consider lowering `--n-epochs` for a quick test run, or use [Google Colab](Tutorial_colab.ipynb) for free GPU access.

---

## Supported input formats

All commands that accept `--input` support the following:

| Format | Description |
|---|---|
| `.h5ad` | AnnData HDF5 file |
| `.h5` | 10x Genomics HDF5 file (Cell Ranger output) |
| `.loom` | Loom file (e.g. from velocyto) |
| `.mtx` | MatrixMarket sparse matrix (cells as rows by default) |
| directory | 10x Genomics MTX folder (`matrix.mtx`, `barcodes.tsv`, `features.tsv`) or Single Cell Expression Atlas folder (`*.mtx`, `*.mtx_cols`, `*.mtx_rows`) — detected automatically |
| `.csv` / `.tsv` | Delimited text (cells as rows, genes as columns by default) |

For `.mtx` and `.csv`/`.tsv` files where genes are rows instead of columns, add `--transpose`.

---

## Commands

### `cyclevi run`

End-to-end pipeline in one step — prepare, train, and save all outputs.

```bash
cyclevi run --input data.h5ad --output results/
```

Accepts every option from `prepare` and `train` combined (see below). Outputs are described in [Understanding the output](#understanding-the-output).

---

### `cyclevi prepare`

Compute the cell cycle initialization and save a prepared `.h5ad` file ready for training.

CycleVI infers cell cycle position itself during training — this step provides the **initial guesses** the model starts from. It scores cells against known S and G2/M marker genes, derives a continuous angle from those scores, and quantile-transforms it so cells are spread evenly around the circle.

```bash
cyclevi prepare --input data.h5ad --output prepared.h5ad
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--input` | — | Input file or directory |
| `--output` | — | Path to save the prepared `.h5ad` file |
| `--gene-id-type` | `auto` | Gene identifier type: `auto` detects from `var_names`, or set `ensembl` / `symbol` explicitly. Ignored when `--s-genes` and `--g2m-genes` are provided. |
| `--s-genes` | — | Text file with S-phase marker genes, one per line. Overrides the bundled gene list (must be paired with `--g2m-genes`). |
| `--g2m-genes` | — | Text file with G2/M-phase marker genes, one per line. Must be provided together with `--s-genes`. |
| `--var-names` | `gene_symbols` | For 10x MTX directories: use `gene_symbols` or `gene_ids` as `var_names`. |
| `--transpose` | off | Transpose matrix after loading (for MTX/CSV with genes as rows). |
| `--counts-layer` | `counts` | Layer name for storing a copy of the raw counts. |
| `--phase-key` | `phase` | `adata.obs` key for discrete phase labels (G1 / S / G2M). |
| `--angle-key` | `cycle_angle` | `adata.obs` key for the raw arctan2 angle. |
| `--uniform-angle-key` | `cycle_angle_uniform` | `adata.obs` key for the quantile-transformed angle (used for training). |

#### Custom marker genes (non-human data)

By default CycleVI uses the bundled Regev-lab human marker genes (Ensembl IDs or gene symbols, detected automatically). For a different organism, supply your own plain-text lists with one gene name per line:

```bash
cyclevi prepare --input data.h5ad --output prepared.h5ad \
  --s-genes my_s_genes.txt \
  --g2m-genes my_g2m_genes.txt
```

Both flags must be given together, and the gene names must match your dataset's `var_names`. `prepare` reports how many of your marker genes were found in the data — check this number is reasonable.

#### Columns added to `adata.obs`

| Column | Description |
|---|---|
| `phase` | Initial phase label: G1, S, or G2M |
| `S_score` | Continuous S-phase score |
| `G2M_score` | Continuous G2/M-phase score |
| `cycle_angle` | Raw angle: `arctan2(G2M_score, S_score)` in [0, 2π] |
| `cycle_angle_uniform` | Quantile-transformed angle, uniformly distributed in [0, 2π] |

These column names are stored in `adata.uns["cyclevi"]` so that `cyclevi train` picks them up automatically.

---

### `cyclevi train`

Train a CycleVI model from a prepared `.h5ad` file.

```bash
cyclevi train --input prepared.h5ad --output results/
```

If the file came from `cyclevi prepare`, no extra flags are needed — the counts layer and column names are read from the file automatically.

**Options:**

| Option | Default | Description |
|---|---|---|
| `--input` | — | Prepared `.h5ad` file |
| `--output` | — | Output directory |
| `--batch-key` | `None` | `adata.obs` column for experimental batch (enables batch correction). |
| `--labels-key` | `None` | `adata.obs` column for cell type labels. |
| `--cycle-label-key` | auto | `adata.obs` column for phase labels — read from the file if prepared with `cyclevi prepare`. |
| `--cycle-angle-key` | auto | `adata.obs` column for cycle angle — read from the file if prepared with `cyclevi prepare`. |
| `--layer` | auto | AnnData layer with raw counts — read from the file if prepared with `cyclevi prepare`. |
| `--n-latent` | `10` | Total latent dimensions (the first 2 are always `z_cycle`). |
| `--n-hidden` | `128` | Hidden units per encoder/decoder layer. |
| `--n-layers` | `1` | Number of encoder/decoder layers. |
| `--n-epochs` | `400` | Training epochs. |
| `--batch-size` | `128` | Mini-batch size. |
| `--lr` | `1e-3` | Learning rate. |

Outputs are described in [Understanding the output](#understanding-the-output).

---

### `cyclevi extract`

Extract latent representations from a previously saved model — useful for applying a trained model to new cells without retraining.

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

Prefer working in a notebook or script? The same pipeline is available directly:

```python
import numpy as np
from cyclevi import CycleVI
from cyclevi.prepare import load_adata, compute_phase_initialization

# 1. Load data (raw counts in .X) and compute the cell cycle initialization
adata = load_adata("data.h5ad")
compute_phase_initialization(adata)
# For a non-human organism, supply your own marker lists instead:
# compute_phase_initialization(adata, s_genes_file="s.txt", g2m_genes_file="g2m.txt")

# 2. Train
CycleVI.setup_anndata(
    adata,
    layer="counts",
    cycle_initiation_label_key="phase",
    cycle_initiation_angle_key="cycle_angle_uniform",
)
model = CycleVI(adata)
model.train(max_epochs=400)

# 3. Extract representations
z = model.get_latent_representation(adata)
z_cycle = z[:, :2]                        # 2D circular cell cycle coordinates
z_other = z[:, 2:]                        # cell cycle-free embedding
angles  = np.arctan2(z[:, 1], z[:, 0])    # cell cycle angle per cell
```

For a full walkthrough, see [`Tutorial.ipynb`](Tutorial.ipynb) or [`Tutorial_colab.ipynb`](Tutorial_colab.ipynb) (ready to run on Google Colab).

---

## Feedback

Questions or comments? Contact [Gustavo S. Jeuken](mailto:g.stolfjeuken@vu.nl) or open an issue on [GitHub](https://github.com/jeuken/CycleVI/issues).

## License

BSD 3-Clause License

## Citation

If you use CycleVI in a publication, please cite:

> CycleVI: Isolating cell cycle variation with an interpretable deep generative model
>
> Pia Mozdzanowski, Marcel Tarbier, Gustavo S. Jeuken
>
> Bioinformatics, Volume 42, Issue 6, June 2026, btag372, https://doi.org/10.1093/bioinformatics/btag372
</content>
</invoke>
