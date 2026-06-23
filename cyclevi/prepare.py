import importlib.resources
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

_ENSG_RE = re.compile(r"^ENS[A-Z]*G\d{11}$")


def detect_gene_id_type(var_names) -> str:
    """Return ``'ensembl'`` or ``'symbol'`` by inspecting ``var_names``.

    Checks up to the first 200 names: if more than half match the Ensembl
    gene ID pattern (``ENSG...`` / ``ENSMUSG...``), returns ``'ensembl'``.
    """
    sample = [str(v) for v in list(var_names)[:200]]
    n_ensg = sum(1 for v in sample if _ENSG_RE.match(v))
    return "ensembl" if n_ensg / len(sample) > 0.5 else "symbol"


def _is_scea_dir(p: Path) -> bool:
    """Return True if *p* looks like a Single Cell Expression Atlas MTX directory."""
    return bool(list(p.glob("*.mtx_cols")))


def _load_scea_mtx(directory: Path) -> AnnData:
    """Load a Single Cell Expression Atlas MTX directory.

    Expects ``*.mtx``, ``*.mtx_cols`` (cell IDs), and ``*.mtx_rows`` (gene
    info) files. The matrix is stored genes × cells and transposed on load.
    The first column of ``*.mtx_rows`` becomes ``var_names``; if a second
    column is present it is stored as ``adata.var["gene_name"]``.
    """
    import scipy.io

    mtx_files = list(directory.glob("*.mtx"))
    cols_files = list(directory.glob("*.mtx_cols"))
    rows_files = list(directory.glob("*.mtx_rows"))

    if not mtx_files:
        raise FileNotFoundError(f"No .mtx file found in {directory}")
    if not cols_files:
        raise FileNotFoundError(f"No .mtx_cols file found in {directory}")
    if not rows_files:
        raise FileNotFoundError(f"No .mtx_rows file found in {directory}")

    # SCEA stores genes × cells → transpose to cells × genes
    mat = scipy.io.mmread(mtx_files[0]).T.tocsr()

    cell_ids = pd.read_csv(cols_files[0], header=None, sep="\t")[0].tolist()
    rows_df = pd.read_csv(rows_files[0], header=None, sep="\t")
    gene_ids = rows_df[0].tolist()

    adata = AnnData(X=mat)
    adata.obs_names = cell_ids
    adata.var_names = gene_ids
    if rows_df.shape[1] > 1:
        adata.var["gene_name"] = rows_df[1].values

    return adata


def load_adata(
    path: str,
    var_names: str = "gene_symbols",
    transpose: bool = False,
) -> AnnData:
    """Load an AnnData object from a variety of file formats.

    Parameters
    ----------
    path:
        Path to one of:

        * ``*.h5ad`` — AnnData HDF5 file.
        * ``*.h5`` — 10x Genomics HDF5 file (Cell Ranger output).
        * ``*.loom`` — Loom file (e.g. from velocyto).
        * ``*.mtx`` — MatrixMarket sparse matrix. Cells are assumed to be rows
          unless *transpose* is True.
        * A **directory** — if it contains ``*.mtx_cols`` files, treated as a
          Single Cell Expression Atlas MTX folder; otherwise treated as a 10x
          Genomics MTX folder containing ``matrix.mtx[.gz]``,
          ``barcodes.tsv[.gz]``, and ``features.tsv[.gz]`` / ``genes.tsv[.gz]``.
        * ``*.csv`` or ``*.tsv`` — delimited text where rows are cells and
          columns are genes, unless *transpose* is True.

    var_names:
        Only used for 10x MTX directories. ``'gene_symbols'`` (default) sets
        ``adata.var_names`` to gene symbols; ``'gene_ids'`` uses Ensembl IDs.
    transpose:
        Transpose the matrix after loading. Useful when a MTX or CSV file has
        genes as rows and cells as columns.

    Returns
    -------
    AnnData
    """
    import scanpy as sc

    p = Path(path)

    if p.is_dir():
        if _is_scea_dir(p):
            adata = _load_scea_mtx(p)
        else:
            adata = sc.read_10x_mtx(path, var_names=var_names, cache=False)

    elif p.suffix == ".h5ad":
        adata = sc.read_h5ad(path)

    elif p.suffix == ".h5":
        adata = sc.read_10x_h5(path)

    elif p.suffix == ".loom":
        adata = sc.read_loom(path)

    elif p.suffix == ".mtx":
        adata = sc.read_mtx(path)
        if transpose:
            adata = adata.T

    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        adata = sc.read_csv(path, delimiter=sep)
        if transpose:
            adata = adata.T

    else:
        raise ValueError(
            f"Unrecognised format '{p.suffix}'. "
            "Supported: .h5ad, .h5, .loom, .mtx, .csv, .tsv, or a 10x MTX directory."
        )

    return adata


def _load_gene_lists(gene_id_type: str) -> tuple[list, list]:
    """Return (s_genes, g2m_genes) from the bundled gene lists."""
    pkg = importlib.resources.files("cyclevi.data")

    if gene_id_type == "ensembl":
        df = pd.read_csv(StringIO((pkg / "homo_sapiens_cc_genes.csv").read_text()), index_col=0)
        s_genes = df.loc["S", "geneID"].tolist()
        g2m_genes = df.loc["G2/M", "geneID"].tolist()

    elif gene_id_type == "symbol":
        lines = (pkg / "regev_lab_cell_cycle_genes.txt").read_text().splitlines()
        genes = [l.strip() for l in lines if l.strip()]
        s_genes = genes[:43]
        g2m_genes = genes[43:]

    else:
        raise ValueError(f"gene_id_type must be 'ensembl' or 'symbol', got '{gene_id_type}'")

    return s_genes, g2m_genes


def _read_gene_file(path: str) -> list[str]:
    """Read a one-gene-per-line text file and return a list of gene names."""
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip()]


def compute_phase_initialization(
    adata: AnnData,
    gene_id_type: str = "auto",
    counts_layer: str = "counts",
    phase_key: str = "phase",
    angle_key: str = "cycle_angle",
    uniform_angle_key: str = "cycle_angle_uniform",
    s_genes_file: str | None = None,
    g2m_genes_file: str | None = None,
) -> AnnData:
    """Compute initial phase labels and angles used to initialize CycleVI training.

    CycleVI infers cell cycle position from the data. This function provides
    the initial guesses it needs to start training:

    1. Save raw counts to ``adata.layers[counts_layer]``.
    2. Compute S and G2/M phase scores with ``scanpy.tl.score_genes_cell_cycle``.
    3. Derive a continuous angle as ``arctan2(G2M_score, S_score)`` wrapped to [0, 2π].
    4. Quantile-transform the angle so cells are uniformly spread around the circle.
    5. Store the column names in ``adata.uns["cyclevi"]`` so that downstream
       tools can pick them up automatically.

    Parameters
    ----------
    adata:
        AnnData with raw counts in ``.X``.
    gene_id_type:
        ``'auto'`` (default) detects the type from ``var_names``.
        ``'ensembl'`` matches Ensembl IDs; ``'symbol'`` matches gene symbols.
        Ignored when *s_genes_file* and *g2m_genes_file* are both provided.
    counts_layer:
        Layer name in which to store a copy of ``adata.X`` before normalization.
    phase_key:
        ``adata.obs`` key for the discrete phase labels (G1 / S / G2M).
    angle_key:
        ``adata.obs`` key for the raw arctan2 angle in [0, 2π].
    uniform_angle_key:
        ``adata.obs`` key for the quantile-transformed uniform angle in [0, 2π].
        Pass this as ``cycle_initiation_angle_key`` to ``CycleVI.setup_anndata``.
    s_genes_file:
        Path to a plain-text file with S-phase marker genes, one per line.
        When provided together with *g2m_genes_file*, the bundled gene lists
        are not used and *gene_id_type* is ignored.
    g2m_genes_file:
        Path to a plain-text file with G2/M-phase marker genes, one per line.
        Must be provided together with *s_genes_file*.

    Returns
    -------
    AnnData
        The input object modified in place (also returned for convenience).
    """
    import scanpy as sc
    from sklearn.preprocessing import QuantileTransformer

    if (s_genes_file is None) != (g2m_genes_file is None):
        raise ValueError("Provide both s_genes_file and g2m_genes_file, or neither.")

    if s_genes_file is not None:
        s_genes = _read_gene_file(s_genes_file)
        g2m_genes = _read_gene_file(g2m_genes_file)
        gene_id_type = "custom"
    else:
        if gene_id_type == "auto":
            gene_id_type = detect_gene_id_type(adata.var_names)
        s_genes, g2m_genes = _load_gene_lists(gene_id_type)

    # 1. Save raw counts
    adata.layers[counts_layer] = adata.X.copy()

    # 2. Compute initial phase scores used as training initialization
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

    # Rename scanpy's default 'phase' column if a different key was requested
    if phase_key != "phase" and "phase" in adata.obs.columns:
        adata.obs[phase_key] = adata.obs.pop("phase")

    # 4. Continuous angle from scores: arctan2(G2M_score, S_score) → [0, 2π]
    raw_angle = np.arctan2(adata.obs["G2M_score"].values, adata.obs["S_score"].values)
    adata.obs[angle_key] = np.mod(raw_angle, 2 * np.pi)

    # 5. Quantile-transform to spread cells uniformly around the circle
    angles = adata.obs[angle_key].values.reshape(-1, 1)
    qt = QuantileTransformer(
        output_distribution="uniform",
        random_state=0,
        n_quantiles=min(len(angles), 1000),
        subsample=len(angles),
    )
    adata.obs[uniform_angle_key] = qt.fit_transform(angles).flatten() * 2 * np.pi

    # 6. Store config so downstream commands can read keys automatically
    var_set = set(adata.var_names)
    n_s = sum(1 for g in s_genes if g in var_set)
    n_g2m = sum(1 for g in g2m_genes if g in var_set)
    adata.uns["cyclevi"] = {
        "layer": counts_layer,
        "cycle_label_key": phase_key,
        "cycle_angle_key": uniform_angle_key,
        "gene_id_type": gene_id_type,
        "n_s_genes_matched": n_s,
        "n_g2m_genes_matched": n_g2m,
        "n_s_genes_total": len(s_genes),
        "n_g2m_genes_total": len(g2m_genes),
    }

    return adata
