import os

import click
import numpy as np
import pandas as pd
import torch


# ── Console helpers ───────────────────────────────────────────────────────────

def _section(title: str):
    bar = "─" * (len(title) + 4)
    click.echo(click.style(f"\n{bar}", fg="cyan"))
    click.echo(click.style(f"  {title}", fg="cyan", bold=True))
    click.echo(click.style(bar, fg="cyan"))


def _ok(msg: str):
    click.echo(click.style("  ✓  ", fg="green", bold=True) + msg)


def _info(msg: str):
    click.echo("     " + msg)


def _warn(msg: str):
    click.echo(click.style("  !  ", fg="yellow", bold=True) + msg)


# ── Shared logic ──────────────────────────────────────────────────────────────

def _save_latent_outputs(model, adata, output_dir):
    z = model.get_latent_representation(adata)
    idx = adata.obs_names

    pd.DataFrame(
        z[:, :2], index=idx, columns=["z_cycle_x", "z_cycle_y"]
    ).to_csv(os.path.join(output_dir, "latent_cycle.csv"))

    n_other = z.shape[1] - 2
    pd.DataFrame(
        z[:, 2:], index=idx, columns=[f"z_{i + 1}" for i in range(n_other)]
    ).to_csv(os.path.join(output_dir, "latent_other.csv"))

    pd.DataFrame(
        np.arctan2(z[:, 1], z[:, 0]), index=idx, columns=["cycle_angle"]
    ).to_csv(os.path.join(output_dir, "cycle_angles.csv"))

    _ok("latent_cycle.csv  latent_other.csv  cycle_angles.csv")


def _run_prepare(adata_obj, gene_id_type, counts_layer,
                 phase_key, angle_key, uniform_angle_key,
                 s_genes_file=None, g2m_genes_file=None):
    from cyclevi.prepare import compute_phase_initialization, detect_gene_id_type

    if s_genes_file is not None:
        _info(f"S genes file:   {s_genes_file}")
        _info(f"G2M genes file: {g2m_genes_file}")
    elif gene_id_type == "auto":
        detected = detect_gene_id_type(adata_obj.var_names)
        _info(f"Gene ID type detected: {detected}")
        gene_id_type = detected

    compute_phase_initialization(
        adata_obj,
        gene_id_type=gene_id_type,
        counts_layer=counts_layer,
        phase_key=phase_key,
        angle_key=angle_key,
        uniform_angle_key=uniform_angle_key,
        s_genes_file=s_genes_file,
        g2m_genes_file=g2m_genes_file,
    )

    cfg = adata_obj.uns["cyclevi"]
    _ok(
        f"S genes matched: {cfg['n_s_genes_matched']}/{cfg['n_s_genes_total']}  "
        f"G2M genes matched: {cfg['n_g2m_genes_matched']}/{cfg['n_g2m_genes_total']}"
    )

    dist = adata_obj.obs[phase_key].value_counts()
    for phase, n in dist.items():
        _info(f"{phase:<4}  {n:>7,} cells  ({100 * n / len(adata_obj):4.1f} %)")


def _run_train(adata_obj, batch_key, labels_key, layer, cycle_label_key,
               cycle_angle_key, n_latent, n_hidden, n_layers, n_epochs,
               batch_size, lr, output):
    from cyclevi.model import CycleVI

    # Fill unset flags from adata.uns["cyclevi"] written by prepare
    uns = adata_obj.uns.get("cyclevi", {})
    layer            = layer            or uns.get("layer")
    cycle_label_key  = cycle_label_key  or uns.get("cycle_label_key")
    cycle_angle_key  = cycle_angle_key  or uns.get("cycle_angle_key")

    # Validate that the required columns and layer actually exist
    errors = []
    if cycle_label_key is None:
        errors.append(
            "--cycle-label-key is required. "
            "Either pass it explicitly or run 'cyclevi prepare' first."
        )
    elif cycle_label_key not in adata_obj.obs:
        errors.append(
            f"--cycle-label-key '{cycle_label_key}' not found in adata.obs.\n"
            f"     Available columns: {', '.join(adata_obj.obs.columns.tolist())}"
        )
    if cycle_angle_key is None:
        errors.append(
            "--cycle-angle-key is required. "
            "Either pass it explicitly or run 'cyclevi prepare' first."
        )
    elif cycle_angle_key not in adata_obj.obs:
        errors.append(
            f"--cycle-angle-key '{cycle_angle_key}' not found in adata.obs.\n"
            f"     Available columns: {', '.join(adata_obj.obs.columns.tolist())}"
        )
    if layer and layer not in adata_obj.layers:
        errors.append(
            f"--layer '{layer}' not found in adata.layers.\n"
            f"     Available layers: {', '.join(adata_obj.layers.keys()) or 'none'}"
        )
    if errors:
        raise click.UsageError("\n  ".join(errors))

    _info(
        f"layer={layer!r}   "
        f"cycle_label_key={cycle_label_key!r}   "
        f"cycle_angle_key={cycle_angle_key!r}"
    )

    CycleVI.setup_anndata(
        adata_obj,
        layer=layer,
        batch_key=batch_key,
        labels_key=labels_key,
        cycle_initiation_label_key=cycle_label_key,
        cycle_initiation_angle_key=cycle_angle_key,
    )

    model = CycleVI(adata_obj, n_latent=n_latent, n_hidden=n_hidden, n_layers=n_layers)
    _info(str(model))

    if torch.cuda.is_available():
        _info(f"device: GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        _info("device: GPU (MPS)")
    else:
        _info("device: CPU  (no GPU detected — training may be slow)")

    model.train(max_epochs=n_epochs, batch_size=batch_size, plan_kwargs={"lr": lr})
    _ok("Training complete.")

    os.makedirs(output, exist_ok=True)
    model_dir = os.path.join(output, "model")
    model.save(model_dir, overwrite=True)
    _ok(f"Model saved → {model_dir}")

    _save_latent_outputs(model, adata_obj, output)


# ── Shared option lists ───────────────────────────────────────────────────────

def _add_options(options):
    """Apply a list of click.option decorators to a command function."""
    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f
    return decorator


_INPUT_OPTIONS = [
    click.option("--var-names", default="gene_symbols", show_default=True,
                 type=click.Choice(["gene_symbols", "gene_ids"]),
                 help="For 10x MTX dirs: which field to use as var_names."),
    click.option("--transpose", is_flag=True, default=False,
                 help="Transpose matrix after loading "
                      "(use when genes are rows in MTX/CSV)."),
]

_PREPARE_OPTIONS = [
    click.option("--gene-id-type", default="auto", show_default=True,
                 type=click.Choice(["auto", "ensembl", "symbol"]),
                 help="Gene identifier type in var_names. "
                      "'auto' detects from the data. "
                      "Ignored when --s-genes and --g2m-genes are provided."),
    click.option("--s-genes", "s_genes_file", default=None,
                 type=click.Path(exists=True, dir_okay=False),
                 help="Text file with S-phase marker genes, one per line. "
                      "Overrides the bundled gene list."),
    click.option("--g2m-genes", "g2m_genes_file", default=None,
                 type=click.Path(exists=True, dir_okay=False),
                 help="Text file with G2/M-phase marker genes, one per line. "
                      "Must be provided together with --s-genes."),
]

_TRAIN_OPTIONS = [
    click.option("--batch-key", default=None,
                 help="adata.obs column for experimental batch."),
    click.option("--labels-key", default=None,
                 help="adata.obs column for cell type labels."),
    click.option("--cycle-label-key", default=None,
                 help="adata.obs column for discrete phase labels (G1/S/G2M). "
                      "Read automatically from a prepared file if not set."),
    click.option("--cycle-angle-key", default=None,
                 help="adata.obs column for the cell cycle angle. "
                      "Read automatically from a prepared file if not set."),
    click.option("--layer", default=None,
                 help="AnnData layer with raw counts. "
                      "Read automatically from a prepared file if not set."),
    click.option("--n-latent", default=10, show_default=True, type=int,
                 help="Total latent dimensions (first 2 are always z_cycle)."),
    click.option("--n-hidden", default=128, show_default=True, type=int,
                 help="Hidden units per encoder/decoder layer."),
    click.option("--n-layers", default=1, show_default=True, type=int,
                 help="Number of encoder/decoder layers."),
    click.option("--n-epochs", default=400, show_default=True, type=int,
                 help="Training epochs."),
    click.option("--batch-size", default=128, show_default=True, type=int,
                 help="Mini-batch size."),
    click.option("--lr", default=1e-3, show_default=True, type=float,
                 help="Learning rate."),
]


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """CycleVI: cell cycle-aware VAE for scRNA-seq data.

    \b
    Quickstart (one command):
      cyclevi run --input data.h5ad --output results/

    \b
    Step by step:
      cyclevi prepare --input data.h5ad --output prepared.h5ad
      cyclevi train   --input prepared.h5ad --output results/
    """
    pass


# ── run ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Input data: .h5ad, .h5, .loom, .mtx, .csv/.tsv, or 10x MTX dir.")
@click.option("--output", required=True, type=click.Path(),
              help="Output directory.")
@_add_options(_PREPARE_OPTIONS)
@_add_options(_INPUT_OPTIONS)
@_add_options(_TRAIN_OPTIONS)
def run(input_path, output, gene_id_type, s_genes_file, g2m_genes_file,
        var_names, transpose,
        batch_key, labels_key, cycle_label_key, cycle_angle_key, layer,
        n_latent, n_hidden, n_layers, n_epochs, batch_size, lr):
    """Prepare data, train a model, and save all outputs in one step.

    \b
    Writes to OUTPUT:
      model/             saved model
      latent_cycle.csv   2D cell cycle coordinates (z_cycle_x, z_cycle_y)
      latent_other.csv   non-cycling latent dimensions (z_1 … z_N)
      cycle_angles.csv   inferred cell cycle angle per cell
    """
    from cyclevi.prepare import load_adata

    if (s_genes_file is None) != (g2m_genes_file is None):
        raise click.UsageError("--s-genes and --g2m-genes must be provided together.")

    _section("Load")
    _info(input_path)
    adata_obj = load_adata(input_path, var_names=var_names, transpose=transpose)
    _ok(f"{adata_obj.n_obs:,} cells  ×  {adata_obj.n_vars:,} genes")

    _section("Phase initialization")
    _run_prepare(adata_obj, gene_id_type,
                 "counts", "phase", "cycle_angle", "cycle_angle_uniform",
                 s_genes_file=s_genes_file, g2m_genes_file=g2m_genes_file)

    _section("Training")
    _run_train(adata_obj, batch_key, labels_key, layer, cycle_label_key, cycle_angle_key,
               n_latent, n_hidden, n_layers, n_epochs, batch_size, lr, output)

    _section("Done")
    _ok(f"All outputs saved to {output}/")


# ── prepare ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Input data: .h5ad, .h5, .loom, .mtx, .csv/.tsv, or 10x MTX dir.")
@click.option("--output", required=True, type=click.Path(),
              help="Path to save the prepared .h5ad file.")
@_add_options(_PREPARE_OPTIONS)
@_add_options(_INPUT_OPTIONS)
@click.option("--counts-layer", default="counts", show_default=True,
              help="Layer name for storing raw counts.")
@click.option("--phase-key", default="phase", show_default=True,
              help="adata.obs key for discrete phase labels (G1/S/G2M).")
@click.option("--angle-key", default="cycle_angle", show_default=True,
              help="adata.obs key for the raw arctan2 angle.")
@click.option("--uniform-angle-key", default="cycle_angle_uniform", show_default=True,
              help="adata.obs key for the quantile-transformed angle (used for training).")
def prepare(input_path, output, gene_id_type, s_genes_file, g2m_genes_file,
            var_names, transpose,
            counts_layer, phase_key, angle_key, uniform_angle_key):
    """Compute phase initialization and save a prepared .h5ad file ready for training.

    CycleVI infers cell cycle position from the data. This command provides
    the initial phase guesses the model needs to start training.

    \b
    Adds to adata.obs:
      phase                   G1 / S / G2M  (initial phase label)
      S_score, G2M_score      phase scores from marker genes
      cycle_angle             raw arctan2 angle in [0, 2π]
      cycle_angle_uniform     quantile-uniform angle (used for training)

    Column names are saved in adata.uns["cyclevi"] so that 'cyclevi train'
    picks them up automatically without extra flags.
    """
    from cyclevi.prepare import load_adata

    if (s_genes_file is None) != (g2m_genes_file is None):
        raise click.UsageError("--s-genes and --g2m-genes must be provided together.")

    _section("Load")
    _info(input_path)
    adata_obj = load_adata(input_path, var_names=var_names, transpose=transpose)
    _ok(f"{adata_obj.n_obs:,} cells  ×  {adata_obj.n_vars:,} genes")

    _section("Phase initialization")
    _run_prepare(adata_obj, gene_id_type,
                 counts_layer, phase_key, angle_key, uniform_angle_key,
                 s_genes_file=s_genes_file, g2m_genes_file=g2m_genes_file)

    adata_obj.write_h5ad(output)
    _ok(f"Prepared AnnData saved → {output}")


# ── train ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Prepared .h5ad file (output of 'cyclevi prepare').")
@click.option("--output", required=True, type=click.Path(),
              help="Output directory.")
@_add_options(_TRAIN_OPTIONS)
def train(input_path, output,
          batch_key, labels_key, cycle_label_key, cycle_angle_key, layer,
          n_latent, n_hidden, n_layers, n_epochs, batch_size, lr):
    """Train a CycleVI model from a prepared .h5ad file.

    If the file was created by 'cyclevi prepare', the layer and column names
    are read from the file automatically — no extra flags needed.

    \b
    Writes to OUTPUT:
      model/             saved model
      latent_cycle.csv   2D cell cycle coordinates (z_cycle_x, z_cycle_y)
      latent_other.csv   non-cycling latent dimensions (z_1 … z_N)
      cycle_angles.csv   inferred cell cycle angle per cell
    """
    import anndata as ad

    _section("Load")
    _info(input_path)
    adata_obj = ad.read_h5ad(input_path)
    _ok(f"{adata_obj.n_obs:,} cells  ×  {adata_obj.n_vars:,} genes")

    _section("Training")
    _run_train(adata_obj, batch_key, labels_key, layer, cycle_label_key, cycle_angle_key,
               n_latent, n_hidden, n_layers, n_epochs, batch_size, lr, output)

    _section("Done")
    _ok(f"All outputs saved to {output}/")


# ── extract ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Input .h5ad file.")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True),
              help="Directory of a saved CycleVI model.")
@click.option("--output", required=True, type=click.Path(),
              help="Directory to write output CSVs.")
def extract(input_path, model_path, output):
    """Extract latent representations from a saved model.

    \b
    Writes to OUTPUT:
      latent_cycle.csv   2D circular cell cycle coordinates (z_cycle_x, z_cycle_y)
      latent_other.csv   non-cycling latent dimensions (z_1 … z_N)
      cycle_angles.csv   inferred cell cycle angle per cell (radians, -π to π)
    """
    import anndata as ad
    from cyclevi.model import CycleVI

    _section("Load")
    _info(input_path)
    adata_obj = ad.read_h5ad(input_path)
    _ok(f"{adata_obj.n_obs:,} cells  ×  {adata_obj.n_vars:,} genes")

    _info(f"Loading model from {model_path} ...")
    model = CycleVI.load(model_path, adata=adata_obj)
    _ok("Model loaded.")

    _section("Extract")
    os.makedirs(output, exist_ok=True)
    _save_latent_outputs(model, adata_obj, output)

    _section("Done")
    _ok(f"All outputs saved to {output}/")
