from cyclevi.model import (
    CycleVI,
    CycleVI_VAE,
    CYCLE_REGISTRY_KEYS,
    DecoderCycleVI,
    PhaseAdversarialTrainingPlan,
    Classifier,
    create_cell_cycle_gene_mask,
)
from importlib.resources import files as _files


def get_cc_genes_path() -> str:
    """Return the path to the bundled human cell cycle genes CSV file."""
    return str(_files("cyclevi.data").joinpath("homo_sapiens_cc_genes.csv"))


__all__ = [
    "CycleVI",
    "CycleVI_VAE",
    "CYCLE_REGISTRY_KEYS",
    "DecoderCycleVI",
    "PhaseAdversarialTrainingPlan",
    "Classifier",
    "create_cell_cycle_gene_mask",
    "get_cc_genes_path",
]
