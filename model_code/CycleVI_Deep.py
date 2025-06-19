
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Imports
# Standard Library
import os
import logging
import warnings
import collections
from typing import TYPE_CHECKING, Callable, Iterable, Literal
import math
from functools import partial
from scvi.utils import track
from scvi.distributions._utils import DistributionConcatenator
from scvi.model._utils import _get_batch_code_from_category, scrna_raw_counts_properties
from scvi.model.base._de_core import _de_core
 
# ─────────────────────────────────────────────────────────────
# Scientific Python Libraries
import numpy as np
import pandas as pd
from functools import partial

# ─────────────────────────────────────────────────────────────
# PyTorch Core 
import torch
from torch.distributions import Distribution
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList
from torch.nn.functional import one_hot
from torch.distributions import Normal

# ─────────────────────────────────────────────────────────────
# Single-cell analysis tools
import anndata
import scanpy as sc
from anndata import AnnData
from numbers import Number
from collections.abc import Iterator
from torch import Tensor
# ─────────────────────────────────────────────────────────────
# scvi-tools Core

import scvi
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    CategoricalObsField,
    CategoricalJointObsField,
    NumericalObsField,
    NumericalJointObsField,
    LayerField,
)


from scvi.nn import FCLayers
# ─────────────────────────────────────────────────────────────
# cvi-tools Models & Modules
from scvi.model._utils import _init_library_size
from scvi.model.base import (
    EmbeddingMixin,
    UnsupervisedTrainingMixin,
    ArchesMixin,
    BaseMinifiedModeModelClass,
    RNASeqMixin,
    VAEMixin,
)

from scvi.module._constants import MODULE_KEYS
from scvi.module.base import (
    BaseMinifiedModeModuleClass,
    EmbeddingModuleMixin,
    LossOutput,
    auto_move_data,
)

from scvi.nn import Encoder
from scvi.utils import (
    unsupported_if_adata_minified,
    setup_anndata_dsp,
)

# ─────────────────────────────────────────────────────────────
# Helpers
def _identity(x):
    return x

# Logger setup
logger = logging.getLogger(__name__)

from scvi.train import AdversarialTrainingPlan, TrainRunner

from scvi.train import TrainingPlan
from scvi.module.base import BaseModuleClass

from scvi import REGISTRY_KEYS
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scvi.train import TrainingPlan
from scvi.module.base import BaseModuleClass
from scvi._constants import REGISTRY_KEYS


class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_labels, n_layers=2, logits=True):
        super().__init__()
        self.logits = logits

        layers = []
        in_dim = n_input
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, n_hidden))
            layers.append(nn.ReLU())
            in_dim = n_hidden
        layers.append(nn.Linear(in_dim, n_labels))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PhaseAdversarialTrainingPlan(TrainingPlan):
    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: str = "Adam",
        optimizer_creator=None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: str = "elbo_validation",
        lr_min: float = 0.0,
        scale_adversarial_loss: float | str = "auto",
        compile: bool = False,
        compile_kwargs: dict | None = None,
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            compile=compile,
            compile_kwargs=compile_kwargs,
            **loss_kwargs,
        )

        n_phases = 3  
        self.adversarial_classifier = Classifier(
            n_input=self.module.n_latent - 2,  
            n_hidden=32,
            n_labels=n_phases,
            n_layers=2,
            logits=True,
        )

        self.scale_adversarial_loss = scale_adversarial_loss
        self.automatic_optimization = False

    def loss_adversarial_classifier(self, z_other, phase_index, predict_true_class=True):
        n_classes = self.adversarial_classifier.network[-1].out_features
        logits = self.adversarial_classifier(z_other)
        cls_logits = torch.nn.LogSoftmax(dim=1)(logits)

        if predict_true_class:
            cls_target = torch.nn.functional.one_hot(phase_index.squeeze(-1), n_classes)
        else:
            one_hot = torch.nn.functional.one_hot(phase_index.squeeze(-1), n_classes)
            cls_target = (~one_hot.bool()).float() / (n_classes - 1)

        l_soft = cls_logits * cls_target
        return -l_soft.sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
            self.log("kl_weight", self.kl_weight, on_step=True, on_epoch=False)

        kappa = 1 - self.kl_weight if self.scale_adversarial_loss == "auto" else self.scale_adversarial_loss
        batch_cat = batch[REGISTRY_KEYS.CAT_COVS_KEY][:, 0]  # assumes "phase" is first categorical covariate

        opt1, opt2 = self.optimizers() if isinstance(self.optimizers(), list) else (self.optimizers(), None)
        outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)

        z = outputs["z"]
        z_other = z[:, 2:]  # Skip 2D z_cycle
        loss = scvi_loss.loss

        if kappa > 0:
            fool_loss = self.loss_adversarial_classifier(z_other, batch_cat, predict_true_class=False)
            loss += fool_loss * kappa * 1000

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")

        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        if opt2 is not None:
            cls_loss = self.loss_adversarial_classifier(z_other.detach(), batch_cat, predict_true_class=True)
            cls_loss *= kappa
            opt2.zero_grad()
            self.manual_backward(cls_loss)
            opt2.step()

    def configure_optimizers(self):
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}

        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
            )
            config1["lr_scheduler"] = {"scheduler": scheduler, "monitor": self.lr_scheduler_metric}

        params2 = filter(lambda p: p.requires_grad, self.adversarial_classifier.parameters())
        optimizer2 = torch.optim.Adam(params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay)

        if "lr_scheduler" in config1:
            return [config1["optimizer"], optimizer2], [config1["lr_scheduler"]]
        return [config1["optimizer"], optimizer2]


class DecoderCCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_cat_list: list[int] = None,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: str = "softmax",
        cycle_gene_mask: torch.Tensor = None,
        n_fourier: int = 3,
        **kwargs
    ):
        super().__init__()

        if cycle_gene_mask is None:
            cycle_gene_mask = torch.ones(n_output, dtype=torch.bool)
        elif cycle_gene_mask.shape[0] != n_output:
            raise ValueError("`cycle_gene_mask` must match n_output")

        self.register_buffer("cycle_mask", cycle_gene_mask.float())
        self.n_output = n_output
        self.n_fourier = n_fourier

        # Intermediate FC layers (including final layer)
        self.non_cycle_fc = FCLayers(
            n_in=n_input-2,
            n_out=n_hidden,
            n_cont=0,  # only drug concentration
            n_cat_list=None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=True,
            use_layer_norm=False,
            inject_covariates=False,
            activation_fn=nn.ReLU,
            use_activation=True,
            bias=True
        )
        self.non_cycle_linear = nn.Linear(n_hidden, n_output)
        
        # Fourier weights for all genes
        self.fourier_W = nn.Parameter(0.01 *torch.randn(2 * n_fourier, n_output))

        # Only update weights for cycle genes
        grad_mask = torch.zeros_like(self.fourier_W)
        grad_mask[:, cycle_gene_mask] = 1.0
        self.fourier_W.register_hook(lambda grad: grad * grad_mask.to(grad.device))

        # Dispersion
        self.disp_raw = nn.Parameter(0.01 * torch.randn(n_output))

        self.px_scale_activation = nn.Softmax(dim=-1) if scale_activation == "softmax" else nn.Softplus()

        
    def forward(self, z: torch.Tensor, library: torch.Tensor, t: torch.Tensor, remove_cell_cycle: bool = False):

        z_cycle = z[..., 0:2]        
        z_latent = z[..., 2:]

        x_input = z_latent
        x = self.non_cycle_fc(x_input)
        non_cycle_out = self.non_cycle_linear(x)
        x, y = z_cycle[..., 0], z_cycle[..., 1]
        angle = torch.atan2(y, x)
        if remove_cell_cycle:
            cycle_effect = 0.0  
        else:
            basis = [torch.cos(k * angle) for k in range(1, self.n_fourier + 1)] + \
                    [torch.sin(k * angle) for k in range(1, self.n_fourier + 1)]
            fourier_basis = torch.stack(basis, dim=-1)  # [N, 2 * n_fourier]
            cycle_effect = torch.matmul(fourier_basis, self.fourier_W) * self.cycle_mask

        eta = non_cycle_out + cycle_effect  # [N, G]


        px_scale = self.px_scale_activation(eta)
        px_rate = torch.exp(library) * px_scale
        disp = self.disp_raw

        return (
            px_scale,
            disp,
            px_rate,
            None,
            angle.unsqueeze(1),
            None,
            None,
            self.fourier_W,
            None,
            non_cycle_out  # return just NN output for debugging if needed
        )


# ─────────────────────────────────────────────────────────────

class CC_VAE(
# 1. CLASS INHERITANCE
    EmbeddingModuleMixin, BaseMinifiedModeModuleClass):
# 2. ClASS DOCSTRING
    """
    Variational auto-encoder.

    This class implements a variational autoencoder (VAE) for single-cell RNA-seq data.
    It inherits from an embedding mixin (for latent representations) and a base module class
    that supports minified AnnData mode.
    
    Parameters
    ----------
    n_input
        Number of input features.
    n_batch
        Number of batches. If ``0``, no batch correction is performed.
    n_labels
        Number of labels.
    n_hidden
        Number of nodes per hidden layer. Passed into Encoder and DecoderSCVI.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers. Passed into Encoder and DecoderSCVI.
    n_continuous_cov
        Number of continuous covariates.
    n_cats_per_cov
        A list of integers containing the number of categories for each categorical covariate.
    dropout_rate
        Dropout rate, passed into the Encoder.
    dispersion
        Parameter controlling dispersion for the likelihood distribution.
    log_variational
        Whether to apply log1p to input data for numerical stability.
    gene_likelihood
        Likelihood distribution for gene expression (e.g. "zinb", "nb", "poisson").
    latent_distribution
        Distribution for the latent space (e.g. "normal", "ln").
    encode_covariates
        Whether to concatenate covariates to the gene expression before encoding.
    deeply_inject_covariates
        If True and n_layers > 1, covariates are injected at each hidden layer.
    batch_representation
        How to represent batch information ("one-hot" or "embedding").
    use_batch_norm
        Where to use Batch Normalization ("none", "encoder", "decoder", "both").
    use_layer_norm
        Where to use Layer Normalization ("none", "encoder", "decoder", "both").
    use_size_factor_key
        If True, use an AnnData.obs column as the scaling factor for the likelihood.
    use_observed_lib_size
        If True, use the observed library size for scaling.
    library_log_means
        Numpy array with means for log library sizes (if not using observed library size).
    library_log_vars
        Numpy array with variances for log library sizes (if not using observed library size).
    var_activation
        Callable for ensuring positivity of the variance output in the encoder.
    extra_encoder_kwargs
        Extra keyword arguments for the Encoder.
    extra_decoder_kwargs
        Extra keyword arguments for the DecoderSCVI.
    batch_embedding_kwargs
        Keyword arguments for the batch embedding layer (if using embedding representation).
    """

# 3. CONSTRUCTOR
    '''
    - Checks and stores all parameters
    - Handles dispersion type (per gene/cell/batch/label)
    - Initializes encoders: 
            z_encoder for latent variables
        latent_distribution: Literal["normal", "ln"] = "normal",
            # Distribution used for latent variables ("normal" or "logistic normal")
        encode_covariates: bool = False,  # Whether to concatenate covariates with gene expression data
        deeply_inject_covariates: bool = True,  # Whether to inject covariates at deeper layers in the encoder/decoder
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
            # How to represent batch information (one-hot vector or learned embedding)
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            # Where to apply Batch Normalization in the network
            l_encoder for library size
    - Initializes decoder based on latent space + covariates
    - Handles categorical and continuous covariates'''
    
    

    def __init__(
        self,
        n_input: int,   # Number of input features (e.g., number of genes)
        n_batch: int = 0,  # Number of batches; 0 implies no batch correction
        n_labels: int = 0,  # Number of labels (if any)
        n_hidden: int = 128,  # Number of nodes in each hidden layer
        n_latent: int = 10,   # Dimensionality of the latent space
        n_layers: int = 1,    # Number of layers in the encoder/decoder networks
        n_continuous_cov: int = 0,  # Number of continuous covariates
        n_cats_per_cov: list[int] | None = None,  # List with number of categories for each categorical covariate
        dropout_rate: float = 0.1,  # Dropout rate for the neural network layers
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene-label",
            # Dispersion model: how variance is modeled (per gene, per batch, etc.)
        log_variational: bool = True,  # Whether to apply log1p on input data for numerical stability
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
            # Likelihood model for gene expression (Zero-Inflated Negative Binomial, etc.)
        latent_distribution: Literal["normal", "ln"] = "normal",
            # Distribution used for latent variables ("normal" or "logistic normal")
        encode_covariates: bool = False,  # Whether to concatenate covariates with gene expression data
        deeply_inject_covariates: bool = True,  # Whether to inject covariates at deeper layers in the encoder/decoder
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
            # How to represent batch information (one-hot vector or learned embedding)
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            # Where to apply Batch Normalization in the network
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
            # Where to apply Layer Normalization in the network
        use_size_factor_key: bool = False,  # Whether to use size factors from AnnData.obs as scaling factors
        use_observed_lib_size: bool = True,  # Whether to use the observed library size directly for scaling
        library_log_means: np.ndarray | None = None,  # Precomputed means of log library sizes (if not observed)
        library_log_vars: np.ndarray | None = None,   # Precomputed variances of log library sizes (if not observed)
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,  # Activation for variance output (e.g., torch.exp)
        extra_encoder_kwargs: dict | None = None,  # Additional parameters for the Encoder
        extra_decoder_kwargs: dict | None = None,  # Additional parameters for the DecoderSCVI
        batch_embedding_kwargs: dict | None = None,  # Additional parameters for batch embedding layer (if used)
        cycle_gene_mask: torch.Tensor | None = None,
    ):

        super().__init__()  # Initialize parent classes (EmbeddingModuleMixin and BaseMinifiedModeModuleClass)

        # Store various model parameters as attributes
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        # If size factor key is used, then use_observed_lib_size is True; otherwise, use the provided flag
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size

        # If not using observed library size, then library_log_means and library_log_vars must be provided
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, must provide library_log_means and library_log_vars."
                )
            # Register these as buffers (non-parameter tensors that move with the model)
            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        # Setup batch representation; if embedding is chosen, initialize an embedding layer
        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            # Initialize embedding for batches using a key from REGISTRY_KEYS and extra kwargs if provided
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            # Get the embedding dimension from the initialized embedding layer
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        # Determine where to apply normalization in the encoder and decoder
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Calculate the input dimension for the encoder. Start with gene counts and add covariates if encoding them.
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            n_input_encoder += batch_dim * encode_covariates  # Add batch embedding dimension if applicable
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            # For one-hot, add n_batch as a categorical variable
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        # Only include categorical covariates if requested
        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        # Initialize the encoder for the latent variable "z"
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,  # Return a distribution rather than a fixed tensor
            **_extra_encoder_kwargs,
        )

        # Calculate input dimension for the decoder: latent dimension plus continuous covariates
        n_input_decoder = n_latent
        if self.batch_representation == "embedding":
            n_input_decoder += batch_dim  # Add embedding dimension for batch representation

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        # Initialize the decoder module that maps latent space back to the original input space
        self.decoder = DecoderCCVI(
            n_input=n_input_decoder,
            n_output=n_input,
            n_cell_types=n_labels,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            cycle_gene_mask=cycle_gene_mask, # Or set a specific float if you want to control bump width
            **_extra_decoder_kwargs,
        )
# 4. Prepare tensors for inference 
    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        # Decide which data loader to use based on full_forward_pass flag and the minified data type
        if full_forward_pass or self.minified_data_type is None:
            loader = "full_data"
        elif self.minified_data_type in [
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR_WITH_COUNTS,
        ]:
            loader = "minified_data"
        else:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")

        # For full data, return the standard tensors used in the model
        if loader == "full_data":
            return {
                MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
                MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
                MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            }
        else:
            # For minified data, use cached latent parameters
            return {
                MODULE_KEYS.QZM_KEY: tensors[REGISTRY_KEYS.LATENT_QZM_KEY],
                MODULE_KEYS.QZV_KEY: tensors[REGISTRY_KEYS.LATENT_QZV_KEY],
                REGISTRY_KEYS.OBSERVED_LIB_SIZE: tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE],
            }

# 5. Prepare tensors for generative model  
    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        # Retrieve and transform size factor if provided
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        # Return a dictionary mapping module keys to the appropriate tensors/distributions
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.Y_KEY: tensors[REGISTRY_KEYS.LABELS_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
        }

# 6. For each cell, computes the mean and variance of the log library size for the corresponding batch.
    def _compute_local_library_params(
        self,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes local library parameters.

        For each cell, computes the mean and variance of the log library size
        for the corresponding batch.
        """
        from torch.nn.functional import linear

        n_batch = self.library_log_means.shape[1]  # Number of batches from the library means buffer
        # Compute local means using one-hot encoding for the batch index and linear transformation
        local_library_log_means = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
        )
        # Compute local variances similarly
        local_library_log_vars = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )

        return local_library_log_means, local_library_log_vars

    @auto_move_data  # Automatically move inputs/outputs to the correct device (CPU/GPU)

# 7. Encodes input data into latent variables:
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process with normalization by library size."""
        # Step 1: Compute observed library size (sum over genes per cell)
        library = torch.sum(x, dim=1, keepdim=True)  # shape [N, 1]
    
        # Step 2: Normalize expression per cell
        x_normalized = x / (library + 1e-8)  # Add small epsilon to avoid division by zero
    
        # Step 3: Apply log1p for numerical stability if enabled
        if self.log_variational:
            x_normalized = torch.log1p(x_normalized)
    
        # Step 4: Prepare encoder input
        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_normalized, cont_covs), dim=-1)
        else:
            encoder_input = x_normalized
    
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
    
        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            qz, z = self.z_encoder(encoder_input, *categorical_input)
        else:
            qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
    
        # If you're not using observed_lib_size, compute encoded one
        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded
    
        # Expand for MC sampling if needed
        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1))) \
                      if self.use_observed_lib_size else ql.sample((n_samples,))
    
        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: torch.log(library + 1e-8),  # used in decoder
        }


    @auto_move_data
    def _cached_inference(
        self,
        qzm: torch.Tensor,          # Cached latent mean values
        qzv: torch.Tensor,          # Cached latent variance values
        observed_lib_size: torch.Tensor,  # Observed library size values
        n_samples: int = 1,         # Number of samples for Monte Carlo approximation
    ) -> dict[str, torch.Tensor | None]:
        """Run the cached inference process."""

        # Reconstruct the latent distribution using the cached parameters
        qz = Normal(qzm, qzv.sqrt())
        # Sample from the latent distribution; using sample() (non-reparameterized)
        untran_z = qz.sample() if n_samples == 1 else qz.sample((n_samples,))
        # Transform the sampled latent variables if necessary
        z = self.z_encoder.z_transformation(untran_z)
        # Compute the library by taking log of the observed library size
        library = torch.log(observed_lib_size)
        if n_samples > 1:
            library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: None,
            MODULE_KEYS.LIBRARY_KEY: library,
        }

    @auto_move_data

# 8. Decodes latent z back to gene expression
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
        remove_cell_cycle: bool = False,  # New argument
    ) -> dict[str, Distribution | None]:
        from scvi.distributions import NegativeBinomial, Normal, Poisson, ZeroInflatedNegativeBinomial
    
        t = y.squeeze(-1)
    
        decoder_input = z
        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input = torch.cat([z, batch_rep], dim=-1)
    
        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
    
        if not self.use_size_factor_key:
            size_factor = library
    
        # Pass remove_cell_cycle to decoder
        px_scale, disp, px_rate, _, angle, radius, _, W_fourier, _, baseline = self.decoder(
            decoder_input, size_factor, t, remove_cell_cycle=remove_cell_cycle
        )

    
        px_r = torch.exp(disp)
    
        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=None, scale=px_scale)
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(rate=px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            px = Normal(px_rate, px_r, normal_mu=px_scale)
    
        if self.use_observed_lib_size:
            pl = None
        else:
            local_library_log_means, local_library_log_vars = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
    
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
    
        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
            "angle": angle,
        }


    @unsupported_if_adata_minified  # Mark this method as unsupported if AnnData is in minified mode
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | torch.Tensor | None],
        kl_weight: torch.Tensor | float = 1.0,
    ) -> LossOutput:
        from torch.distributions import kl_divergence
    
        x = tensors[REGISTRY_KEYS.X_KEY]
    
        # KL divergence for z
        kl_divergence_z = kl_divergence(
            inference_outputs[MODULE_KEYS.QZ_KEY], generative_outputs[MODULE_KEYS.PZ_KEY]
        ).sum(dim=-1)
    
        # KL for library size
        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(
                inference_outputs[MODULE_KEYS.QL_KEY], generative_outputs[MODULE_KEYS.PL_KEY]
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)
    
        # Reconstruction loss
        reconst_loss = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x).sum(-1)
    
        # Latent position
        z_latent = inference_outputs[MODULE_KEYS.QZ_KEY].loc
        x_latent, y_latent = z_latent[:, 0], z_latent[:, 1]
        angle = torch.atan2(y_latent, x_latent)
        radius = torch.sqrt(x_latent**2 + y_latent**2 + 1e-6)
    
        # G2M and S scores from continuous covariates
        cont_covs = tensors[REGISTRY_KEYS.CONT_COVS_KEY]
        target_angle = cont_covs[:, 0]
    
        # Angle loss (squared angular distance)
        delta_angle = angle - target_angle
        angle_loss = torch.mean(1.0 - torch.cos(delta_angle))
    
        # Radius penalty (high as r → 0)
        radius_penalty = torch.mean(torch.exp(-10 * radius))
        
        # Weighted loss terms
        weighted_kl_local = kl_weight * kl_divergence_z + kl_divergence_l
        cycle_pos_weight = 100 * (1.0 - kl_weight)**2
        weighted_angle_loss = cycle_pos_weight * angle_loss
        weighted_radius_penalty = 100 * (kl_weight**2)*radius_penalty
    
        # Total loss
        loss = torch.mean(reconst_loss + weighted_kl_local) + weighted_angle_loss + weighted_radius_penalty
    
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local={
                MODULE_KEYS.KL_L_KEY: kl_divergence_l,
                MODULE_KEYS.KL_Z_KEY: kl_divergence_z,
            },
            extra_metrics={
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
                "angle_loss": angle_loss,
                "radius_penalty": radius_penalty,
                "weighted_angle_loss": weighted_angle_loss,
                "weighted_radius_penalty": weighted_radius_penalty,
            },
        )

    

    @torch.inference_mode()

# 10. Samples gene expression from the posterior predictive distribution
    def sample(
        self,
        tensors: dict[str, torch.Tensor],  # Input tensors for sampling
        n_samples: int = 1,                  # Number of Monte Carlo samples to draw per observation
        max_poisson_rate: float = 1e8,       # Maximum value to clip Poisson rate to avoid numerical issues
    ) -> torch.Tensor:
        r"""Generate predictive samples from the posterior predictive distribution.

        The posterior predictive distribution is denoted as :math:`p(\hat{x} \mid x)`, where
        :math:`x` is the input data and :math:`\hat{x}` is the sampled data.

        We sample from this distribution by first sampling ``n_samples`` times from the posterior
        distribution :math:`q(z \mid x)` for a given observation, and then sampling from the
        likelihood :math:`p(\hat{x} \mid z)` for each of these.
        """
        from scvi.distributions import Poisson

        inference_kwargs = {"n_samples": n_samples}
        # Run a forward pass to get generative outputs (without computing loss)
        _, generative_outputs = self.forward(
            tensors, inference_kwargs=inference_kwargs, compute_loss=False
        )

        dist = generative_outputs[MODULE_KEYS.PX_KEY]
        if self.gene_likelihood == "poisson":
            # Handle potential issues on MPS devices by clamping the Poisson rate
            dist = (
                Poisson(torch.clamp(dist.rate.to("cpu"), max=max_poisson_rate))
                if self.device.type == "mps"
                else Poisson(torch.clamp(dist.rate, max=max_poisson_rate))
            )

        # Draw samples from the likelihood distribution; shape depends on n_samples
        samples = dist.sample()
        # If multiple samples were drawn, permute dimensions so that output is (n_obs, n_vars, n_samples)
        samples = torch.permute(samples, (1, 2, 0)) if n_samples > 1 else samples

        return samples.cpu()  # Return samples on CPU

    @torch.inference_mode()
    @auto_move_data

# Estimates marginal log-likelihood with Monte Carlo sampling

    def marginal_ll(
        self,
        tensors: dict[str, torch.Tensor],  # Input tensors for marginal likelihood computation
        n_mc_samples: int,  # Total number of Monte Carlo samples for estimation
        return_mean: bool = False,  # Whether to return the mean marginal likelihood over cells
        n_mc_samples_per_pass: int = 1,  # Number of samples per computation pass (to reduce memory usage)
    ):
        """Compute the marginal log-likelihood of the data under the model."""
        from torch import logsumexp
        from torch.distributions import Normal

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = []  # List to accumulate log probabilities over multiple passes
        if n_mc_samples_per_pass > n_mc_samples:
            warnings.warn(
                "Number of chunks is larger than the total number of samples, setting it to the "
                "number of samples",
                RuntimeWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            n_mc_samples_per_pass = n_mc_samples
        n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
        for _ in range(n_passes):
            # For each pass, run a forward pass to get inference outputs and loss components
            inference_outputs, _, losses = self.forward(
                tensors,
                inference_kwargs={"n_samples": n_mc_samples_per_pass},
                get_inference_input_kwargs={"full_forward_pass": True},
            )
            qz = inference_outputs[MODULE_KEYS.QZ_KEY]
            ql = inference_outputs[MODULE_KEYS.QL_KEY]
            z = inference_outputs[MODULE_KEYS.Z_KEY]
            library = inference_outputs[MODULE_KEYS.LIBRARY_KEY]

            # Get the reconstruction loss from the losses output
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Compute log probabilities for the latent variable and reconstruction
            p_z = (
                Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale)).log_prob(z).sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                # Compute additional log probabilities for library size if not observed
                local_library_log_means, local_library_log_vars = self._compute_local_library_params(batch_index)
                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )
                q_l_x = ql.log_prob(library).sum(dim=-1)
                log_prob_sum += p_l - q_l_x
            if n_mc_samples_per_pass == 1:
                log_prob_sum = log_prob_sum.unsqueeze(0)

            to_sum.append(log_prob_sum)
        # Concatenate all passes and compute log-sum-exp for a Monte Carlo estimate
        to_sum = torch.cat(to_sum, dim=0)
        batch_log_lkl = logsumexp(to_sum, dim=0) - np.log(n_mc_samples)
        if return_mean:
            batch_log_lkl = torch.mean(batch_log_lkl).item()
        else:
            batch_log_lkl = batch_log_lkl.cpu()
        return batch_log_lkl
      
class CCVI(EmbeddingMixin,        # Adds methods for getting latent representations
    RNASeqMixin,                  # Adds single-cell RNA-seq-specific logic
    VAEMixin,                     # Adds methods for working with a VAE model
    ArchesMixin,                  # Adds functionality for transfer learning (ARCHES)
    UnsupervisedTrainingMixin,    # Adds methods for unsupervised training
    BaseMinifiedModeModelClass):
    
    _module_cls = CC_VAE
    _LATENT_QZM_KEY = "ccvi_latent_qzm"  # Key for the latent mean in AnnData
    _LATENT_QZV_KEY = "ccvi_latent_qzv"  # Key for the latent variance in AnnData
    _training_plan_cls = PhaseAdversarialTrainingPlan

# 4. CONSTRUCTOR
    def __init__(
        self,
        adata: AnnData | None = None,  # Input data; can be None (if adata is not provided, the model will delay initialization until train is called).
        n_hidden: int = 128,           # Hidden units per layer
        n_latent: int = 10,            # Dimensionality of latent space
        n_layers: int = 1,             # Number of layers in encoder/decoder neural networks
        dropout_rate: float = 0.1,     # Dropout rate
        dispersion: Literal[...] = "gene-label",         # Type of dispersion parameter
        gene_likelihood: Literal[...] = "nb",    # Distribution to model gene expression
        latent_distribution: Literal[...] = "normal",  # Latent distribution type
        cycle_gene_mask: torch.Tensor | None = None,
        **kwargs,                      # Any other parameters passed to the VAE
    ):
        super().__init__(adata)  # Call the constructor of the parent mixin/base classes

        # Store inputs in a dictionary
        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            "cycle_gene_mask": cycle_gene_mask,
            **kwargs,
        }

        # Create a summary string
        self._model_summary_string = (
            "CCVI drug model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}."
        )

        # If lazy initialization is enabled (adata is not provided), postpone model creation until training
        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model was initialized without `adata`. The module will be initialized when "
                "calling `train`. This behavior is experimental and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            # Create a list with the number of categories per categorical variable
            n_cats_per_cov = (
                self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
                if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
                else None
            )

            # Get number of batches
            n_batch = self.summary_stats.n_batch

            # Determine if size factor is provided in the data
            use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry

            # Initialize library size params if needed
            library_log_means, library_log_vars = None, None
            if (
                not use_size_factor_key
                and self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
            ):
                library_log_means, library_log_vars = _init_library_size(
                    self.adata_manager, n_batch
                )

            # Instantiate the actual VAE model
            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,  # Number of genes
                n_batch=n_batch,
                n_labels=self.summary_stats.n_labels,
                n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                n_cats_per_cov=n_cats_per_cov,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                dispersion=dispersion,
                gene_likelihood=gene_likelihood,
                latent_distribution=latent_distribution,
                use_size_factor_key=use_size_factor_key,
                library_log_means=library_log_means,
                library_log_vars=library_log_vars,
                cycle_gene_mask=cycle_gene_mask,
                **kwargs,
            )

            # Set minified type to the model (used for memory optimization)
            self.module.minified_data_type = self.minified_data_type

        # Save init parameters for reproducibility
        self.init_params_ = self._get_init_params(locals())

# 5. Define setup_anndata for preproccessing AnnData

    @classmethod
    @setup_anndata_dsp.dedent  # Automatically formats docstring from template
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,  # Which layer of AnnData.X to use
        batch_key: str | None = None,  # Batch annotation column in adata.obs
        labels_key: str | None = None,  # Label annotation column
        size_factor_key: str | None = None,  # Precomputed size factor
        categorical_covariate_keys: list[str] | None = None,  # Categorical covariates
        continuous_covariate_keys: list[str] | None = None,   # Continuous covariates
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """

        # Get arguments as a dictionary
        setup_method_args = cls._get_setup_method_args(**locals())

        # Define how to extract relevant fields from AnnData
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        # If this is a "minified" AnnData, add extra required fields
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)

        # Create a manager to track and validate all fields
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)

        # Register fields into the manager
        adata_manager.register_fields(adata, **kwargs)

        # Register the manager for this class (global to model)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
   
    
    
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        transform_batch: list[Number | str] | None = None,
        gene_list: list[str] | None = None,
        library_size: float | Literal["latent"] = 1,
        n_samples: int = 1,
        n_samples_overall: int = None,
        weights: Literal["uniform", "importance"] | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        silent: bool = True,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
        remove_cell_cycle: bool = False,
        **importance_weighting_kwargs,
    ) -> np.ndarray | pd.DataFrame:
        
        
        if dataloader is None:
            adata = self._validate_anndata(adata)
            if indices is None:
                indices = np.arange(adata.n_obs)
            if n_samples_overall is not None:
                assert n_samples == 1
                n_samples = n_samples_overall // len(indices) + 1
            scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
            transform_batch = _get_batch_code_from_category(
                self.get_anndata_manager(adata, required=True), transform_batch
            )
            gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)
        else:
            scdl = dataloader
            gene_mask = slice(None)
            transform_batch = [None]
    
        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn("return_numpy must be True if n_samples > 1 and return_mean is False.")
            return_numpy = True
    
        generative_output_key = "mu" if library_size == "latent" else "scale"
        scaling = 1 if library_size == "latent" else library_size
    
        exprs = []
        zs = []
        qz_store = DistributionConcatenator()
        px_store = DistributionConcatenator()
    
        for tensors in scdl:
            per_batch_exprs = []
            for batch in track(transform_batch, disable=silent):
                generative_kwargs = self._get_transform_batch_gen_kwargs(batch)
                generative_kwargs["remove_cell_cycle"] = remove_cell_cycle
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs={"n_samples": n_samples},
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )
                px_generative = generative_outputs["px"]
                exp_ = px_generative.get_normalized(generative_output_key)
                exp_ = exp_[..., gene_mask] * scaling
                per_batch_exprs.append(exp_[None].cpu())
                if weights == "importance":
                    qz_store.store_distribution(inference_outputs["qz"])
                    px_store.store_distribution(px_generative)
    
            zs.append(inference_outputs["z"].cpu())
            per_batch_exprs = torch.cat(per_batch_exprs, dim=0).mean(0).numpy()
            exprs.append(per_batch_exprs)
    
        cell_axis = 1 if n_samples > 1 else 0
        exprs = np.concatenate(exprs, axis=cell_axis)
    
        if n_samples_overall is not None:
            exprs = exprs.reshape(-1, exprs.shape[-1])
            n_samples_ = exprs.shape[0]
            if weights is None or weights == "uniform":
                p = None
            else:
                qz = qz_store.get_concatenated_distributions(axis=0)
                px = px_store.get_concatenated_distributions(axis=0 if n_samples == 1 else 1)
                p = self._get_importance_weights(
                    adata, indices, qz, px, torch.concat(zs, dim=cell_axis), **importance_weighting_kwargs
                )
            exprs = exprs[np.random.choice(n_samples_, n_samples_overall, p=p, replace=True)]
        elif n_samples > 1 and return_mean:
            exprs = exprs.mean(0)
    
        if (return_numpy is None or not return_numpy) and dataloader is None:
            return pd.DataFrame(exprs, columns=adata.var_names[gene_mask], index=adata.obs_names[indices])
        return exprs
    
    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: list[str] | None = None,
        group2: str | None = None,
        idx1: list[int] | list[bool] | str | None = None,
        idx2: list[int] | list[bool] | str | None = None,
        mode: Literal["vanilla", "change"] = "vanilla",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: list[str] | None = None,
        batchid2: list[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        weights: Literal["uniform", "importance"] | None = "uniform",
        filter_outlier_cells: bool = False,
        remove_cell_cycle: bool = False,
        importance_weighting_kwargs: dict | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        adata = self._validate_anndata(adata)
        col_names = adata.var_names
        importance_weighting_kwargs = importance_weighting_kwargs or {}
    
        model_fn = partial(
            self.get_normalized_expression,
            return_numpy=True,
            n_samples=1,
            batch_size=batch_size,
            weights=weights,
            remove_cell_cycle=remove_cell_cycle,
            **importance_weighting_kwargs,
        )
        representation_fn = self.get_latent_representation if filter_outlier_cells else None
    
        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            representation_fn,
            groupby,
            group1,
            group2,
            idx1,
            idx2,
            all_stats,
            scrna_raw_counts_properties,
            col_names,
            mode,
            batchid1,
            batchid2,
            delta,
            batch_correction,
            fdr_target,
            silent,
            **kwargs,
        )
    
        return result


# Use full path to file on the cluster
adata = sc.read_h5ad("deepcycle_data_cycle_angle.h5ad")

# Preserve raw counts
adata.layers["counts"] = adata.layers["matrix"]

# Set up AnnData for scvi-tools
CCVI.setup_anndata(
    adata,
    layer="counts",
    continuous_covariate_keys=["cycle_angle_uniform"],
    categorical_covariate_keys=["phase"]
)


# Create a boolean mask for genes in the cell cycle gene list
def create_cell_cycle_gene_mask(
    adata: AnnData,
    gene_list_txt: str,
    var_column: str = None  # If None, uses adata.var_names
) -> torch.Tensor:
    """
    Create a mask indicating which genes in adata are cell cycle genes.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing gene expression data.
    gene_list_txt : str
        Path to a text file with one gene name per line.
    var_column : str, optional
        Column in adata.var to use for gene names. Defaults to adata.var_names.
    
    Returns
    -------
    torch.BoolTensor
        Boolean mask of shape (n_genes,) — True for cell cycle genes.
    """
    cell_cycle_genes = set(pd.read_csv(gene_list_txt, header=None)[0].str.upper())

    if var_column:
        gene_names = adata.var[var_column].astype(str).str.upper().tolist()
    else:
        gene_names = [str(g).upper() for g in adata.var_names]

    mask = [g in cell_cycle_genes for g in gene_names]
    return torch.tensor(mask, dtype=torch.bool)

# ─────────────────────────────────────────────────────────────
mask = create_cell_cycle_gene_mask(adata, "GO_cell_cycle_annotation_human.txt")

model = CCVI(adata,n_latent = 10,cycle_gene_mask=None)

# Explicitly move model to GPU (if available)
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

model.module.to(device)
# Optionally check if it's really on GPU
print("Model device:", next(model.module.parameters()).device)


model.train(
    max_epochs=400,
    train_size=0.7,
    validation_size=0.2,
    shuffle_set_split=True,
    enable_progress_bar=True,
    simple_progress_bar=False,
    batch_size=128
)


model.save("deepcycle_model", overwrite=True)

