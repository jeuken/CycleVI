# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Imports
# Standard Library
import os
import logging
import warnings
import collections
from typing import TYPE_CHECKING, Callable, Iterable, Literal

# ─────────────────────────────────────────────────────────────
# Scientific Python Libraries
import numpy as np
import pandas as pd

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

from scvi.nn import FCLayers
from scvi.nn._utils import ExpActivation
from scvi.utils import (
    unsupported_if_adata_minified,
    setup_anndata_dsp,
)

# ─────────────────────────────────────────────────────────────
# Type Hints (not executed)
if TYPE_CHECKING:
    from torch.distributions import Distribution
    from anndata import AnnData

#os.chdir("/Users/piamozdzanowski/Documents/GitHub/PerturbCycleVI")
from power_spherical import PowerSpherical, HypersphericalUniform
# ─────────────────────────────────────────────────────────────
# Helpers
def _identity(x):
    return x

# Logger setup
logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(
    self,
    n_input: int,
    n_cycle_latent: int,
    n_drug_latent: int,
    n_cat_list: Iterable[int] = None,
    n_layers: int = 1,
    n_hidden: int = 128,
    dropout_rate: float = 0.1,
    distribution: str = "powerspherical",  # ← ADD THIS LINE
    var_eps: float = 1e-4,
    var_activation: Callable | None = None,
    return_dist: bool = False,
    **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.return_dist = return_dist
        
        # Remove 'distribution' from kwargs if present
        kwargs.pop("distribution", None)
        
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        # Cell cycle encoding
        self.theta_encoder = nn.Linear(n_hidden, 1)
        self.scale_encoder = nn.Linear(n_hidden, 1)

        # Drug susceptibility encoding
        self.drug_mean_encoder = nn.Linear(n_hidden, n_drug_latent)
        self.drug_scale_encoder = nn.Linear(n_hidden, n_drug_latent)

        self.var_activation = torch.nn.Softplus() if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        q = self.encoder(x, *cat_list)

        # ---------- Cell cycle latent ----------
        theta = self.theta_encoder(q).squeeze(-1)  # [N]
        loc = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # [N, 2]

        scale_logits = self.scale_encoder(q)
        raw_scale = self.var_activation(scale_logits)
        scale = torch.clamp(raw_scale.squeeze(-1), min=1e-4)

        dist_cycle = PowerSpherical(loc=loc, scale=scale)
        z_cycle = dist_cycle.rsample()

        # ---------- Drug susceptibility latent ----------
        drug_loc = self.drug_mean_encoder(q)
        drug_scale = self.var_activation(self.drug_scale_encoder(q)) + self.var_eps

        dist_drug = Normal(drug_loc, drug_scale.sqrt())
        z_drug = dist_drug.rsample()

        if self.return_dist:
            return (dist_cycle, z_cycle, dist_drug, z_drug)
        else:
            return (z_cycle, z_drug)

# ─────────────────────────────────────────────────────────────
# Classes
class DecoderCCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cell_types: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_cat_list: Iterable[int] = None,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        cycle_gene_mask: torch.Tensor | None = None,
        **kwargs
    ):
        super().__init__()

        # Cycle mask setup
        if cycle_gene_mask is None:
            self.register_buffer("cycle_mask", torch.ones(n_output))
        else:
            self.register_buffer("cycle_mask", cycle_gene_mask.float())

        self.n_output = n_output
        self.k_max = 2

        # Phase modulation weights
        self.W_cycle = nn.Parameter(torch.randn(2 * self.k_max, n_output))  # [2k, G]

        # Drug response decoder uses z_drug + d
        self.drug_response_decoder = FCLayers(
            n_in= 2,   # ← now expecting z_drug
            n_out=1,
            n_cont=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            inject_covariates=True,
            use_batch_norm=False,
            use_layer_norm=False,
            **kwargs
        )

        self.W_drug = nn.Parameter(torch.randn(1, n_output))
        self.b_type = nn.Parameter(torch.randn(n_cell_types, n_output))
        self.disp_raw = nn.Parameter(torch.randn(n_cell_types, n_output))

        self.px_scale_activation = (
            nn.Softmax(dim=-1) if scale_activation == "softmax" else nn.Softplus()
        )

    def forward(
        self,
        z_cycle: torch.Tensor,
        z_drug: torch.Tensor,
        library: torch.Tensor,
        d: torch.Tensor,
        t: torch.Tensor,
    ):
        assert d.ndim == 2 and d.size(1) == 1
        assert t.ndim == 1

        N = z_cycle.size(0)
        device = z_cycle.device
        control_mask = (d.squeeze() == 0)

        # Cell cycle phase angle
        z_cycle = z_cycle / (z_cycle.norm(dim=-1, keepdim=True) + 1e-8)
        theta = torch.atan2(z_cycle[:, 1], z_cycle[:, 0])  # [N]

        basis = [torch.cos(k * theta).unsqueeze(1) for k in range(1, self.k_max + 1)] + \
                [torch.sin(k * theta).unsqueeze(1) for k in range(1, self.k_max + 1)]
        phase_vector = torch.cat(basis, dim=1)  # [N, 2k]
        W_masked = self.W_cycle * self.cycle_mask  # [2k, G]
        cycle_effect = phase_vector @ W_masked     # [N, G]

        # Drug response
        r = torch.zeros(N, 1, dtype=z_drug.dtype, device=device)
        if (~control_mask).any():
            z_drug_treated = z_drug[~control_mask]
            d_treated = d[~control_mask]
            raw_r = self.drug_response_decoder(z_drug_treated, cont=d_treated)
            r[~control_mask] = F.softplus(raw_r)
        drug_effect = r @ self.W_drug  # [N, G]

        # Baseline + dispersion
        b_t = self.b_type[t]
        b_t = torch.where(control_mask.unsqueeze(1), b_t, b_t.detach())

        disp_raw = self.disp_raw[t]
        disp_raw = torch.where(control_mask.unsqueeze(1), disp_raw, disp_raw.detach())
        disp = F.softplus(disp_raw)

        eta = b_t + cycle_effect + drug_effect
        px_scale = self.px_scale_activation(eta)
        px_rate = torch.exp(library) * px_scale

        return (
            px_scale,  # [N, G]
            disp,      # [N, G]
            px_rate,   # [N, G]
            theta,     # [N]
            None,      # no phase probs
            r,         # [N, 1]
            None,      # unused
            self.W_cycle,
            self.W_drug,
            self.b_type,
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
        latent_distribution: Literal["normal", "ln","powerspherical"] = "powerspherical",
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
            n_input=n_input_encoder,
            n_cycle_latent=2,                 # 2D for circular latent
            n_drug_latent=n_latent - 2,       # the rest for drug susceptibility
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            var_activation=var_activation,
            return_dist=True,
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
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)
    
        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
    
        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            q_cycle, z_cycle, q_drug, z_drug = self.z_encoder(encoder_input, *categorical_input)
        else:
            q_cycle, z_cycle, q_drug, z_drug = self.z_encoder(encoder_input, batch_index, *categorical_input)
    
        z = torch.cat([z_cycle, z_drug], dim=-1)
    
        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded
    
        return {
            MODULE_KEYS.Z_KEY: (z_cycle, z_drug),
            MODULE_KEYS.QZ_KEY: (q_cycle, q_drug),
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
        }
    

    @auto_move_data
    def _cached_inference(
        self,
        qzm: torch.Tensor,  # mean for z_drug
        qzv: torch.Tensor,  # variance for z_drug
        observed_lib_size: torch.Tensor,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | None]:
        # z_cycle is not cached (must be re-inferred), so assume it’s sampled from the prior
        from power_spherical import HypersphericalUniform
    
        # Sample z_cycle from the prior
        z_cycle_prior = HypersphericalUniform(dim=2)
        z_cycle = z_cycle_prior.sample((qzm.shape[0],))  # [N, 2]
    
        # Use qzm and qzv for z_drug (Normal)
        qz_drug = Normal(qzm, qzv.sqrt())
        z_drug = qz_drug.sample()
    
        z = torch.cat([z_cycle, z_drug], dim=-1)  # [N, 2 + d_z_drug]
    
        library = torch.log(observed_lib_size)
        if n_samples > 1:
            z = z.unsqueeze(0).expand(n_samples, -1, -1)
            library = library.unsqueeze(0).expand(n_samples, -1, -1)
    
        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: (z_cycle_prior, qz_drug),
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
    ) -> dict[str, Distribution | None]:
        from scvi.distributions import NegativeBinomial, Normal, Poisson, ZeroInflatedNegativeBinomial
    
        # Extract d and t
        d = cont_covs[:, 0:1] if cont_covs is not None else None
        t = y.squeeze(-1)
    
        z_cycle, z_drug = z
        # Build decoder input
        decoder_input = z
        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input = torch.cat([z, batch_rep], dim=-1)
    
        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
    
        if not self.use_size_factor_key:
            size_factor = library
    
        # Run decoder
        px_scale, disp, px_rate, _, theta, r, _, W_cycle, W_drug, b_type = self.decoder(
            z_cycle, z_drug, size_factor, d, t)
        px_r = torch.exp(disp)
    
        # Handle separate priors for z_cycle and z_drug
        pz_cycle = HypersphericalUniform(dim=z_cycle.shape[-1])  # uniform on S¹
        pz_drug = Normal(torch.zeros_like(z_drug), torch.ones_like(z_drug))  # standard Gaussian
        
        pz = (pz_cycle, pz_drug)

        # Gene likelihood
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
    
        if self.latent_distribution == "powerspherical":
            pz = HypersphericalUniform(dim=2)
        else:
            pz = Normal(torch.zeros_like(z), torch.ones_like(z))
    
        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz
        }

    @unsupported_if_adata_minified  # Mark this method as unsupported if AnnData is in minified mode

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: torch.Tensor | float = 0.1,
    ) -> LossOutput:
        from torch.distributions import kl_divergence
    
        x = tensors[REGISTRY_KEYS.X_KEY]
    
        q_cycle, q_drug = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz = generative_outputs[MODULE_KEYS.PZ_KEY]
    
        # KL divergence for z_cycle (PowerSpherical)
        if isinstance(q_cycle, PowerSpherical) and isinstance(pz, HypersphericalUniform):
            kl_z_cycle = kl_divergence(q_cycle, pz).sum(dim=-1)
        else:
            raise TypeError("Invalid types for KL divergence on z_cycle")
    
        # KL divergence for z_drug (Normal)
        if isinstance(q_drug, Normal):
            prior_drug = Normal(torch.zeros_like(q_drug.loc), torch.ones_like(q_drug.scale))
            kl_z_drug = kl_divergence(q_drug, prior_drug).sum(dim=-1)
        else:
            raise TypeError("Invalid type for q_drug")
    
        kl_divergence_z = kl_z_cycle + kl_z_drug
    
        # KL divergence for library size (optional)
        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(
                inference_outputs[MODULE_KEYS.QL_KEY], generative_outputs[MODULE_KEYS.PL_KEY]
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)
    
        # Reconstruction loss
        reconst_loss = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x).sum(dim=-1)
    
        # ---- Angle-Based Repulsion Loss on z_cycle.loc ----
        locs = q_cycle.loc  # [N, 2], unit vectors on circle
        angles = torch.atan2(locs[:, 1], locs[:, 0]) % (2 * np.pi)  # [0, 2π]
    
        # Compute pairwise angular distances (shortest arc)
        delta = angles.unsqueeze(0) - angles.unsqueeze(1)
        delta = torch.remainder(delta + np.pi, 2 * np.pi) - np.pi  # in [-π, π]
    
        sigma = 0.5  # Controls how "close" is penalized
        angle_repulsion_matrix = torch.exp(-delta**2 / (2 * sigma**2))
        angle_repulsion = angle_repulsion_matrix.sum() - angle_repulsion_matrix.diag().sum()
        angle_repulsion /= (angles.shape[0] * (angles.shape[0] - 1))  # Normalize
    
        lambda_repulsion = 1.0  # Strength of the repulsion penalty
    
        # ---- Total Loss ----
        total_loss = torch.mean(
            reconst_loss + kl_weight * kl_divergence_z + kl_divergence_l + lambda_repulsion * angle_repulsion
        )
    
        return LossOutput(
            loss=total_loss,
            reconstruction_loss=reconst_loss,
            kl_local={
                MODULE_KEYS.KL_L_KEY: kl_divergence_l,
                MODULE_KEYS.KL_Z_KEY: kl_divergence_z,
            },
            extra_metrics={
                "angle_repulsion": angle_repulsion.detach(),
                "z_cycle": inference_outputs[MODULE_KEYS.Z_KEY][0],
                "z_drug": inference_outputs[MODULE_KEYS.Z_KEY][1],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
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
        latent_distribution: Literal[...] = "powerspherical",  # Latent distribution type
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
    def get_dual_latent_representation(
        self,
        adata=None,
        indices=None,
        give_mean=True,
        mc_samples=1,
        batch_size=None,
        return_dist=False,
        dataloader=None,
    ):
        """Returns separate latent representations for z_cycle and z_drug."""
        self._check_if_trained(warn=False)
    
        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")
    
        if dataloader is None:
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )
    
        z_cycle_list = []
        z_drug_list = []
    
        for tensors in dataloader:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
    
            # Unpack (z_cycle, z_drug)
            z_cycle, z_drug = outputs[MODULE_KEYS.Z_KEY]
    
            z_cycle_list.append(z_cycle.cpu().numpy())
            z_drug_list.append(z_drug.cpu().numpy())
    
        z_cycle = np.concatenate(z_cycle_list, axis=0)
        z_drug = np.concatenate(z_drug_list, axis=0)
    
        return z_cycle, z_drug

# Use full path to file on the cluster
adata = sc.read_h5ad("adata_Dapagliflozin_balanced.h5ad")

# Preserve raw counts
adata.layers["counts"] = adata.X.copy()
adata.raw = adata

# Set up AnnData for scvi-tools
CCVI.setup_anndata(
    adata,
    layer="counts",
    labels_key="cell_name",
    continuous_covariate_keys=["drugconc"],
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
mask = create_cell_cycle_gene_mask(adata, "cell_cycle_genes.txt")

# model = CCVI(adata,n_latent = 2,cycle_gene_mask=mask)
model = CCVI(adata,n_latent = 4,cycle_gene_mask=mask)

# Explicitly move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.module.to(device)

# Optionally check if it's really on GPU
print("Model device:", next(model.module.parameters()).device)

# Train the model
model.train(
    train_size=0.8,
    validation_size=0.2,
    shuffle_set_split=True,
    check_val_every_n_epoch=1,
    enable_progress_bar=True,
    simple_progress_bar = False,
    plan_kwargs={"max_kl_weight": 0.05}
)

model.save("ccvi_model_dapagliflozin", overwrite=True)
