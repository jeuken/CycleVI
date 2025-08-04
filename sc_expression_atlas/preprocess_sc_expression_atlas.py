import scanpy as sc
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import numpy as np


def quantile_to_uniform_2pi(adata, angle_key, output_key):

    all_angles = adata.obs[angle_key].values.reshape(-1, 1)

    # Check for NaNs
    if np.isnan(all_angles).any():
        raise ValueError(f"{angle_key} contains NaNs. Please clean or impute missing values.")

    # Fit on control, transform all
    qt = QuantileTransformer(
        output_distribution='uniform',
        random_state=0,
        n_quantiles=min(len(all_angles), 1000),
        subsample=len(all_angles)
    )
    qt.fit(all_angles)
    transformed = qt.transform(all_angles).flatten()
    adata.obs[output_key] = transformed * 2 * np.pi

adata = sc.datasets.ebi_expression_atlas('E-MTAB-9067')

cycle_markers = pd.read_csv('Homo_sapiens.csv', index_col=0)

sc.tl.score_genes_cell_cycle(adata, s_genes=cycle_markers.loc['S','geneID'], g2m_genes=cycle_markers.loc['G2/M','geneID'])


score_angle = np.arctan2(adata.obs["G2M_score"], adata.obs["S_score"])
score_angle = np.mod(score_angle, 2 * np.pi)  # wrap to [0, 2π]


# Store in obs
adata.obs["cycle_angle"] = score_angle

quantile_to_uniform_2pi(
    adata,
    angle_key="cycle_angle",
    output_key="cycle_angle_uniform",
)

adata.write_h5ad('E-MTAB-9067-processed.h5ad')
