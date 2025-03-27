import scanpy as sc
import pandas as pd

# Step 1: Load the Filtered Data
adata = sc.read_h5ad("tahoe_data/plate1_DMSO_TF_00_CVCL_0480.h5ad")

# Extract number of genes and samples
num_genes = adata.shape[1]
num_samples = adata.shape[0]

# Create summary table
summary_df = pd.DataFrame({
    "Number of Genes": [num_genes],
    "Number of Samples": [num_samples]
})

# Display the summary table
print(summary_df)


# Step 2: Preprocessing
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize total counts per cell
sc.pp.log1p(adata)  # Log-transform the data
sc.pp.highly_variable_genes(adata, n_top_genes=20000)  # Select top 20000 most variable genes

adata = adata[:, adata.var.highly_variable].copy()

# Step 3: Scaling
sc.pp.scale(adata)  # Standardize the data

# Step 4: Perform PCA
#sc.pp.pca(adata)  # Compute PCA
#sc.pl.pca(adata, color="phase", title="PCA Plot Colored by Phase", show=True)


# Step 5: Perform t-SNE
sc.tl.tsne(adata)  # Compute t-SNE

sc.pl.tsne(adata, color="phase", title="t-SNE Plot Colored by Phase", show=True)

sc.pl.tsne(adata, color="TOP2A", title="t-SNE Plot Colored by TOP2A", show=True)

