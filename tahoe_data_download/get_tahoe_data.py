import scanpy as sc
import pandas as pd
import numpy as np
import ast
import os

# --- Configuration ---

# Path to the input full AnnData file (backed mode)
adata_path = "tahoe_data/plate13_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"

# Plate identifier to tag output and summary
plate = "plate13"

# Path to the summary CSV file (used to track drug-cell-treatment combos)
summary_path = "tahoe_data/summary.csv"

# Path where the filtered dataset will be saved
output_path = f"tahoe_data/{plate}_irinotecan_HT29.h5ad"

# --- Step 1: Load metadata only ---

# Load just the .obs (cell-level metadata) using backed mode to save memory
adata_obs = sc.read_h5ad(adata_path, backed='r').obs

# --- Step 2: Summarize drug usage ---

# Count the number of cells treated with each drug and display the top 40
drug_counts = (
    adata_obs["drug"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "drug", "drug": "cell_count"})
    .sort_values("cell_count", ascending=False)
)

print("Top drugs by treated cell count:")
print(drug_counts.head(40))

# --- Step 3: Filter metadata ---

# Select only rows for Irinotecan or DMSO treatments in the HT-29 cell line
filtered_obs = adata_obs[
    adata_obs['drug'].isin(['DMSO_TF', 'Irinotecan (hydrochloride)']) &
    adata_obs['cell_name'].isin(["HT-29"])
]

# Exit early if no relevant cells are found
if filtered_obs.empty:
    print("No cells found matching the filter criteria.")
    exit()

# --- Step 4: Filter expression data ---

# Load the full dataset again (still in backed mode)
adata_backed = sc.read_h5ad(adata_path, backed='r')

# Subset to only the cells of interest and load them fully into memory
adata_filtered = adata_backed[adata_backed.obs_names.isin(filtered_obs.index)].to_memory()

# --- Step 5: Parse drug concentration ---

# Extract the drug concentration from the stringified list of tuples
# Example input: "[('Irinotecan (hydrochloride)', 10.0)]"
adata_filtered.obs["drugconc"] = adata_filtered.obs["drugname_drugconc"].astype(str).apply(
    lambda x: ast.literal_eval(x)[0][1] if x.startswith("[(") else np.nan
)

# --- Step 6: Save the filtered AnnData object ---

adata_filtered.write_h5ad(output_path)
print(f"Saved filtered data to {output_path}")

# --- Step 7: Update the treatment summary CSV ---

# Create a small DataFrame with key treatment information
summary_df = adata_filtered.obs[["cell_name", "drug", "drugconc"]].copy()
summary_df["plate"] = plate
summary_df = summary_df[["plate", "cell_name", "drug", "drugconc"]].drop_duplicates()

# If the summary file already exists, append and deduplicate
if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    summary_df = pd.concat([existing, summary_df]).drop_duplicates()

# Write the updated summary to disk
summary_df.to_csv(summary_path, index=False)
print(f"Updated summary file at {summary_path}")
