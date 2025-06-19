import scanpy as sc
import pandas as pd
import numpy as np
import ast
import os

os.chdir("/Users/piamozdzanowski/Documents/GitHub/PerturbCycleVI")


# --- Config ---
adata_path = "tahoe_data/plate13_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
plate = "plate13"
summary_path = "tahoe_data/summary.csv"
output_path = f"tahoe_data/{plate}_irinotecan_HT29.h5ad"

# --- Step 1: Load metadata ---
adata_obs = sc.read_h5ad(adata_path, backed='r').obs

# --- Step X: Count and sort drugs by number of treated cells ---
drug_counts = (
    adata_obs["drug"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "drug", "drug": "cell_count"})
    .sort_values("count", ascending=False)
)

print("Top drugs by treated cell count:")
print(drug_counts[0:40])



# --- Step 2: Filter metadata ---
filtered_obs = adata_obs[
    adata_obs['drug'].isin(['DMSO_TF','Irinotecan (hydrochloride)']) &
    adata_obs['cell_name'].isin([
    "HT-29",])]

if filtered_obs.empty:
    print("No cells found matching the filter criteria.")
    exit()

# --- Step 3: Load and filter expression data ---
adata_backed = sc.read_h5ad(adata_path, backed='r')
adata_filtered = adata_backed[adata_backed.obs_names.isin(filtered_obs.index)].to_memory()

# --- Step 4: Extract drug concentration ---
adata_filtered.obs["drugconc"] = adata_filtered.obs["drugname_drugconc"].astype(str).apply(
    lambda x: ast.literal_eval(x)[0][1] if x.startswith("[(") else np.nan
)

# --- Step 5: Save filtered dataset ---
adata_filtered.write_h5ad(output_path)
print(f"Saved filtered data to {output_path}")

# --- Step 6: Append summary of combinations ---
summary_df = adata_filtered.obs[["cell_name", "drug", "drugconc"]].copy()
summary_df["plate"] = plate
summary_df = summary_df[["plate", "cell_name", "drug", "drugconc"]].drop_duplicates()

# Append to summary file
if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    summary_df = pd.concat([existing, summary_df]).drop_duplicates()

summary_df.to_csv(summary_path, index=False)
print(f"Updated summary file at {summary_path}")

