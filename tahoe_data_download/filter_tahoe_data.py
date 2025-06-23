# ─────────────────────────────────────────────────────────────
# Imports
import scanpy as sc
import pandas as pd
import numpy as np
import ast
import os

# ─────────────────────────────────────────────────────────────
# Configuration

# Mapping of plate IDs to their corresponding data file paths
plate_paths = {
    "plate7": "tahoe_data/plate7_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    "plate8": "tahoe_data/plate8_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    "plate9": "tahoe_data/plate9_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
}

# Output path for the combined AnnData object
output_path = "tahoe_data/Lung_DMSO_allplates.h5ad"

# Path to summary file tracking all filtered experiments
summary_path = "tahoe_data/summary.csv"

# ─────────────────────────────────────────────────────────────
# Step 1: Load and filter metadata for all plates

filtered_adatas = []       # List to store filtered AnnData objects per plate
summary_entries = []       # List to accumulate summary entries

for plate, path in plate_paths.items():
    print(f"Loading {plate}...")
    
    # Load backed AnnData (only metadata is immediately loaded)
    adata_backed = sc.read_h5ad(path, backed='r')
    obs = adata_backed.obs

    # Filter for DMSO-treated SHP-77 or A549 cells
    filtered_idx = obs[
        (obs["cell_name"].isin(["SHP-77", "A549"])) &
        (obs["drug"] == "DMSO_TF")
    ].index

    # Skip if no cells match
    if len(filtered_idx) == 0:
        print(f"No matching cells found in {plate}.")
        continue

    # Load expression data for matching cells into memory
    adata_filtered = adata_backed[filtered_idx].to_memory()

    # Add plate metadata
    adata_filtered.obs["plate"] = plate

    # Extract numerical drug concentration from string-encoded field
    adata_filtered.obs["drugconc"] = adata_filtered.obs["drugname_drugconc"].astype(str).apply(
        lambda x: ast.literal_eval(x)[0][1] if x.startswith("[(") else np.nan
    )

    # Store filtered AnnData and summary info
    filtered_adatas.append(adata_filtered)

    df = adata_filtered.obs[["cell_name", "drug", "drugconc"]].copy()
    df["plate"] = plate
    summary_entries.append(df[["plate", "cell_name", "drug", "drugconc"]].drop_duplicates())

# ─────────────────────────────────────────────────────────────
# Step 2: Concatenate filtered data and save

# If no data found across plates, exit
if len(filtered_adatas) == 0:
    print("No matching cells found in any plate.")
    exit()

# Concatenate all filtered AnnData objects, track plate as batch key
adata_combined = filtered_adatas[0].concatenate(
    *filtered_adatas[1:],
    batch_key="batch",
    batch_categories=[d.obs['plate'].iloc[0] for d in filtered_adatas]
)

# Save to disk
adata_combined.write_h5ad(output_path)
print(f"Saved combined filtered dataset to {output_path}")

# ─────────────────────────────────────────────────────────────
# Step 3: Update summary file

# Combine all summary entries
summary_df = pd.concat(summary_entries).drop_duplicates()

# Append to existing summary file if it exists
if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    summary_df = pd.concat([existing, summary_df]).drop_duplicates()

# Save updated summary
summary_df.to_csv(summary_path, index=False)
print(f"Updated summary file at {summary_path}")
