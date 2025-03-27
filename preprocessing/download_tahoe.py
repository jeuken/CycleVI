import scanpy as sc
import pandas as pd

# Step 1: Load only metadata (not expression data)
adata_path = "tahoe_data/plate1_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
adata_obs = sc.read_h5ad(adata_path, backed='r').obs

print("loaded obs")
plate = "plate1"

# Step 2: Filter adata_obs for drug == DMSO or Dinaciclib and cell_name == A549 or HT-29
filtered_obs = adata_obs[
    adata_obs['drug'].isin(['DMSO_TF', 'Dinaciclib']) &
    adata_obs['cell_name'].isin(['A549', 'HT-29'])
]

print("filtered obs")
# Step 3: If any cells meet these criteria, load the corresponding expression (backed mode)
if not filtered_obs.empty:
    adata_backed = sc.read_h5ad(adata_path, backed='r')
    filtered_indices = adata_backed.obs_names.isin(filtered_obs.index)
    
    
    print("filtering expression")
    # Step 4: Convert to memory before saving
    adata_filtered = adata_backed[filtered_indices, :].to_memory()

else:
    print("No cells found matching the filter criteria.")

print(adata_filtered.var_names[:4].tolist())

 # Step 5: Save the filtered dataset
adata_filtered.obs["drugconc"] = adata_filtered.obs["drugname_drugconc"].apply(
    lambda x: float(x[0][1])
)
adata_filtered.write_h5ad(f"tahoe_data/{plate}_DMSO_Dinaciclib_A549_HT-29.h5ad")