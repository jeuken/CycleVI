
"""
Created on Tue Mar 11 14:25:47 2025

@author: piamozdzanowski
"""

import scanpy as sc
import gcsfs

# Define the file path
infile = 'gs://arc-ctc-tahoe100/2025-02-25/h5ad/plate2_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad'

# Initialize the GCS filesystem
fs = gcsfs.GCSFileSystem()

# Open and read the file from Google Cloud Storage
with fs.open(infile, 'rb') as f:
    adata = sc.read_h5ad(f)

# Check the data
adata