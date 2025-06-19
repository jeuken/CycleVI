import scanpy as sc
import gcsfs

fs = gcsfs.GCSFileSystem()

plates = [7, 8, 9]
for plate in plates:
    infile = f'gs://arc-ctc-tahoe100/2025-02-25/h5ad/plate{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad'
    with fs.open(infile, 'rb') as f:
        adata = sc.read_h5ad(f)
    adata.write(f"adata_plate{plate}.h5ad")  # saves locally
