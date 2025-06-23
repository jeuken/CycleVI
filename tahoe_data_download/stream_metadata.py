import pandas as pd
import gcsfs


# initialize GCS file system for reading data from GCS
fs = gcsfs.GCSFileSystem()

infile = "gs://arc-ctc-tahoe100/2025-02-25/metadata/sample_metadata.parquet"

with fs.open(infile, 'rb') as f:
    sample_metadata = pd.read_parquet(f, engine='pyarrow')

sample_metadata['drug'].unique()

sample_metadata[sample_metadata['drug'] == 'Dapagliflozin']


