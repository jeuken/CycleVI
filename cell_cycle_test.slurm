#!/bin/bash -l
#SBATCH -J Cell_Cycle_Test
#SBATCH -N 1
#SBATCH -p defq
#SBATCH --gpus=1
#SBATCH --output=Cell_Cycle_Test.out
#SBATCH --error=Cell_Cycle_Test.err
#SBATCH --time=01:00:00
#SBATCH --mail-user=p.mozdzanowski@student.vu.nl
#SBATCH --mail-type=ALL

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Scratch dir: ${TMPDIR}"

# Move to scratch space
cd "$TMPDIR"

# Load necessary modules
module load shared 2022
module load python/3.12.4
module load PyTorch/1.12.1-foss-2021a-CUDA-11.3.1

# Set up virtual environment
python -m venv pcvi_env
source pcvi_env/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install numpy pandas scanpy anndata scvi-tools

# Run your application
echo "== Running on: $(hostname)"
cd ~/PerturbCycleVI/
python PerturbCycleVI.py
echo "== Job completed at $(date)"