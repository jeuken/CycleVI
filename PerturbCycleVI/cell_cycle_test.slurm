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

# Load Conda and activate the environment
module load miniconda/23
eval "$(conda shell.bash hook)"
conda activate pcvi_env

# Change to your working directory
cd ~/PerturbCycleVI/

# Run your script
echo "== Running on: $(hostname)"
python PerturbCycleVI.py
echo "== Job completed at $(date)"

