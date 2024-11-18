#!/bin/sh
#SBATCH --job-name=test
#SBATCH --account=luebbers_masters
#SBATCH --partition=scc-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=c.luebbers@stud.uni-goettingen.de 
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

module load miniforge3
module load cuda
source activate dpo_env

export PYTHONPATH=/home/uni08/hpc/c.luebbers/u12246/.conda/envs/dpo_env/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch-scc/users/u12246/huggingface_cache
export TOKENIZERS_PARALLELISM=false

echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

python3 src/test.py
