#!/bin/sh
#SBATCH --job-name=para_cls
#SBATCH --account=luebbers_masters
#SBATCH --partition=gpu
#SBATCH -t 2:00:00
#SBATCH --gpus 1
#SBATCH --mem=32G
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=c.luebbers@stud.uni-goettingen.de 
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load miniforge3
module load cuda
source activate dpo_env

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Model: $1"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

# Set PYTHONPATH to point to the correct site-packages directory
export PYTHONPATH=/home/uni08/hpc/c.luebbers/u12246/.conda/envs/wahle_env/lib/python3.10/site-packages:$PYTHONPATH

export TOKENIZERS_PARALLELISM=false

python3 src/seq-cls.py --model_name $1 

# done

#TODO
# sbatch slurm_seq_cls.sh microsoft/deberta-v3-large
# sbatch slurm_seq_cls.sh microsoft/deberta-base
