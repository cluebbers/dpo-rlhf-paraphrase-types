#!/bin/sh
#SBATCH --job-name=pd
#SBATCH --account=luebbers_masters
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G  # for microsoft/deberta-v3-large
#SBATCH --gpus=V100:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=all              
#SBATCH --mail-user=c.luebbers@stud.uni-goettingen.de 
#SBATCH --output=./slurm_files/slurm-%x-%j.out    
#SBATCH --error=./slurm_files/slurm-%x-%j.err      

module load miniforge3
module load cuda
source activate wahle_env

export PYTHONPATH=/home/uni08/hpc/c.luebbers/u12246/.conda/envs/wahle_env/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch1/users/u12246/huggingface_cache
export TOKENIZERS_PARALLELISM=false

echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
echo "Model: $1"

python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

python3 src/finetune_pd.py --model_name $1 

# done
# sbatch slurm_pd.sh microsoft/deberta-base

#TODO
# sbatch slurm_pd.sh microsoft/deberta-v3-large
