#!/bin/sh
#SBATCH --job-name=dpo_ptg
#SBATCH --account=luebbers_masters
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=RTX5000:1
#SBATCH --time=2:00:00
#SBATCH --mail-type=all          
#SBATCH --mail-user=c.luebbers@stud.uni-goettingen.de 
#SBATCH --output=./slurm_files/slurm-%x-%j.out   
#SBATCH --error=./slurm_files/slurm-%x-%j.err 

module load miniforge3
module load cuda
source activate dpo_env

export PYTHONPATH=/home/uni08/hpc/c.luebbers/u12246/.conda/envs/dpo_env/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch1/users/u12246/huggingface_cache
export TOKENIZERS_PARALLELISM=false

echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
echo "Model: $1"
echo "Task: $2"
echo "loss_type: $3"

python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

python3 src/dpo_ptg.py --model_name $1 --task_name $2 --loss_type $3

# done
# sbatch slurm_dpo_ptg.sh facebook/bart-large paraphrase-type-generation sigmoid
# sbatch slurm_dpo_ptg.sh facebook/bart-large paraphrase-type-generation ipo

#TODO
