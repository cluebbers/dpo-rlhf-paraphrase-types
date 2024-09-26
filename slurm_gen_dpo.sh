#!/bin/sh
#SBATCH --job-name=dpo_paraphrase_type_gen
#SBATCH --account=luebbers_masters
#SBATCH --partition=gpu
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH -c 1
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=c.luebbers@stud.uni-goettingen.de 
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load miniforge3
module load cuda
source activate unsloth_env

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Model: $1"
echo "Adapter: $2"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

# Set PYTHONPATH to point to the correct site-packages directory
export PYTHONPATH=/home/uni08/hpc/c.luebbers/u12246/.conda/envs/unsloth_env/lib/python3.11/site-packages:$PYTHONPATH

export TOKENIZERS_PARALLELISM=false

python3 src/dpo_generation_llama.py --model_name $1 --adapter_dir $2

#TODO
# sbatch slurm_gen_dpo.sh meta-llama/Llama-2-7b-hf llama/llama-7b-etpc
# sbatch slurm_gen_dpo.sh meta-llama/Llama-2-13b-hf llama/llama-13b-etpc