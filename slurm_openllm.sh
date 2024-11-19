#!/bin/sh
#SBATCH --job-name=openllm
#SBATCH --account=luebbers_masters
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=V100:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=c.luebbers@stud.uni-goettingen.de 
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

module load miniforge3
module load cuda
source activate /scratch-scc/users/u12246/environments/openllm_env

export PYTHONPATH=/scratch-scc/users/u12246/environments/openllm_env/lib/python3.11/site-packages:$PYTHONPATH
export HF_HOME=/scratch-scc/users/u12246/huggingface_cache
export TOKENIZERS_PARALLELISM=false

echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
echo "Model: $1"
echo "Output Path: $2"

python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

python3 src/openllm.pyp --pretrained=$1 --output_path=$2

# ToDo
# sbatch slurm_openllm.sh cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc out/gen-models/openllm/etpc