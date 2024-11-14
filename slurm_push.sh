#!/bin/sh
#SBATCH --job-name=push
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
# scratch-scc not availableon scc-a100
#export HF_HOME=/scratch-scc/users/u12246/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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

echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
echo "Model: $1"
echo "Adapter: $2"
echo "Loss: $3"

python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

python3 src/push.py --model_name $1 --adapter_dir $2 --output_dir $3

# Done
# sbatch slurm_push.sh meta-llama/Llama-3.1-8B cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc out/gen-models/Llama-3.1-8B-etpc

#TODO
# sbatch slurm_push.sh meta-llama/Llama-3.1-8B cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid out/gen-models/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid
# sbatch slurm_push.sh meta-llama/Llama-3.1-8B cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo out/gen-models/Llama-3.1-8B-paraphrase-type-generation-apty-ipo