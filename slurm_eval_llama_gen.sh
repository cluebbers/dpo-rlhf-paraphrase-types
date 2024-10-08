#!/bin/sh
#SBATCH --job-name=eval_llama_gen
#SBATCH --account=luebbers_masters
#SBATCH --partition=gpu
#SBATCH -t 12:00:00
#SBATCH --gpus V100:1
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
echo "ETPC-Adapter: $2"
echo "DPO-Adapter: $3"
echo "IPO-Adapter: $4"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

export TOKENIZERS_PARALLELISM=false
# Set PYTHONPATH to point to the correct site-packages directory
export PYTHONPATH=/home/uni08/hpc/c.luebbers/u12246/.conda/envs/dpo_env/lib/python3.10/site-packages:$PYTHONPATH

python3 src/eval_llama_gen.py --model_name $1 --etpc_dir $2 --dpo_dir $3 --ipo_dir $4

#TODO
# sbatch slurm_eval_dpo_llama_gen.sh meta-llama/Llama-2-7b-hf out/gen-models/llama-7b-etpc out/gen-models/dpo_meta-llama-Llama-2-7b-hf_sigmoid out/gen-models/dpo_meta-llama-Llama-2-7b-hf_ipo
# sbatch slurm_eval_dpo_llama_gen.sh meta-llama/Llama-2-13b-hf out/gen-models/llama-13b-etpc out/gen-models/dpo_meta-llama-Llama-2-13b-hf_sigmoid out/gen-models/dpo_meta-llama-Llama-2-13b-hf_ipo

# sbatch slurm_eval_dpo_llama_gen.sh meta-llama/Llama-3.1-8B out/gen-models/llama-3.1-8b-etpc out/gen-models/dpo_meta-llama-Llama-3.1-8b_sigmoid out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_ipo
# sbatch slurm_eval_dpo_llama_gen.sh meta-llama/Llama-3.1-70B out/gen-models/llama-3.1-70b-etpc out/gen-models/dpo_meta-llama-Llama-3.1-70b_sigmoid out/gen-models/dpo_meta-llama-Llama-3.1-70b_ipo

