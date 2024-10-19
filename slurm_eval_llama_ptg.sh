#!/bin/sh
#SBATCH --job-name=eval_llama_ptg
#SBATCH --account=luebbers_masters
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=V100:1
#SBATCH --time=12:00:00
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
echo "ETPC-Adapter: $2"
echo "DPO-Adapter: $3"
echo "IPO-Adapter: $4"

python --version
python -m torch.utils.collect_env
nvcc -V
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Current environment: $(which python)"
echo $PATH

python3 src/eval_llama_ptg.py --model_name $1 --etpc_dir $2 --dpo_dir $3 --ipo_dir $4

# Done
# sbatch slurm_eval_llama_ptg.sh meta-llama/Llama-2-7b-hf out/gen-models/llama-7b-etpc out/gen-models/dpo_meta-llama-Llama-2-7b-hf_sigmoid out/gen-models/dpo_meta-llama-Llama-2-7b-hf_ipo
# sbatch slurm_eval_llama_ptg.sh meta-llama/Llama-2-13b-hf out/gen-models/llama-13b-etpc out/gen-models/dpo_meta-llama-Llama-2-13b-hf_sigmoid out/gen-models/dpo_meta-llama-Llama-2-13b-hf_ipo
# sbatch slurm_eval_llama_ptg.sh meta-llama/Llama-3.1-8B out/gen-models/llama-3.1-8b-etpc out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_sigmoid out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_ipo

#TODO
# sbatch slurm_eval_llama_ptg.sh meta-llama/Llama-3.1-8B out/gen-models/llama-3.1-8b-etpc out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_sigmoid out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_ipo
# sbatch slurm_eval_llama_ptg.sh meta-llama/Llama-3.1-70B out/gen-models/llama-3.1-70b-etpc out/gen-models/dpo_meta-llama-Llama-3.1-70b_sigmoid out/gen-models/dpo_meta-llama-Llama-3.1-70b_ipo
