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
source activate dpo_env

export PYTHONPATH=/home/uni08/hpc/c.luebbers/u12246/.conda/envs/dpo_env/lib/python3.10/site-packages:$PYTHONPATH
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

# Start logging GPU usage in the background
gpu_log_file=gpu_usage_$SLURM_JOB_ID.log
echo "Logging GPU usage to $gpu_log_file"

while sleep 600; do  # Log every 10 minutes
   nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv >> $gpu_log_file
done &  # Run in the background

# Run the main Python workload
python3 src/finetune_pd.py --model_name $1 

# Wait for the Python job to finish
wait

echo "Job completed. Stopping GPU usage logging."

# done
# sbatch slurm_pd.sh microsoft/deberta-base

#TODO
# sbatch slurm_pd.sh microsoft/deberta-v3-large
