#!/bin/sh
#SBATCH --job-name=paraphrase_cls
#SBATCH --account=luebbers_masters
#SBATCH --partition=gpu
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -c 1
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=c.luebbers@stud.uni-goettingen.de 
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load miniforge3
module load cuda
source activate wahle_env

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Model: $1"
echo "Task: $2"
echo "Dataset: $3"

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

python3 src/finetune_detection.py --model_name $1 --task_name $2 --dataset_name $3

# done
# sbatch slurm_cls.sh bert-base-uncased paraphrase-type-detection jpwahle/etpc
# sbatch slurm_cls.sh bert-base-uncased paraphrase-detection jpwahle/etpc
# sbatch slurm_cls.sh microsoft/deberta-base paraphrase-type-detection jpwahle/etpc
# sbatch slurm_cls.sh microsoft/deberta-base paraphrase-detection jpwahle/etpc
# sbatch slurm_cls.sh FacebookAI/roberta-base paraphrase-type-detection jpwahle/etpc
# sbatch slurm_cls.sh FacebookAI/roberta-base paraphrase-detection jpwahle/etpc
#TODO
# sbatch slurm_cls.sh nghuyong/ernie-2.0-large-en paraphrase-type-detection jpwahle/etpc
# sbatch slurm_cls.sh nghuyong/ernie-2.0-large-en paraphrase-detection jpwahle/etpc
