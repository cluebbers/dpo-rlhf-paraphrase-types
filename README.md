# dpo-rhlf-paraphrase-types

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/cluebbers/dpo-rhlf-paraphrase-types.git
   cd repository
   
2. Create and activate a virtual environment:
   ```bash
   conda create --name unsloth_env \
    python=3.10 \
    pytorch-cuda=<11.8/12.1> \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

pip install trl

## Run
   ```bash
   python src/run_dpo.py \
    --dataset_name=/home/slim/Documents/06_DataScience/06_Projects/M.Inf.2901_Master/emnlp23-paraphrase-types-dpo/data/apty_dataset.parquet \
    --model_name_or_path=meta-llama/Llama-2-7b-hf \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_steps 10 \
    --output_dir="dpo_llama_apty_output" \
    --optim rmsprop \
    --warmup_steps 10 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
