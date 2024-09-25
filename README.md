# dpo-rhlf-paraphrase-types

Repository for master thesis "Enhancing Paraphrase Type Generation: The Impact of DPO and RHLF Evaluated with Human-Ranked Data"
Student: Christopher L. Luebbers
Supervisor: Dominik Meier, Terry Ruas

## Sources

- finetuned Llama2-7b model: <https://github.com/jpwahle/emnlp23-paraphrase-types>
- DPO script: <https://huggingface.co/docs/trl/dpo_trainer>
- Dataset: <https://huggingface.co/datasets/worta/apty>
- unsloth: <https://github.com/unslothai/unsloth>

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/cluebbers/dpo-rhlf-paraphrase-types.git
   cd repository
   ```

2. Create and activate a virtual environment:

   ```bash
   conda create --name unsloth_env \
    python=3.10 \
    pytorch-cuda=<11.8/12.1> \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

   conda activate unsloth_env

   pip install unsloth

   pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

   pip install trl

   pip install fire
   ```

## Run

### Generate Prompts from APTY-Ranked dataset

   ```bash
   python src/generate_promps_apty_ranked.py
   ```

### Training

   ```bash
   python src/dpo_generation.py \
   --model_name_or_path=meta-llama/Llama-2-7b-hf \
   --output_dir="out/dpo_llama-7b_apty"
   ```

### Generate Paraphrases

   ```bash
   python src/generation.py
   ```
