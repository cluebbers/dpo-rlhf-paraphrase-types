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

tested with python=3.10.15 and requirements.txt

## Run

### Generate Prompts from APTY-Ranked dataset

   ```bash
   python src/generate_promps_apty_ranked.py
   ```

### Training

   ```bash
   python src/dpo_llama_generation.py \
   --model_name=meta-llama/Llama-2-7b-hf \
   --adapter_dir=src/llama/llama-7b-etpc
   ```

### Generate Paraphrases

   ```bash
   python src/generation.py
   ```
