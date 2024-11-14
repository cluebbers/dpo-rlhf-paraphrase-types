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

```python
conda create --name dpo_env \
    python=3.11 \
    pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate dpo_env

# Paraphrase Type Generation
pip install transformers trl peft bitsandbytes nltk rouge scikit-learn
# Paraphrase Type Detection
pip install evaluate optuna
```

The used environment can be found in dpo_env.yml

## Run

### Paraphrase Type Generation Training

RLHF training on APTY

```python
# Reward Training on APTY
python3 src/reward.py
# PPO Training
python3 src/ppo.py
```

DPO/IPO Training on APTY

```python
# loss_type=sigmoid for DPO
# loss_type=ipo for IPO
python3 src/dpo_llama_ptg.py \
--model_name=meta-llama/Llama-3.1-8B \
--adapter_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc \
--loss_type=sigmoid
```

Evaluation of Base Model, SFT/ETPC, DPO/APTY, IPO/APTY

```python
python3 src/eval_llama_ptg.py \
--model_name=meta-llama/Llama-3.1-8B \
--etpc_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc \
--dpo_dir=out/gen-models/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid \
--ipo_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo
```

### Paraphrase Type Detection Training

```python
# Paraphrase Detection on QQP
python3 src/sft_pd.py
# Paraphrase Type Detection on ETPC
python3 src/sft_ptd.py
```

### Plots

```python
notebooks/plots.ipynb
```
