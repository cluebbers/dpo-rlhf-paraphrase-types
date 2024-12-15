# Enhancing Paraphrase Type Generation: The Impact of DPO and RLHF Evaluated with Human-Ranked Data

Repository for master thesis "Enhancing Paraphrase Type Generation: The Impact of DPO and RHLF Evaluated with Human-Ranked Data"
Student: Christopher L. Luebbers
Supervisor: Dominik Meier, Terry Ruas

## Requirements

To install requirements:

```setup
conda create --name dpo_env \
    python=3.11 \
    pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate dpo_env
pip install -r requirements.txt
```

This project uses huggingface datasets and models.
Llama models are gated and you need to sign up with Huggingface and accept the community licence agreement at [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B).

Datasets:

- [ETPC Dataset](https://huggingface.co/datasets/jpwahle/etpc)
- [APTY-ranked Dataset](https://huggingface.co/datasets/worta/apty)
- base sentences for evaluation can be found at [generate_apt_paraphrases/Sentences](https://github.com/worta/generate_apt_paraphrases)

## Training

### Paraphrase Type Generation

To train the model SFT/ETPC on the APTY dataset with reward modeling to get the model Reward/APTY:

```python
python3 src/reward.py
```

To train the model SFT/ETPC with Reward/APTY using PPO to get the model RLHF/APTY:

```python
python3 src/ppo.py
```

To train the model SFT/ETPC on the APTY-ranked dataset using DPO to get the model DPO/APTY:

```python
python3 src/dpo_llama_ptg.py \
--model_name=meta-llama/Llama-3.1-8B \
--adapter_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc \
--loss_type=sigmoid
```

To train the model SFT/ETPC on the APTY-ranked dataset using IPO to get the model IPO/APTY:

```python
python3 src/dpo_llama_ptg.py \
--model_name=meta-llama/Llama-3.1-8B \
--adapter_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc \
--loss_type=ipo
```

### Paraphrase Type Detection

To train the model microsoft/deberta-base on the Glue/QQP dataset on paraphrase detection (binary classification):

```python
python3 src/sft_pd.py
```

To continue training the model on the ETPC dataset on paraphrase type detection (multilabel classification):

```python
python3 src/sft_ptd.py
```

### Evaluation

Evaluation of Base Model, SFT/ETPC, DPO/APTY, IPO/APTY on paraphrase type generation:

```python
python3 src/eval_llama_ptg.py \
--model_name=meta-llama/Llama-3.1-8B \
--etpc_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc \
--dpo_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid \
--ipo_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo
```

For Open LLM Leaderboard evaluation, [submit your model](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/).

Further evaluation is done in the jupyter notebook. All plots and tables from the project are generated there.

```python
notebooks/plots.ipynb
```


## Pre-trained models

- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [SFT/ETPC](https://huggingface.co/cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc)
- [Reward/APTY](https://huggingface.co/cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc-apty-reward)
- [DPO/APTY](https://huggingface.co/cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid)
- [IPO/APTY](https://huggingface.co/cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo)
- [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base)

## Results

- Directly Leveraging Human-Ranked Data for Improved Paraphrase Types: By training models with DPO on the APTY dataset, we achieve a 3~\% higher human annotation accuracy on specific paraphrase types than a supervised fine-tuned model.
- Enhancing User-Aligned Quality: Human annotators preferred DPO-generated paraphrases over baseline outputs in 16~\% more cases (section~\ref{sec:human_preferences}).
- Uncovering Metric Limitations and Guiding Future Evaluation: Comparisons between automatic metrics and human rankings show weak correlations (Spearman’s $r < 0.3$), exposing the shortcomings of standard evaluation methods.
- Advancing Fine-Grained Paraphrase Type Detection: We introduce a paraphrase type detection model achieving F1 scores of 0.91 on addition/deletion and 0.77 on same polarity substitution, enabling more granular assessments.
- Broadening Impact to Complex Reasoning Tasks: Incorporating human-ranked data also boosts performance on multistep soft reasoning (MuSR) tasks by 38~\%, demonstrating that human-guided optimization extends beyond paraphrase quality.

## Citation

If you use the APTY dataset, please cite:

```bib
@misc{meier2024humanunderstandingparaphrasetypes,
      title={Towards Human Understanding of Paraphrase Types in ChatGPT}, 
      author={Dominik Meier and Jan Philip Wahle and Terry Ruas and Bela Gipp},
      year={2024},
      eprint={2407.02302},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.02302}, 
}
```

The SFT/ETPC model is provided by

```bib
@inproceedings{wahle-etal-2023-paraphrase,
    title = "Paraphrase Types for Generation and Detection",
    author = "Wahle, Jan Philip  and
      Gipp, Bela  and
      Ruas, Terry",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.746",
    doi = "10.18653/v1/2023.emnlp-main.746",
    pages = "12148--12164",
    abstract = "Current approaches in paraphrase generation and detection heavily rely on a single general similarity score, ignoring the intricate linguistic properties of language. This paper introduces two new tasks to address this shortcoming by considering paraphrase types - specific linguistic perturbations at particular text positions. We name these tasks Paraphrase Type Generation and Paraphrase Type Detection. Our results suggest that while current techniques perform well in a binary classification scenario, i.e., paraphrased or not, the inclusion of fine-grained paraphrase types poses a significant challenge. While most approaches are good at generating and detecting general semantic similar content, they fail to understand the intrinsic linguistic variables they manipulate. Models trained in generating and identifying paraphrase types also show improvements in tasks without them. In addition, scaling these models further improves their ability to understand paraphrase types. We believe paraphrase types can unlock a new paradigm for developing paraphrase models and solving tasks in the future.",
}
```

If you use the ETPC datase, please cite:

```bib
@inproceedings{kovatchev-etal-2018-etpc,
    title = "{ETPC} - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negation",
    author = "Kovatchev, Venelin  and
      Mart{\'\i}, M. Ant{\`o}nia  and
      Salam{\'o}, Maria",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1221",
}
```

If you use DeBERTa, please cite:

```bib
@inproceedings{he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```

## Licence

Licensed under the [Apache 2.0](LICENSE) license.

Llama-3.1 models are licensed under the [LLaMA 3.1 Community License Agreement](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)
