# Enhancing Paraphrase Type Generation: The Impact of DPO and RLHF Evaluated with Human-Ranked Data

Repository for master thesis "Enhancing Paraphrase Type Generation: The Impact of DPO and RLHF Evaluated with Human-Ranked Data"
Student: Christopher L. Luebbers
Supervisors: Dominik Meier, Dr. Terry Lima Ruas

Paraphrasing adds variety to language by rephrasing ideas without altering their meaning.
Paraphrases enhance text comprehension, information retrieval, and natural language applications by improving communication clarity.
Paraphrase types provide insights into linguistic variation, facilitating fine-grained semantic analysis and robust language modeling.
These insights enhance tasks like text simplification, translation and question answering, extending the utility of paraphrase generation.
Current paraphrase-type generation systems fail to align with human preferences due to a lack of human-ranked datasets and reliance on automated metrics like BLEU and ROUGE.
We use a human-ranked paraphrase-type dataset and apply Direct Preference Optimization (DPO) to guide type-specific paraphrase generation and detection.
This work is the first to apply DPO training for paraphrase-type generation.

## Requirements

To install requirements:

```setup
conda create --name dpo_env \
    python=3.11 \
    pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate dpo_env
pip3 install -r requirements.txt
```

This project uses huggingface datasets and models.
Llama models are gated and you need to sign up with Huggingface and accept the community licence agreement at [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B).

Datasets:

- [ETPC Dataset](https://huggingface.co/datasets/jpwahle/etpc)
- [APTY-ranked Dataset](https://huggingface.co/datasets/worta/apty)
- base sentences for evaluation can be found at [generate_apt_paraphrases/Sentences](https://github.com/worta/generate_apt_paraphrases). Copy files into ./out/basesentences

Output:
Output directory is currently hardcoded to ./out
You probably want to adapt this or even better, add it to script arguments.

## Paraphrase Type Generation (PTG) Training

- Llama-3.1-8B
  Please note: Our scripts use LoRA adapters.
  We store the merged model in the huggingface repository main directory and adapter files in a subfolder 'adapter'.
  We load the adapters from those subfolders.
  This structure is necessary to submit the models to [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/add).
  If you want to train your own models, adapt the scripts accordingly, meaning you should uncomment the line 'subfolder="adapter"'.
  Using LoRA, you should be able to train the models on consumer grade hardware.
  Our models were trained on a GeForce RTX 3080 (10 GB).
  We also commented the push_to_hub functionality, so you do not accidently push your models.

- BART-large
  ParaScore is tested for a limited number of [models](https://github.com/shadowkiller33/parascore_toolkit/blob/master/parascore/utils.py).
  We decided to train bart-large.

### Supervised Fine-Tuning on ETPC (SFT/ETPC)

- Llama-3.1-8B
  - We use the Llama-3.1-8B model finetuned on ETPC (SFT/ETPC) by [Wahle et al.](https://github.com/jpwahle/emnlp23-paraphrase-types).
- BART-large
  - We used code adapted from [Wahle et al.](https://github.com/jpwahle/emnlp23-paraphrase-types)

```python
python3 src/sft_ptg.py \
--model_name=facebook/bart-large \
--task_name=paraphrase-type-generation
```

### Reward modeling on APTY-ranked dataset (Reward/APTY)

- Llama-3.1-8B

```python
python3 src/reward.py
```

- We didn't continue to train the model SFT/ETPC with Reward/APTY using PPO to get the model RLHF/APTY, because of the low accuracy of the reward model.
  If you want, you can do so by finishing ppo.py and running:

```python
python3 src/ppo.py
```

### DPO optimization of SFT/ETPC on APTY-ranked dataset (DPO/APTY)

- Llama-3.1-8B
  - a table with conducted hyperparameter trials can be found [here](results/hyperparameters_dpo.csv)

```python
python3 src/dpo_llama_ptg.py \
--model_name=meta-llama/Llama-3.1-8B \
--adapter_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc \
--loss_type=sigmoid
```

- BART-large

```python
python3 src/dpo_ptg.py \
--model_name=cluebbers/bart-large-paraphrase-type-generation-etpc \
--task_name=paraphrase-type-generation \
--loss_type=sigmoid
```

### IPO optimization of SFT/ETPC on APTY-ranked dataset (IPO/APTY)

- Llama-3.1-8B
  - a table with conducted hyperparameter trials can be found [here](results/hyperparameters_ipo.csv)

```python
python3 src/dpo_llama_ptg.py \
--model_name=meta-llama/Llama-3.1-8B \
--adapter_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc \
--loss_type=ipo
```

- BART-large

```python
python3 src/dpo_ptg.py \
--model_name=cluebbers/bart-large-paraphrase-type-generation-etpc
--task_name=paraphrase-type-generation \
--loss_type=ipo
```

## Paraphrase Type Detection (PTD) Training

- Binary Classification on QQP dataset

  - We adapted code provided by [Wahle et al.](https://github.com/jpwahle/emnlp23-paraphrase-types).

```python
python3 src/sft_pd.py \
--model_name=microsoft/deberta-base
```

- Multilabel Classification on ETPC dataset

```python
python3 src/sft_ptd.py \
--model_name=cluebbers/deberta-base-paraphrase-detection-qqp
```

After training, a csv file with the evaluation results is created (for thesis: [hyperparameter results](results/deberta-base_qqp_pd_ptd_results_hyperclass_20241024_161522.csv)).

### Hyperparameter Tuning

If you want to reproduce the hyperparameter tuning, you need to uncomment that part in sft_ptd.py.
It will train with the best hyperparameters found and create a csv file with the best hyperparameters (for thesis: [hyperparameters](results/deberta-base_qqp_pd_hyperparameters_ptd_20241024_211334.csv)).
If you want to train with the found hyperparameters another time, you need to manually set the path to the newly created hyperparameter-file.

## Evaluation

Paraphrase Type Generation and ROUGE+BLEU evaluation of base model, SFT/ETPC, DPO/APTY, IPO/APTY:

- Llama-3.1-8B

```python
python3 src/eval_llama_ptg.py \
--model_name=meta-llama/Llama-3.1-8B \
--etpc_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc \
--dpo_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid \
--ipo_dir=cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo
```

- BART-large
  Same as above, but includeing ParaScore evaluation

```python
python3 src/eval_dpo_ptg.py \
--model_name=facebook/bart-large \
--etpc_dir=cluebbers/bart-large-paraphrase-type-generation-etpc \
--dpo_dir=cluebbers/bart-large-paraphrase-type-generation-apty-sigmoid \
--ipo_dir=cluebbers/bart-large-paraphrase-type-generation-apty-ipo
```

- For Open LLM Leaderboard evaluation, [submit your model](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/).
- Further evaluation is done in the jupyter notebook. All plots and tables from the project are generated there.

```python
evaluation.ipynb
```

## Pre-trained models

| Model        | Dataset     | Task | Link                                                                                                                                                |
| ------------ | ----------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama-3.1-8B |             |      | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)                                                                           |
| Llama-3.1-8B | ETPC        | PTG  | [SFT/ETPC](https://huggingface.co/cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc)                                                           |
| Llama-3.1-8B | ETPC + APTY | PTG  | [Reward/APTY](https://huggingface.co/cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc-apty-reward)                                            |
| Llama-3.1-8B | ETPC + APTY | PTG  | [DPO/APTY](https://huggingface.co/cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid)                                                   |
| Llama-3.1-8B | ETPC + APTY | PTG  | [IPO/APTY](https://huggingface.co/cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo)                                                       |
| DeBERTa-base |             |      | [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base)                                                                             |
| DeBERTa-base | QQP         | PD   | [cluebbers/deberta-base-paraphrase-detection-qqp](https://huggingface.co/cluebbers/deberta-base-paraphrase-detection-qqp)                           |
| DeBERTa-base | QQP + ETPC  | PTD  | [cluebbers/deberta-base-paraphrase-type-detection-etpc](https://huggingface.co/cluebbers/deberta-base-paraphrase-type-detection-etpc)               |
| BART-large   |             |      | [facebook/bart-large](https://huggingface.co/facebook/bart-large)                                                                                   |
| BART-large   | ETPC        | PTG  | [cluebbers/bart-large-paraphrase-type-generation-etpc](https://huggingface.co/cluebbers/bart-large-paraphrase-type-generation-etpc)                 |
| BART-large   | ETPC + APTY | PTG  | [cluebbers/bart-large-paraphrase-type-generation-apty-sigmoid](https://huggingface.co/cluebbers/bart-large-paraphrase-type-generation-apty-sigmoid) |
| BART-large   | ETPC + APTY | PTG  | [cluebbers/bart-large-paraphrase-type-generation-apty-ipo](https://huggingface.co/cluebbers/bart-large-paraphrase-type-generation-apty-ipo)         |

## Results

| Model        | Generated Paraphrases + automated metric scores                                                  | Annotation file                                                  |
| ------------ | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Llama-3.1-8B | [Llama-3.1-8B generated paraphrases](results/generated_paraphrases_Llama-3.1-8B_2024-11-15.json) | [project 6](results/project-6-at-2024-12-14-09-22-25077dd1.json) |
| Llama-2-7B   | [Llama-2-7B generated paraphrases](results/generated_paraphrases_Llama-2-7b-hf_2024-09-03.json)  | [project 5](results/project-5-at-2024-10-21-12-25-522ea966.json) |
| bart-large   | [bart-large generated paraphrases](results/generated_paraphrases_bart-large.json)                | None                                                             |

- Enhanced paraphrase-type generation accuracy:
  DPO training on APTY increases human-annotated accuracy by 3~\% over a supervised baseline, aligning outputs with nuanced linguistic transformations.
- Improved user-aligned quality:
  Human evaluators favor these improved outputs 7~\% more than baseline paraphrases, underscoring enhanced semantic fidelity and stylistic appropriateness.
  -A new human-ranked dataset:
  The dataset we produce enables a more rigorous, fine-grained evaluation of paraphrase quality and paves the way for future research.
- Exposing metric limitations:
  Weak correlations (Spearman's $r<0.3$) between automated metrics and human rankings motivate the development of richer evaluation frameworks.
- Improved paraphrase-type detection:
  Our PTD model achieves F1 scores of 0.91 on addition/deletion, 0.78 on same polarity substitution, and 0.70 for punctuation changes, enabling more granular assessments.
- Improved reasoning:
  PTG boosts multistep soft reasoning (MuSR) task performance by 38~\%, demonstrating broader benefits for language generation and reasoning tasks.

## Citation

Cite the paper:

```bibtex
@misc{lübbers2025enhancingparaphrasetypegeneration,
      title={Enhancing Paraphrase Type Generation: The Impact of DPO and RLHF Evaluated with Human-Ranked Data}, 
      author={Christopher Lee Lübbers},
      year={2025},
      eprint={2506.02018},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.02018}, 
}

```
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
