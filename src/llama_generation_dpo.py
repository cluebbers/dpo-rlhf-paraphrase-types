# evaluation
import argparse
import json
import os
import time
from typing import List

import fire
import numpy as np
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel, PeftConfig
import torch

# Install required libraries first
# pip install rouge nltk sacrebleu

def load_model_and_tokenizer(model_name_or_path, adapter_dir=None, max_seq_length =2048):
    """
    Load a model and tokenizer, applying PEFT adapters if specified.

    Args:
        model_name_or_path (str): The name or path of the model to load.
        adapter_dir (str, optional): The path to the PEFT adapter to apply. Defaults to None.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        load_in_4bit=True,
        dtype=torch.float16,
        max_seq_length =max_seq_length ,
    )

    if adapter_dir:
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = FastLanguageModel.get_peft_model(model, **vars(peft_config))

    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

# Load data
def load_data(filename):
    """Loads data from a file in JSON format.

    Args:
        filename (str): The path to the file to load.

    Returns:
        list: The loaded data as a list of dictionaries.

    Example:
        ```python
        filename = "data.json"
        data = load_data(filename)
        print(data)
        ```
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def generate_paraphrases(
    data, model, tokenizer, max_length, temperature, top_p, max_batch_size, num_examples=3
):
    paraphrases = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, num_examples, max_batch_size)):
        batch = data[i : i + max_batch_size]
        user_messages = [instance["messages"][0]["content"] for instance in batch]
        
        inputs = tokenizer(user_messages, return_tensors="pt", padding=True, truncation=True).to(device)

        results = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )

        # Decode generated outputs into text
        paraphrases.extend([tokenizer.decode(result, skip_special_tokens=True) for result in results])

    return paraphrases


def evaluate(paraphrases, references):
    """
    Evaluates the quality of paraphrases compared to reference texts.

    Args:
        paraphrases (list): The generated paraphrases.
        references (list): The reference texts.

    Returns:
        dict: A dictionary containing the evaluation scores.

    Example:
        ```python
        paraphrases = ["Paraphrase 1", "Paraphrase 2"]
        references = ["Reference 1", "Reference 2"]

        scores = evaluate(paraphrases, references)
        print(scores)
        ```
    """

    rouge = Rouge()

    # ROUGE scores
    rouge_scores = rouge.get_scores(paraphrases, references, avg=True)

    # BLEU scores
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], paraphrase, smoothing_function=smoothie)
        for ref, paraphrase in zip(references, paraphrases)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-2": rouge_scores["rouge-2"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],
        "BLEU": avg_bleu,
    }


def main(
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    adapter_path: str = "out/dpo_llama-7b_apty",
    data_file: str = "out/generation_etpc_test.jsonl",
    num_examples: int = 100, # 1000
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_length : int = 2048,
    max_length: int = 1024,
    max_batch_size: int = 4, 
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, adapter_path,
        max_seq_length =max_seq_length ,
    )

    # Load data and predict
    test_data = load_data(data_file)
    
    generated_paraphrases = generate_paraphrases(
        test_data,
        model,
        tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        max_batch_size=max_batch_size,
        num_examples=num_examples,
    )
    references = [item["messages"][1]["content"] for item in test_data[:num_examples]]
    scores = evaluate(generated_paraphrases, references)
    print(f"Model: {model_name_or_path}")
    print(scores)


if __name__ == "__main__":
    fire.Fire(main)