# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse
import json
import os
import time
from typing import List

import fire
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm import tqdm

from unsloth import FastLanguageModel
from peft import PeftModel, PeftConfig
import torch

ALL_TYPES = {
    "Derivational Changes",
    "Inflectional Changes",
    "Modal Verb Changes",
    "Spelling changes",
    "Change of format",
    "Same Polarity Substitution (contextual)",
    "Same Polarity Substitution (habitual)",
    "Same Polarity Substitution (named ent.)",
    "Converse substitution",
    "Opposite polarity substitution (contextual)",
    "Opposite polarity substitution (habitual)",
    "Synthetic/analytic substitution",
    "Coordination changes",
    "Diathesis alternation",
    "Ellipsis",
    "Negation switching",
    "Subordination and nesting changes",
    "Direct/indirect style alternations",
    "Punctuation changes",
    "Syntax/discourse structure changes",
    "Entailment",
    "Identity",
    "Non-paraphrase",
    "Addition/Deletion",
    "Change of order",
    "Semantic-based",
}

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


def classify(
    data, model, tokenizer, max_length, temperature, top_p, max_batch_size, num_examples=100
):
    """Classifies the data using the provided model and parameters. Now forwards max_batch_size prompts into the model instead of one.

    Args:
        data (list): The data to classify.
        model (object): The model to use for classification.
        max_length (int): The maximum generation length.
        temperature (float): The temperature parameter for the model.
        top_p (float): The top_p parameter for the model.
        max_batch_size (int): The maximum batch size for the model.
        num_examples (int, optional): The number of examples to classify. Defaults to 100.

    Returns:
        tuple: The true and predicted labels.
    """
    y_true = []
    y_pred = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, num_examples, max_batch_size)):
        batch = data[i : i + max_batch_size]
        user_messages = [instance["messages"][0]["content"] for instance in batch]
        inputs = tokenizer(user_messages, return_tensors="pt", padding=True, truncation=True).to(device)
        true_response_labels = [
            set(instance["messages"][1]["content"].split(", ")) for instance in batch
        ]

        # Call the API and retry if it fails

        results = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Check if results have any elements
        if results.numel() > 0:

            # Decode the generated outputs into text before processing
            predicted_response_labels = [
                set(tokenizer.decode(result, skip_special_tokens=True).split(", ")) for result in results
            ]

            y_true.extend(true_response_labels)
            y_pred.extend(predicted_response_labels)

    return y_true, y_pred


def evaluate(y_true, y_pred):
    """Evaluates the performance of a classification model.

    Args:
        y_true (list): The true labels.
        y_pred (list): The predicted labels.

    Returns:
        tuple: A tuple containing the F1 score and accuracy.

    Example:
        ```python
        y_true = [[1, 0, 1], [0, 1, 0]]
        y_pred = [[1, 1, 0], [0, 1, 1]]

        acc = evaluate(y_true, y_pred)
        print(acc)
        ```
    """

    y_true_bin = [[1 if t in labels else 0 for t in ALL_TYPES] for labels in y_true]
    y_pred_bin = [[1 if t in labels else 0 for t in ALL_TYPES] for labels in y_pred]

    # Convert lists to numpy arrays for easier calculations
    y_true_np = np.array(y_true_bin)
    y_pred_np = np.array(y_pred_bin)

    # Calculate per-class accuracy
    acc = np.mean(np.equal(y_true_np, y_pred_np).astype(int))

    return acc


def main(
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    adapter_path: str = "out/dpo_llama-7b_apty",
    data_file: str = "out/detection_etpc_test.jsonl",
    num_examples: int = 8, # 1000
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_length: int = 2048,
    max_length: int = 1024,
    max_batch_size: int = 4,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, adapter_path,
        max_seq_length =max_seq_length ,
    )

    # Load data and predict
    test_data = load_data(data_file)
    
    y_true, y_pred = classify(
        test_data,
        model,
        tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        max_batch_size=max_batch_size,
        num_examples=num_examples,
    )
    acc = evaluate(y_true, y_pred)

    print(f"Model: {model_name_or_path}")
    print(f"Eval set size: {len(y_pred)}")
    print(f"Accuracy: {acc:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
