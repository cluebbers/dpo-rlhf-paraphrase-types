import argparse
import os
import csv
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import requests
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

TOP_10_PARAPHRASE_TYPES = [
    "addition/deletion", "change of order", "derivational changes", "inflectional changes",
    "punctuation changes", "same polarity substitution (contextual)", "semantic based",
    "spelling changes", "subordination and nesting changes", "synthetic/analytic substitution"
]

def write_results_to_csv(results, output_file="evaluation_results.csv"):
    """Writes evaluation results to a CSV file."""
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in results.items():
            writer.writerow([key, value])

        print(f"Results written to {output_file}")

def fetch_paraphrase_types():
    """Fetches paraphrase types from an external XML source."""
    url = "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/paraphrase_types.xml"
    response = requests.get(url)
    root = ET.fromstring(response.text)
    return ([child.find("type_name").text.strip().lower() for child in root],
            [int(child.find("type_id").text) for child in root],
            [child.find("type_category").text for child in root])

def create_label_maps():
    """Creates label maps for ETPC paraphrase types."""
    paraphrase_types, paraphrase_type_ids, _ = fetch_paraphrase_types()
    paraphrase_types = [ptype for ptype in paraphrase_types if ptype in TOP_10_PARAPHRASE_TYPES]
    
    label2cls_id = {label: idx for idx, label in enumerate(paraphrase_types)}
    cls_id2label = {idx: label for label, idx in label2cls_id.items()}
    paraphrase_type2cls_id = {ptype: idx for ptype, idx in zip(paraphrase_types, paraphrase_type_ids)}
    
    cls_id2paraphrase_type_id = {i: paraphrase_type2cls_id[cls_id2label[i]] for i in cls_id2label}
    paraphrase_type_id2cls_id = {v: k for k, v in cls_id2paraphrase_type_id.items()}
    
    return label2cls_id, cls_id2label, paraphrase_type2cls_id, cls_id2paraphrase_type_id, paraphrase_type_id2cls_id

def tokenize_examples(examples: Dict[str, List[str]], sentence1_key: str, tokenizer: PreTrainedTokenizerBase, paraphrase_type2cls_id: Dict[str, int], sentence2_key: Optional[str] = None):
    """Tokenizes input sentences and generates corresponding labels."""
    tokenized_inputs = tokenizer(
        examples[sentence1_key], examples[sentence2_key] if sentence2_key else None, truncation=True, max_length=256
    )

    # The number of labels should correspond to the number of top 10 paraphrase types
    num_labels = len(TOP_10_PARAPHRASE_TYPES)
    labels = []
    for paraphrase_type_list in examples["paraphrase_types"]:
        binary_labels = [0] * num_labels
        for paraphrase_type in paraphrase_type_list:
            paraphrase_type_clean = paraphrase_type.strip().lower()
            # Only assign labels to top 10 paraphrase types
            if paraphrase_type_clean in TOP_10_PARAPHRASE_TYPES:
                cls_id = paraphrase_type2cls_id.get(paraphrase_type_clean)
                if cls_id is not None and cls_id < num_labels:
                    binary_labels[cls_id] = 1
        labels.append(binary_labels)

    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)

    # Debugging: print just a small sample (1% of examples)
    if random.random() < 0.01:
        print(f"Sample paraphrase types: {examples['paraphrase_types'][0]}")
        print(f"Corresponding binary labels: {labels[0]}")

    return tokenized_inputs

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="out/cls-models/deberta-base_qqp_pd", help="Name of the model to use")
    parser.add_argument("--max_samples_per_class", type=int, default=100, help="Maximum samples per class for downsampling")
    return parser.parse_args()

def compute_metrics(predictions, labels, cls_id2label):
    """Computes various metrics for multi-label evaluation."""
    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # Convert logits to probabilities
    probs = sigmoid(predictions)
    
    # Threshold probabilities to get binary predictions (multi-label classification)
    preds = (probs > 0.5).astype(int)

    top_10_types = [ptype for ptype in TOP_10_PARAPHRASE_TYPES]
    top_10_indices = [i for i, label in cls_id2label.items() if label in top_10_types]

    filtered_preds = preds[:, top_10_indices]
    filtered_labels = labels[:, top_10_indices]

    # Debugging: Ensure predictions and labels shape alignment
    print(f"Filtered Predictions shape: {filtered_preds.shape}")
    print(f"Filtered Labels shape: {filtered_labels.shape}")

    # Fix: Get the index for 'inflectional changes' paraphrase type
    inflectional_idx = list(cls_id2label.values()).index("inflectional changes")
    
    # Print raw probabilities and predictions for 'inflectional changes'
    print(f"Raw logits for 'inflectional changes': {predictions[:, inflectional_idx]}")
    print(f"Predictions for 'inflectional changes': {filtered_preds[:, inflectional_idx]}")
    
    # Compute multi-label metrics (macro averages)
    accuracy = accuracy_score(filtered_labels.flatten(), filtered_preds.flatten())
    precision = precision_score(filtered_labels.flatten(), filtered_preds.flatten(), average='macro', zero_division=0)
    recall = recall_score(filtered_labels.flatten(), filtered_preds.flatten(), average='macro', zero_division=0)
    f1 = f1_score(filtered_labels.flatten(), filtered_preds.flatten(), average='macro', zero_division=0)

    # Generate detailed classification report for individual paraphrase types
    report = classification_report(
        filtered_labels, filtered_preds, target_names=[cls_id2label[i] for i in top_10_indices], output_dict=True
    )

    # Debug: Print out the support counts
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{label}: support = {metrics['support']}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def split_dataset_by_type(
    dataset: Dataset, 
    train_percent: float = 0.8, 
    min_samples: int = 100, 
    max_samples: int = 400
) -> Tuple[Dataset, Dataset, Dict[str, int], Dict[str, int]]:
    """
    Split dataset by paraphrase type while ensuring each paraphrase type has 
    at least `min_samples` and at most `max_samples` examples. Filter out sentences with paraphrase types 
    not in the TOP 10 list.
    """
    paraphrase_type_to_examples: Dict[str, List[Dict[str, Any]]] = {}

    # Normalize TOP_10_PARAPHRASE_TYPES to lowercase for consistent comparison
    normalized_top_10_types = [ptype.strip().lower() for ptype in TOP_10_PARAPHRASE_TYPES]

    # Step 2: Filter the dataset
    for example in dataset:
        if "paraphrase_types" not in example:
            continue

        # Normalize the paraphrase types in the dataset to lowercase and strip spaces
        filtered_types = [ptype.strip().lower() for ptype in example["paraphrase_types"] if ptype.strip().lower() in normalized_top_10_types]
        
        # If no paraphrase type from TOP_10_PARAPHRASE_TYPES, skip this sample
        if not filtered_types:
            continue

        example["paraphrase_types"] = filtered_types
        for ptype in filtered_types:
            paraphrase_type_to_examples.setdefault(ptype, []).append(example)

    # Check if any examples remain after filtering
    if not paraphrase_type_to_examples:
        print("No examples remain after filtering. Please check the dataset structure and filtering logic.")
        return Dataset.from_list([]), Dataset.from_list([]), {}, {}

    train_examples: List[Dict[str, Any]] = []
    test_examples: List[Dict[str, Any]] = []
    train_type_counts: Dict[str, int] = {}
    test_type_counts: Dict[str, int] = {}

    # Step 3: Downsampling and splitting into train/test
    for ptype, examples in paraphrase_type_to_examples.items():
        num_examples = len(examples)
        print(f"Type {ptype} has {num_examples} examples")  # Debugging output
        
        if num_examples < min_samples:
            print(f"Skipping {ptype} due to insufficient samples ({num_examples} < {min_samples})")
            continue  # Skip types with fewer than the minimum required samples
        
        examples = random.sample(examples, min(num_examples, max_samples))
        split_idx = int(train_percent * len(examples))
        train_subset = examples[:split_idx]
        test_subset = examples[split_idx:]

        train_examples.extend(train_subset)
        test_examples.extend(test_subset)

        # Count the occurrences of each paraphrase type
        train_type_counts[ptype] = len(train_subset)
        test_type_counts[ptype] = len(test_subset)

    # Debug: Print final train and test counts
    print(f"Train counts: {train_type_counts}")
    print(f"Test counts: {test_type_counts}")

    # Create train and test datasets
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)

    # Return datasets and counts
    return train_dataset, test_dataset, train_type_counts, test_type_counts

def count_paraphrase_types(dataset: Dataset) -> Dict[str, int]:
    """
    Counts the occurrences of each paraphrase type in the dataset.
    """
    type_counts: Dict[str, int] = {}

    for example in dataset:
        # Check if "paraphrase_types" field exists and is a list
        if "paraphrase_types" not in example or not isinstance(example["paraphrase_types"], list):
            print(f"Skipping example due to missing paraphrase_types: {example}")
            continue

        for ptype in example["paraphrase_types"]:
            ptype = ptype.strip().lower()
            if ptype in TOP_10_PARAPHRASE_TYPES:
                type_counts[ptype] = type_counts.get(ptype, 0) + 1

    # Debug: Print the resulting paraphrase type counts
    print(f"Paraphrase type counts: {type_counts}")

    return type_counts

def hyperparameter_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
    }

def main():
    args = parse_arguments()

    dataset = load_dataset("jpwahle/etpc").filter(lambda x: x["etpc_label"] == 1)["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, clean_up_tokenization_spaces=True)

    # Count paraphrase types before splitting
    pre_split_type_counts = count_paraphrase_types(dataset)
    print("Number of paraphrase types before splitting:")
    for ptype, count in pre_split_type_counts.items():
        print(f"{ptype}: {count}")
        
    label_maps = create_label_maps()
    _, cls_id2label, paraphrase_type2cls_id, _, _ = label_maps

    dataset_tokenized = dataset.map(
        tokenize_examples, batched=True, fn_kwargs={"sentence1_key": "sentence1", "sentence2_key": "sentence2", "tokenizer": tokenizer, "paraphrase_type2cls_id": paraphrase_type2cls_id}
    )

    # Split the dataset into training and testing sets with type counts
    train_dataset, test_dataset, train_type_counts, test_type_counts = split_dataset_by_type(dataset_tokenized, train_percent=0.8, min_samples=100, max_samples=400)

    # Print out the number of paraphrase types in train and test sets
    print("Number of paraphrase types in the training set:")
    for ptype, count in train_type_counts.items():
        print(f"{ptype}: {count}")
    
    print("\nNumber of paraphrase types in the test set:")
    for ptype, count in test_type_counts.items():
        print(f"{ptype}: {count}")
        
    num_labels = len(paraphrase_type2cls_id)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels, problem_type="multi_label_classification")
    
    def model_init(trial=None):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, config=config, ignore_mismatched_sizes=True
        )
        return model

    trainer = Trainer(
        model_init=lambda trial: model_init(trial),  # Pass the trial object to model_init
        args=TrainingArguments(
            output_dir=f"./out/cls-models/{args.model_name.split('/')[-1]}_etpc_ptd", 
            per_device_train_batch_size=32, 
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            fp16=True,
            num_train_epochs=5,
            metric_for_best_model='f1',
            load_best_model_at_end=True,
            greater_is_better=True,
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        compute_metrics=lambda p: compute_metrics(p.predictions, p.label_ids, cls_id2label)
    )

    # Perform the hyperparameter search using Optuna
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hyperparameter_space,
        n_trials=20,
        backend="optuna"
    )

    # Apply the best hyperparameters found by the search
    trainer.args.learning_rate = best_run.hyperparameters['learning_rate']
    trainer.args.weight_decay = best_run.hyperparameters['weight_decay']
    trainer.args.per_device_train_batch_size = best_run.hyperparameters['per_device_train_batch_size']

    # Continue training with the best hyperparameters
    trainer.train()

    # Evaluate and write results to CSV
    results = trainer.evaluate()
    write_results_to_csv(results, output_file=f"out/cls-models/{args.model_name.split('/')[-1]}_ptd_results_hyperclass.csv")

if __name__ == "__main__":
    main()
