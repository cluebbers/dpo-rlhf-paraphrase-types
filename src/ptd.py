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
    "punctuation changes", "same polarity substitution (contextual)", "semantic-based",
    "spelling changes", "subordination and nesting changes", "synthetic/analytic substitution"
]

def write_results_to_csv(results, output_file="evaluation_results.csv"):
    """Writes evaluation results to a CSV file."""
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in results.items():
            if key != 'eval_detailed_report':
                writer.writerow([key, value])

        writer.writerow([])
        detailed_report = results.get("eval_detailed_report", None)
        if isinstance(detailed_report, dict):
            writer.writerow(["Label", "Precision", "Recall", "F1-Score", "Support"])
            for label, metrics in detailed_report.items():
                writer.writerow([
                    label,
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1-score", 0),
                    metrics.get("support", 0)
                ])
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

    num_labels = len(paraphrase_type2cls_id)
    labels = []
    for paraphrase_type_list in examples["paraphrase_types"]:
        binary_labels = [0] * num_labels
        for paraphrase_type in paraphrase_type_list:
            cls_id = paraphrase_type2cls_id.get(paraphrase_type.strip().lower())
            if cls_id is not None and cls_id < num_labels:
                binary_labels[cls_id] = 1
        labels.append(binary_labels)

    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
    return tokenized_inputs

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base", help="Name of the model to use")
    parser.add_argument("--max_samples_per_class", type=int, default=100, help="Maximum samples per class for downsampling")
    return parser.parse_args()

def compute_metrics(predictions, labels, cls_id2label):
    """Computes various metrics for evaluation."""
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probs = sigmoid(predictions)
    preds = (probs > 0.5).astype(int)

    top_10_types = [ptype for ptype in TOP_10_PARAPHRASE_TYPES]
    top_10_indices = [i for i, label in cls_id2label.items() if label in top_10_types]

    filtered_preds = preds[:, top_10_indices]
    filtered_labels = labels[:, top_10_indices]

    accuracy = accuracy_score(filtered_labels.flatten(), filtered_preds.flatten())
    precision = precision_score(filtered_labels.flatten(), filtered_preds.flatten(), average='macro', zero_division=0)
    recall = recall_score(filtered_labels.flatten(), filtered_preds.flatten(), average='macro', zero_division=0)
    f1 = f1_score(filtered_labels.flatten(), filtered_preds.flatten(), average='macro', zero_division=0)

    report = classification_report(
        filtered_labels, filtered_preds, target_names=[cls_id2label[i] for i in top_10_indices], output_dict=True
    )
    filtered_report = {label: metrics for label, metrics in report.items() if metrics.get("support", 0) > 0}

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'eval_detailed_report': filtered_report}

def split_dataset_by_type(
    dataset: Dataset, 
    train_percent: float = 0.8, 
    min_samples: int = 100, 
    max_samples: int = 400) -> Tuple[Dataset, Dataset, Dict[str, int], Dict[str, int]]:
    """
    Split dataset by paraphrase type while ensuring each paraphrase type has 
    at least `min_samples` and at most `max_samples` examples. Additionally, count the number of paraphrase types in the train and test sets.
    
    Args:
    - dataset: The dataset to split.
    - train_percent: The percentage of the dataset to use for training.
    - min_samples: Minimum number of examples per paraphrase type.
    - max_samples: Maximum number of examples per paraphrase type.
    
    Returns:
    - train_dataset: Training set.
    - test_dataset: Test set.
    - train_type_counts: Dictionary containing paraphrase type counts in the train set.
    - test_type_counts: Dictionary containing paraphrase type counts in the test set.
    """
    paraphrase_type_to_examples: Dict[str, List[Dict[str, Any]]] = {}

    for example in dataset:
        filtered_types = [ptype.strip().lower() for ptype in example["paraphrase_types"] if ptype in TOP_10_PARAPHRASE_TYPES]
        if filtered_types:
            example["paraphrase_types"] = filtered_types
            for ptype in filtered_types:
                paraphrase_type_to_examples.setdefault(ptype, []).append(example)

    train_examples: List[Dict[str, Any]] = []
    test_examples: List[Dict[str, Any]] = []
    train_type_counts: Dict[str, int] = {}
    test_type_counts: Dict[str, int] = {}

    for ptype, examples in paraphrase_type_to_examples.items():
        num_examples = len(examples)
        if num_examples < min_samples:
            continue
        
        examples = random.sample(examples, min(num_examples, max_samples))
        split_idx = int(train_percent * len(examples))
        train_subset = examples[:split_idx]
        test_subset = examples[split_idx:]

        train_examples.extend(train_subset)
        test_examples.extend(test_subset)

        # Count the occurrences of each paraphrase type
        train_type_counts[ptype] = len(train_subset)
        test_type_counts[ptype] = len(test_subset)

    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)

    return train_dataset, test_dataset, train_type_counts, test_type_counts

def count_paraphrase_types(dataset: Dataset) -> Dict[str, int]:
    """
    Counts the occurrences of each paraphrase type in the dataset.
    
    Args:
    - dataset: The dataset to analyze.
    
    Returns:
    - type_counts: A dictionary with paraphrase types as keys and their counts as values.
    """
    type_counts: Dict[str, int] = {}

    for example in dataset:
        for ptype in example["paraphrase_types"]:
            ptype = ptype.strip().lower()
            if ptype in TOP_10_PARAPHRASE_TYPES:
                type_counts[ptype] = type_counts.get(ptype, 0) + 1

    return type_counts


def main():
    args = parse_arguments()

    dataset = load_dataset("jpwahle/etpc").filter(lambda x: x["etpc_label"] == 1)["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, clean_up_tokenization_spaces=True)
    dataset = dataset.map(lambda ex: {**ex, "paraphrase_types": [ptype.strip().lower() for ptype in ex["paraphrase_types"]]})

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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    class_weights = torch.tensor(1.0 / (np.sum(np.array(dataset_tokenized["labels"], dtype=np.float32), axis=0) + np.finfo(np.float32).eps), dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    model.classifier.loss_fct = nn.BCEWithLogitsLoss(weight=class_weights)

    trainer = Trainer(
        model=model, 
        args=TrainingArguments(
            output_dir=f"./out/cls-models/{args.model_name.split('/')[-1]}_etpc_ptd", 
            per_device_train_batch_size=32, 
            fp16=True,
            num_train_epochs=3,
            ),
        train_dataset=train_dataset, 
        eval_dataset=test_dataset, 
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        compute_metrics=lambda p: compute_metrics(p.predictions, p.label_ids, cls_id2label)
    )

    trainer.train()
    results = trainer.evaluate()
    write_results_to_csv(results, output_file=f"out/cls-models/{args.model_name.split('/')[-1]}_results.csv")

if __name__ == "__main__":
    main()
