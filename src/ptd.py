import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import csv
import requests
import torch.nn as nn
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
import random

TOP_10_PARAPHRASE_TYPES = [
    "addition/deletion",
    "change of order",
    "derivational changes",
    "inflectional changes",
    "punctuation changes",
    "same polarity substitution (contextual)",
    "semantic-based",
    "spelling changes",
    "subordination and nesting changes",
    "synthetic/analytic substitution"
]

def write_results_to_csv(results, output_file="evaluation_results.csv"):
    """
    Writes evaluation results to a CSV file in a readable format.

    Args:
        results (dict): Dictionary of evaluation metrics, including the detailed report.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write overall metrics
        writer.writerow(["Metric", "Value"])
        for key, value in results.items():
            if key != 'eval_detailed_report':  # Ensure we skip detailed report for now
                writer.writerow([key, value])

        # Leave a line before detailed report
        writer.writerow([])

        # Check the datatype of eval_detailed_report before processing it
        detailed_report = results.get("eval_detailed_report", None)

        if isinstance(detailed_report, dict):  # Ensure it's a dictionary
            writer.writerow(["Label", "Precision", "Recall", "F1-Score", "Support"])  # Add headers for the detailed report
            for label, metrics in detailed_report.items():
                precision = metrics.get("precision", 0)
                recall = metrics.get("recall", 0)
                f1_score = metrics.get("f1-score", 0)
                support = metrics.get("support", 0)

                writer.writerow([label, precision, recall, f1_score, support])
        else:
            print(f"Warning: eval_detailed_report is of type {type(detailed_report)} and not written to CSV.")

        # Optional: Add a blank line after the detailed report for better formatting
        writer.writerow([])
    
    print("results written to " f"{output_file}")

    
def fetch_paraphrase_types():
    """
    Downloads and parses the paraphrase types XML from ETPC.
    """
    url = "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/paraphrase_types.xml"
    response = requests.get(url)
    root = ET.fromstring(response.text)

    paraphrase_types = [child.find("type_name").text.strip().lower() for child in root]
    type_ids = [int(child.find("type_id").text) for child in root]
    categories = [child.find("type_category").text for child in root]

    return paraphrase_types, type_ids, categories

def create_label_maps():
    """Creates label maps for ETPC paraphrase types."""
    # Fetch paraphrase types and IDs from ETPC
    paraphrase_types, paraphrase_type_ids, _ = fetch_paraphrase_types()

    # Only include the top 10 types
    paraphrase_types = [ptype for ptype in paraphrase_types if ptype in TOP_10_PARAPHRASE_TYPES]

    # Create label maps
    label2cls_id = {label: idx for idx, label in enumerate(paraphrase_types, start=0)}
    cls_id2label = {idx: label for label, idx in label2cls_id.items()}

    # Create additional maps for paraphrase type IDs
    paraphrase_type2cls_id = {ptype: idx for ptype, idx in zip(paraphrase_types, paraphrase_type_ids)}
    paraphrase_id2cls_type = {idx: ptype for ptype, idx in paraphrase_type2cls_id.items()}

    # Create a map from class IDs to paraphrase type IDs
    cls_id2paraphrase_type_id = {i: paraphrase_type2cls_id[cls_id2label[i]] for i in cls_id2label}
    paraphrase_type_id2cls_id = {v: k for k, v in cls_id2paraphrase_type_id.items()}

    return (
        label2cls_id,
        cls_id2label,
        paraphrase_type2cls_id,
        paraphrase_id2cls_type,
        cls_id2paraphrase_type_id,
        paraphrase_type_id2cls_id,
    )


def tokenize_examples(
    examples: Dict[str, List[str]],
    sentence1_key: str,
    tokenizer: PreTrainedTokenizerBase,
    paraphrase_type2cls_id : Dict[str, int],  # <-- Add this argument
    sentence2_key: Optional[str] = None,      
) -> Dict[str, Union[List[int], torch.Tensor]]:
    """
    Tokenizes examples for use in the model.

    Args:
        examples (Dict[str, List[str]]): The examples to tokenize.
        sentence1_key (str): The key for the first sentence.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        paraphrase_type2cls_id  (Dict[str, int]): The mapping of paraphrase types to class IDs.
        sentence2_key (Optional[str], optional): The key for the second sentence if applicable. Defaults to None.

    Returns:
        Dict[str, Union[List[int], torch.Tensor]]: The tokenized inputs.
    """
    max_length = 256
    tokenized_inputs = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key] if sentence2_key else None,
        truncation=True,
        max_length=max_length
    )

    num_labels = len(paraphrase_type2cls_id )
    standardized_type_to_cls_id = {k.strip().lower(): v for k, v in paraphrase_type2cls_id .items()}

    labels = []
    for paraphrase_type_list in examples["paraphrase_types"]:
        binary_labels = [0] * num_labels
        for paraphrase_type in paraphrase_type_list:
            cls_id = standardized_type_to_cls_id.get(paraphrase_type.strip().lower())
            if cls_id is not None and cls_id < num_labels:
                binary_labels[cls_id] = 1
        labels.append(binary_labels)

    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
    return tokenized_inputs


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for running paraphrase type classification experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-base",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=100,
        help="Maximum samples per class for downsampling",
    )
    args = parser.parse_args()
    return args

def compute_metrics(predictions, labels, cls_id2label):
    """
    Compute metrics for the top 10 paraphrase types only, and remove non-top-10 types from the report.
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probs = sigmoid(predictions)
    preds = (probs > 0.5).astype(int)

    # Convert cls_id2label into a list for top 10 filtering
    top_10_types = [ptype for ptype in TOP_10_PARAPHRASE_TYPES]

    # Get the indices of the top 10 paraphrase types
    top_10_indices = [i for i, label in cls_id2label.items() if label in top_10_types]

    # Filter out non-top-10 predictions and labels
    filtered_preds = preds[:, top_10_indices]
    filtered_labels = labels[:, top_10_indices]

    # Flatten for evaluation metrics
    filtered_preds_flat = filtered_preds.flatten()
    filtered_labels_flat = filtered_labels.flatten()

    # Calculate metrics
    accuracy = accuracy_score(filtered_labels_flat, filtered_preds_flat)
    precision = precision_score(filtered_labels_flat, filtered_preds_flat, average='macro', zero_division=0)
    recall = recall_score(filtered_labels_flat, filtered_preds_flat, average='macro', zero_division=0)
    f1 = f1_score(filtered_labels_flat, filtered_preds_flat, average='macro', zero_division=0)

    # Only include detailed report for top 10 types
    top_10_target_names = [cls_id2label[i] for i in top_10_indices]
    report = classification_report(
        filtered_labels, filtered_preds, target_names=top_10_target_names, labels=np.arange(len(top_10_target_names)), output_dict=True
    )

    # Filter out types with support == 0 in the detailed report
    filtered_report = {label: metrics for label, metrics in report.items() if metrics.get("support", 0) > 0}

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detailed_report': filtered_report  # Ensure detailed_report is included
    }




def split_dataset_by_type(dataset: Dataset, train_percent: float = 0.8, min_samples: int = 100, max_samples: int = 200) -> Tuple[Dataset, Dataset]:
    """
    Split dataset by paraphrase type while ensuring each paraphrase type has 
    at least `min_samples` and at most `max_samples` examples.
    
    Args:
    - dataset: The dataset to split.
    - train_percent: The percentage of the dataset to use for training.
    - min_samples: Minimum number of examples per paraphrase type.
    - max_samples: Maximum number of examples per paraphrase type.
    
    Returns:
    - train_dataset: Training set.
    - test_dataset: Test set.
    """
    paraphrase_type_to_examples: Dict[str, List[Dict[str, Any]]] = {}

    for example in dataset:
        paraphrase_types = example["paraphrase_types"]

        filtered_types = [ptype.strip().lower() for ptype in paraphrase_types if ptype in TOP_10_PARAPHRASE_TYPES]

        if not filtered_types:
            continue  # Skip this example if it has no matching paraphrase types

        example["paraphrase_types"] = filtered_types

        for ptype in filtered_types:
            if ptype not in paraphrase_type_to_examples:
                paraphrase_type_to_examples[ptype] = []
            paraphrase_type_to_examples[ptype].append(example)

    train_examples: List[Dict[str, Any]] = []
    test_examples: List[Dict[str, Any]] = []

    for ptype, examples in paraphrase_type_to_examples.items():
        num_examples = len(examples)

        if num_examples < min_samples:
            continue  # Skip this paraphrase type with too few examples

        if num_examples > max_samples:
            examples = random.sample(examples, max_samples)
            num_examples = max_samples

        split_idx = int(train_percent * num_examples)
        train_examples.extend(examples[:split_idx])
        test_examples.extend(examples[split_idx:])

    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)

    return train_dataset, test_dataset

def downsample_overrepresented_classes(
    dataset: Dataset, label_column: str, max_samples_per_class: int
) -> Dataset:
    """
    Downsample overrepresented classes in the dataset.
    
    Args:
    - dataset (Dataset): The dataset to downsample.
    - label_column (str): The name of the column containing class labels.
    - max_samples_per_class (int): Maximum number of samples per class after downsampling.
    
    Returns:
    - downsampled_dataset (Dataset): The downsampled dataset.
    """
    class_groups = {label: [] for label in set(dataset[label_column].apply(lambda x: x.strip().lower()))}
    for example in dataset:
        for label in example[label_column]:
            class_groups[label].append(example)

    downsampled_data = []
    for examples in class_groups.values():
        if len(examples) > max_samples_per_class:
            downsampled_data.extend(random.sample(examples, max_samples_per_class))
        else:
            downsampled_data.extend(examples)

    return Dataset.from_list(downsampled_data)

def main():
    args = parse_arguments()

    # Load dataset and tokenizer
    dataset = load_dataset("jpwahle/etpc").filter(
        lambda x: x["etpc_label"] == 1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True, clean_up_tokenization_spaces=True)
    sentence1_key, sentence2_key = "sentence1", "sentence2"
    dataset = dataset["train"]

    # Convert all paraphrase types in the dataset to lowercase for consistency
    dataset = dataset.map(lambda examples: {**examples, "paraphrase_types": [ptype.strip().lower() for ptype in examples["paraphrase_types"]]})

    # Create label maps for the paraphrase types
    label_maps = create_label_maps()
    (
        label2cls_id,
        cls_id2label,
        paraphrase_type2cls_id,
        paraphrase_id2cls_type,
        cls_id2paraphrase_type_id,
        paraphrase_type_id2cls_id,
    ) = label_maps

    # Tokenize the dataset and prepare for training
    dataset_tokenized = dataset.map(
        tokenize_examples,
        batched=True,
        fn_kwargs={
            "sentence1_key": sentence1_key,
            "sentence2_key": sentence2_key,
            "tokenizer": tokenizer,
            "paraphrase_type2cls_id": paraphrase_type2cls_id,
        },
    )

    # Split the dataset into training and testing sets with min and max constraints
    train_dataset, test_dataset = split_dataset_by_type(dataset_tokenized, train_percent=0.8, min_samples=100, max_samples=200)

    # Define model configuration with multi-label classification
    num_labels = len(paraphrase_type2cls_id)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    # Load pretrained model and modify it for weighted loss
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Calculate class weights to handle class imbalance
    label_counts = np.sum(np.array(dataset_tokenized["labels"], dtype=np.float32), axis=0)
    class_weights = 1.0 / (label_counts + np.finfo(np.float32).eps)  # Avoid division by zero
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

    # Modify the model to include the custom weighted loss function
    model.classifier.loss_fct = nn.BCEWithLogitsLoss(weight=class_weights)

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data collator and training arguments
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{args.model_name.replace('/', '_')}_etpc_seq-cls",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        fp16=True,
    )

    # Create Trainer object for model training and evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p.predictions, p.label_ids, cls_id2label),
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()

    # Write results to CSV
    write_results_to_csv(results, output_file=f"out/eval_{args.model_name.replace('/', '_')}_etpc_seq-cls_results.csv")

    # Pop eval_detailed_report for printing
    detailed_report = results.pop('eval_detailed_report', None)
    
    if detailed_report:
        print("Detailed report available:")
        for label, metrics in detailed_report.items():
            print(f"Label: {label}")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    else:
        print("No detailed report found.")
    
    print("Results written to " f"out/eval_{args.model_name.replace('/', '_')}_etpc_seq-cls_results.csv")


if __name__ == "__main__":
    main()
