import argparse
import os
import csv
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import torch
import pandas as pd
import torch.nn as nn
import requests
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import defaultdict
from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    EarlyStoppingCallback,
)

TOP_10_PARAPHRASE_TYPES = [
    "addition/deletion", "change of order", "derivational changes", "inflectional changes",
    "punctuation changes", "same polarity substitution (contextual)", "semantic based",
    "spelling changes", "subordination and nesting changes", "synthetic/analytic substitution"
]

# Manually assign IDs to each of the top-10 paraphrase types
TOP_10_PARAPHRASE_TYPE_TO_ID = {
    "addition/deletion": 0,
    "change of order": 1,
    "derivational changes": 2,
    "inflectional changes": 3,
    "punctuation changes": 4,
    "same polarity substitution (contextual)": 5,
    "semantic based": 6,
    "spelling changes": 7,
    "subordination and nesting changes": 8,
    "synthetic/analytic substitution": 9
}

# Reverse mapping for evaluation purposes
ID_TO_TOP_10_PARAPHRASE_TYPE = {v: k for k, v in TOP_10_PARAPHRASE_TYPE_TO_ID.items()}


# Assuming `detailed_classification_report` is globally available
detailed_classification_report = {}  # Placeholder for the actual report

def write_results_to_csv(results, output_file="evaluation_results.csv"):
    """Writes evaluation results to a CSV file, including the classification report."""
    global detailed_classification_report  # Access the global variable

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])

        # Write standard metrics (excluding 'classification_report')
        for key, value in results.items():
            if key != "classification_report":
                writer.writerow([key, value])

        writer.writerow([])  # Add an empty row for readability
        writer.writerow(["Classification Report"])
        writer.writerow(["Paraphrase Type", "Precision", "Recall", "F1-score", "Support"])

        # Iterate over each label and write the metrics from the global report
        for label, metrics in detailed_classification_report.items():
            if isinstance(metrics, dict):
                writer.writerow([
                    label,
                    metrics.get('precision', 'N/A'),
                    metrics.get('recall', 'N/A'),
                    metrics.get('f1-score', 'N/A'),
                    metrics.get('support', 'N/A')
                ])
            else:
                # Handle average metrics like 'micro avg', 'macro avg', etc.
                writer.writerow([label, metrics])

    print(f"Results written to {output_file}")

def tokenize_examples(examples: Dict[str, List[str]], sentence1_key: str, tokenizer: PreTrainedTokenizerBase, sentence2_key: Optional[str] = None):
    """Tokenizes input sentences and generates corresponding labels."""
    tokenized_inputs = tokenizer(
        examples[sentence1_key], examples[sentence2_key] if sentence2_key else None, truncation=True, max_length=256
    )

    # The number of labels should correspond to the number of top-10 paraphrase types
    num_labels = len(TOP_10_PARAPHRASE_TYPE_TO_ID)
    labels = []
    filtered_paraphrase_types = []  # Store filtered paraphrase types

    # Process each example's paraphrase types
    for paraphrase_type_list in examples["paraphrase_types"]:
        binary_labels = [0] * num_labels
        filtered_types = []
        for paraphrase_type in paraphrase_type_list:
            paraphrase_type_clean = paraphrase_type.strip().lower()
            # Map directly using the fixed dictionary
            cls_id = TOP_10_PARAPHRASE_TYPE_TO_ID.get(paraphrase_type_clean)
            if cls_id is not None:
                binary_labels[cls_id] = 1
                filtered_types.append(paraphrase_type_clean)

        labels.append(binary_labels)
        filtered_paraphrase_types.append(filtered_types)  # Track only the types in the top-10

    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
    tokenized_inputs["paraphrase_types"] = filtered_paraphrase_types  # Update with filtered types

    # Debugging: print just a small sample (1% of examples)
    if random.random() < 0.01:
        print(f"Sample paraphrase types: {filtered_paraphrase_types[0]}")
        print(f"Corresponding binary labels: {labels[0]}")

    return tokenized_inputs




def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="out/cls-models/deberta-base_qqp_pd", help="Name of the model to use")
    return parser.parse_args()

def compute_metrics(predictions, labels):
    """Computes various metrics for multi-label evaluation."""
    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # Convert logits to probabilities
    probs = sigmoid(predictions)
    
    # Threshold probabilities to get binary predictions (multi-label classification)
    preds = (probs > 0.5).astype(int)

    # Map indices back to their respective types for metrics calculation
    target_names = [ID_TO_TOP_10_PARAPHRASE_TYPE[i] for i in range(len(TOP_10_PARAPHRASE_TYPE_TO_ID))]

    # Compute metrics
    accuracy = accuracy_score(labels.flatten(), preds.flatten())
    precision = precision_score(labels.flatten(), preds.flatten(), average='macro', zero_division=0)
    recall = recall_score(labels.flatten(), preds.flatten(), average='macro', zero_division=0)
    f1 = f1_score(labels.flatten(), preds.flatten(), average='macro', zero_division=0)

    # Generate detailed classification report for individual paraphrase types
    report = classification_report(
        labels, preds, target_names=target_names, output_dict=True
    )
    macro_f1_from_report = report['macro avg']['f1-score']

    global detailed_classification_report
    detailed_classification_report = report

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "macro-f1": macro_f1_from_report,
    }

def split_dataset_by_type(
    dataset: Dataset, 
    train_percent: float = 0.8
) -> Tuple[Dataset, Dataset, Dict[str, int], Dict[str, int]]:
    """
    Split dataset by paraphrase type, ensuring a balanced distribution of paraphrase types 
    in both train and test datasets. Each example can belong to multiple paraphrase types.
    """
    paraphrase_type_to_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    type_counts: Dict[str, int] = defaultdict(int)
    
    # Normalize TOP_10_PARAPHRASE_TYPES to lowercase for consistent comparison
    normalized_top_10_types = [ptype.strip().lower() for ptype in TOP_10_PARAPHRASE_TYPES]

    # Step 1: Filter the dataset and group examples by their paraphrase types
    filtered_examples = []
    for example in dataset:
        if "paraphrase_types" not in example:
            continue

        # Normalize and filter paraphrase types
        filtered_types = [ptype.strip().lower() for ptype in example["paraphrase_types"] if ptype.strip().lower() in normalized_top_10_types]
        
        if not filtered_types:
            continue
        
        example["paraphrase_types"] = filtered_types
        filtered_examples.append(example)
        
        # Track examples for each paraphrase type
        for ptype in filtered_types:
            paraphrase_type_to_examples[ptype].append(example)
            type_counts[ptype] += 1

    train_examples = []
    test_examples = defaultdict(list)

    # Step 2: Balanced splitting into train and test sets
    for ptype, examples in paraphrase_type_to_examples.items():
        # Shuffle the examples for random distribution
        random.shuffle(examples)

        # Split the examples into train/test sets for this paraphrase type
        split_idx = int(train_percent * len(examples))
        train_subset = examples[:split_idx]
        test_subset = examples[split_idx:]

        train_examples.extend(train_subset)
        test_examples[ptype].extend(test_subset)

    # Step 3: Remove duplicates based on unique 'idx' field
    train_examples = list({ex['idx']: ex for ex in train_examples}.values())
    test_examples_flat = list({ex['idx']: ex for ex in [ex for sublist in test_examples.values() for ex in sublist]}.values())

    # Step 4: Create train and test datasets
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples_flat)

    # Step 5: Use the existing count_paraphrase_types function to count occurrences
    train_type_counts = count_paraphrase_types(train_dataset)
    test_type_counts = count_paraphrase_types(test_dataset)

    # Debug: Print final train and test counts
    print(f"Train counts: {train_type_counts}")
    print(f"Test counts: {test_type_counts}")

    # Return datasets and counts
    return train_dataset, test_dataset

def count_paraphrase_types(dataset: Dataset) -> Dict[str, int]:
    """
    Counts the number of examples containing each paraphrase type, 
    counting each example only once per type.
    """
    type_counts: Dict[str, int] = {}

    for example in dataset:
        # Check if "paraphrase_types" field exists and is a list
        if "paraphrase_types" not in example or not isinstance(example["paraphrase_types"], list):
            print(f"Skipping example due to missing paraphrase_types: {example}")
            continue

        # Use a set to track unique types within the current example
        unique_types = set(ptype.strip().lower() for ptype in example["paraphrase_types"])

        # Increment the count for each unique type that is in the top-10 list
        for ptype in unique_types:
            if ptype in TOP_10_PARAPHRASE_TYPES:
                type_counts[ptype] = type_counts.get(ptype, 0) + 1

    return type_counts


def hyperparameter_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6,1e-2, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
    }

def save_best_params_callback(study, trial):
    # Retrieve the best parameters so far
    best_params = study.best_params

    # Convert the best parameters to a DataFrame
    best_params_df = pd.DataFrame([best_params])

    # Save the best parameters to a CSV file
    best_params_df.to_csv("best_hyperparameters_intermediate.csv", index=False)
    print(f"Saved best parameters so far: {best_params}")
    
def main():
    args = parse_arguments()

    dataset = load_dataset("jpwahle/etpc").filter(lambda x: x["etpc_label"] == 1)["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, clean_up_tokenization_spaces=True)

    dataset_tokenized = dataset.map(
        tokenize_examples, batched=True, fn_kwargs={"sentence1_key": "sentence1", "sentence2_key": "sentence2", "tokenizer": tokenizer}
    )
    
    # Split the dataset into training and testing sets with type counts
    train_dataset, test_dataset = split_dataset_by_type(dataset_tokenized, train_percent=0.8)
    
    
    def model_init(trial=None):
        # Use default weights if no trial is provided (e.g., when not performing hyperparameter search)
        if trial is not None:
            # Extract class weights from the trial
            class_weights = torch.tensor([
                trial.suggest_float(f"class_weight_{i}", 0.01, 20.0, log=True) for i in range(len(TOP_10_PARAPHRASE_TYPE_TO_ID))
            ], dtype=torch.float32)
        else:
            class_weights = torch.tensor([
                0.2259265999213708, 
                3.220232406121911, 
                0.35486436950141176, 
                0.9969864692714715, 
                4.172419625825067, 
                8.297736959636888, 
                8.595439973331159, 
                9.962680162343734, 
                0.14255471526284638, 
                0.21035107399865138
            ], dtype=torch.float32)

        config = AutoConfig.from_pretrained(
            args.model_name, 
            num_labels=len(TOP_10_PARAPHRASE_TYPE_TO_ID), 
            problem_type="multi_label_classification"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, config=config, ignore_mismatched_sizes=True
        )

        # Use the sampled class weights in the loss function
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

        def compute_loss(model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

        model.compute_loss = compute_loss
        return model


    trainer = Trainer(
        model_init=lambda trial: model_init(trial),  # Pass the trial object to model_init
        args=TrainingArguments(
            output_dir=f"./out/cls-models/{args.model_name.split('/')[-1]}_etpc_ptd", 
            #per_device_train_batch_size=16,
            #learning_rate=4.0975283438994273e-05,
            #weight_decay=0.0,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            fp16=True,
            num_train_epochs=50,
            metric_for_best_model='macro-f1',
            load_best_model_at_end=True,
            greater_is_better=True,
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        compute_metrics=lambda p: compute_metrics(p.predictions, p.label_ids),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Perform the hyperparameter search using Optuna
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hyperparameter_space,
        n_trials=300, #300
        backend="optuna",
        callbacks=[save_best_params_callback]
    )
    
    best_hyperparameters = best_run.hyperparameters
    best_params_df = pd.DataFrame([best_hyperparameters])

    # Export to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_params_df.to_csv(f"out/cls-models/{args.model_name.split('/')[-1]}_hyperparameters_ptd_{timestamp}.csv", index=False)

    # Apply the best hyperparameters found by the search
    trainer.args.learning_rate = best_run.hyperparameters['learning_rate']
    trainer.args.weight_decay = best_run.hyperparameters['weight_decay']
    trainer.args.per_device_train_batch_size = best_run.hyperparameters['per_device_train_batch_size']

    # Extract the best class weights for further use or logging
    best_class_weights = [best_run.hyperparameters[f'class_weight_{i}'] for i in range(len(TOP_10_PARAPHRASE_TYPE_TO_ID))]
    print(f"Best class weights: {best_class_weights}")
        
    # Print a few examples from the train and test datasets to verify labels
    trainer.train()

    # Evaluate and write results to CSV
    results = trainer.evaluate()    
    write_results_to_csv(results, output_file=f"out/cls-models/{args.model_name.split('/')[-1]}_ptd_results_hyperclass_{timestamp}.csv")

if __name__ == "__main__":
    main()
