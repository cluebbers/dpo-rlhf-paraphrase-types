import argparse
import os
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
)
import random

def write_results_to_csv(results, detailed_report, output_file="evaluation_results.csv"):
    print(f"Writing results to {output_file}...")
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write the overall metrics
        writer.writerow(["Metric", "Value"])
        for key, value in results.items():
            if key != 'detailed_report':  # Skip the detailed report for now
                writer.writerow([key, value])
        
        # Leave a line before detailed report
        writer.writerow([])

        # Write detailed classification report
        if detailed_report:
            writer.writerow(["Label", "Precision", "Recall", "F1-Score", "Support"])
            for label, metrics in detailed_report.items():
                precision = metrics.get("precision", 0)
                recall = metrics.get("recall", 0)
                f1_score = metrics.get("f1-score", 0)
                support = metrics.get("support", 0)
                writer.writerow([label, precision, recall, f1_score, support])
    
    print(f"Results successfully written to {output_file}.")
    
def download_paraphrase_types_xml():
    print("Downloading paraphrase types XML...")
    url = "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/paraphrase_types.xml"
    response = requests.get(url)
    root = ET.fromstring(response.text)

    paraphrase_types = [child.find("type_name").text.strip().lower() for child in root]
    paraphrase_type_ids = [int(child.find("type_id").text) for child in root]
    paraphrase_type_categories = [child.find("type_category").text for child in root]

    print("Downloaded and parsed paraphrase types XML.")
    return paraphrase_types, paraphrase_type_ids, paraphrase_type_categories

def create_label_maps():
    print("Creating label maps...")
    paraphrase_types, paraphrase_type_ids, paraphrase_type_categories = download_paraphrase_types_xml()

    paraphrase_types = [ptype.strip().lower() for ptype in paraphrase_types]

    paraphrase_type2cls_id = dict(zip(paraphrase_types, paraphrase_type_ids))
    paraphrase_id2cls_type = dict(zip(paraphrase_type_ids, paraphrase_types))
    paraphrase_type_to_category = dict(zip(paraphrase_types, paraphrase_type_categories))

    paraphrase_type2cls_id["no_paraphrase"] = 0
    paraphrase_id2cls_type[0] = "no_paraphrase"
    paraphrase_type_to_category["no_paraphrase"] = "no_paraphrase"

    label2cls_id = {label: idx for idx, label in enumerate(paraphrase_types, start=0)}
    label2cls_id["no_paraphrase"] = 0
    cls_id2label = {idx: label for label, idx in label2cls_id.items()}

    cls_id2paraphrase_type_id = {i: paraphrase_type2cls_id[cls_id2label[i]] for i in cls_id2label}
    paraphrase_type_id2cls_id = {v: k for k, v in cls_id2paraphrase_type_id.items()}

    print("Label maps created successfully.")
    return (
        label2cls_id,
        cls_id2label,
        paraphrase_type2cls_id,
        paraphrase_id2cls_type,
        paraphrase_type_to_category,
        cls_id2paraphrase_type_id,
        paraphrase_type_id2cls_id,
    )

def tokenize_fn(examples, sentence1_key, sentence2_key, tokenizer, paraphrase_type2cls_id):
    print("Tokenizing examples...")
    max_length = 512
    tokenized_inputs = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key] if sentence2_key else None,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    num_labels = len(paraphrase_type2cls_id)
    standardized_paraphrase_type2cls_id = {k.strip().lower(): v for k, v in paraphrase_type2cls_id.items()}

    paraphrase_type_standardized_cache = {}
    labels = []
    for paraphrase_type_list in examples["paraphrase_types"]:
        binary_labels = [0] * num_labels
        for paraphrase_type in paraphrase_type_list:
            paraphrase_type_standardized = paraphrase_type_standardized_cache.get(paraphrase_type, paraphrase_type.strip().lower())
            paraphrase_type_standardized_cache[paraphrase_type] = paraphrase_type_standardized
            cls_id = standardized_paraphrase_type2cls_id.get(paraphrase_type_standardized)
            if cls_id is not None and cls_id < num_labels:
                binary_labels[cls_id] = 1
        labels.append(binary_labels)

    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)  # Ensure labels are in float32
    print("Tokenization completed.")
    return tokenized_inputs

def parse_args():
    print("Parsing command line arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base", help="Name of the model to use")
    parser.add_argument("--max_samples_per_class", type=int, default=100, help="Maximum samples per class for downsampling")
    args = parser.parse_args()
    print(f"Arguments parsed: model_name={args.model_name}, max_samples_per_class={args.max_samples_per_class}")
    return args

def compute_metrics(p, cls_id2label):
    print("Computing metrics...")
    predictions, labels = p
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probs = sigmoid(predictions)
    preds = (probs > 0.5).astype(int)

    preds_flat = preds.flatten()
    labels_flat = np.array(labels).flatten()

    accuracy = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat, average='macro', zero_division=0)
    recall = recall_score(labels_flat, preds_flat, average='macro', zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, average='macro', zero_division=0)

    # Ensure target_names length matches the number of classes
    target_names = [cls_id2label.get(i, f"label_{i}") for i in range(len(cls_id2label))]

    report = classification_report(
        labels, preds, target_names=target_names, labels=np.arange(len(target_names)), output_dict=True
    )

    print("Metrics computed.")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detailed_report': report
    }

def split_dataset_by_type(dataset, train_percent=0.8):
    print("Splitting dataset by paraphrase type...")
    train = []
    test = []
    counts = {}
    for paraphrase_type_list in dataset["paraphrase_types"]:
        for paraphrase_type in paraphrase_type_list:
            paraphrase_type_standardized = paraphrase_type.strip().lower()
            if paraphrase_type_standardized not in counts:
                counts[paraphrase_type_standardized] = 1
            else:
                counts[paraphrase_type_standardized] += 1

    internal_counts = {key: 0 for key in counts}

    for example in dataset:
        types = [ptype.strip().lower() for ptype in example["paraphrase_types"]]
        if len(types) > 0:
            par_type = min(types, key=lambda x: internal_counts[x])
            if internal_counts[par_type] < counts[par_type] * (1 - train_percent):
                test.append(example)
            else:
                train.append(example)
            for par_type in types:
                internal_counts[par_type] += 1

    train_dataset = Dataset.from_list(train)
    test_dataset = Dataset.from_list(test)

    print(f"Dataset split completed. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, test_dataset

def downsample_dataset(dataset, label_column, max_samples_per_class):
    """
    Downsample overrepresented classes in the dataset.
    
    Args:
    - dataset: The dataset to downsample.
    - label_column: The name of the column containing class labels.
    - max_samples_per_class: Maximum number of samples per class after downsampling.
    
    Returns:
    - downsampled_dataset: The downsampled dataset.
    """
    print("Downsampling overrepresented classes...")

    # Group examples by their paraphrase type
    class_groups = {label: [] for label in set([ptype for ptypes in dataset[label_column] for ptype in ptypes])}
    for example in dataset:
        for label in example[label_column]:
            class_groups[label.strip().lower()].append(example)

    # Downsample overrepresented classes
    downsampled_data = []
    for label, examples in class_groups.items():
        if len(examples) > max_samples_per_class:
            downsampled_data.extend(random.sample(examples, max_samples_per_class))
        else:
            downsampled_data.extend(examples)

    downsampled_dataset = Dataset.from_list(downsampled_data)
    print(f"Downsampling complete. Final dataset size: {len(downsampled_dataset)}")
    return downsampled_dataset

def main():
    args = parse_args()

    # Load dataset and tokenizer
    print("Loading dataset...")
    dataset = load_dataset("jpwahle/etpc")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True, clean_up_tokenization_spaces=True)
    sentence1_key, sentence2_key = "sentence1", "sentence2"
    dataset = dataset["train"]
    print("Dataset loaded.")

    # Convert all paraphrase types in the dataset to lowercase for consistency
    print("Converting paraphrase types to lowercase...")
    def lowercase_labels(examples):
        examples["paraphrase_types"] = [ptype.strip().lower() for ptype in examples["paraphrase_types"]]
        return examples

    dataset = dataset.map(lowercase_labels)
    print("Paraphrase types converted to lowercase.")

    # Create label maps for the paraphrase types
    label_maps = create_label_maps()
    (
        label2cls_id,
        cls_id2label,
        paraphrase_type2cls_id,
        paraphrase_id2cls_type,
        paraphrase_type_to_category,
        cls_id2paraphrase_type_id,
        paraphrase_type_id2cls_id,
    ) = label_maps

    # Tokenize the dataset and prepare for training
    print("Tokenizing dataset...")
    dataset_tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={
            "sentence1_key": sentence1_key,
            "sentence2_key": sentence2_key,
            "tokenizer": tokenizer,
            "paraphrase_type2cls_id": paraphrase_type2cls_id,
        },
    )
    print("Dataset tokenized.")

    # Downsample the dataset to address class imbalance
    print("Downsampling dataset...")
    max_samples_per_class = args.max_samples_per_class  # Control this from command-line
    dataset_tokenized = downsample_dataset(dataset_tokenized, 'paraphrase_types', max_samples_per_class)

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = split_dataset_by_type(dataset_tokenized)

    # Define model configuration with multi-label classification
    print("Loading model configuration...")
    num_labels = len(paraphrase_type2cls_id)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    print("Model configuration loaded.")

    # Load pretrained model and modify it for weighted loss
    print("Loading pretrained model...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    print("Pretrained model loaded.")

    # Calculate class weights to handle class imbalance
    print("Calculating class weights...")
    label_counts = np.sum(np.array(dataset_tokenized["labels"], dtype=np.float32), axis=0)
    class_weights = 1.0 / (label_counts + np.finfo(np.float32).eps)  # Avoid division by zero
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Class weights calculated.")

    # Modify the model to include the custom weighted loss function
    model.classifier.loss_fct = nn.BCEWithLogitsLoss(weight=class_weights)
    print("Custom weighted loss function added to the model.")

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data collator and training arguments
    print("Setting up data collator and training arguments...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{args.model_name.replace('/', '_')}_etpc_seq-cls",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
    )
    print("Data collator and training arguments set up.")

    # Create Trainer object for model training and evaluation
    print("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, cls_id2label),
    )
    print("Trainer created.")

    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Evaluate the model
    print("Starting evaluation...")
    results = trainer.evaluate()
    print("Evaluation completed.")
    print("#" * 20)
    print(args.model_name)
    print(results)
    print("#" * 20)

    # Print detailed metrics by paraphrase type
    detailed_report = results.pop('detailed_report', None)
    if detailed_report:
        print("Printing detailed classification report...")
        for label, metrics in detailed_report.items():
            print(f"Label: {label}")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    
    # Ensure directory for output exists
    output_file = f"out/eval_{args.model_name.replace('/', '_')}_etpc_seq-cls_results.csv"
    output_dir = os.path.dirname(output_file)

    # Only create the directory if output_dir is not empty
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Write results to CSV
    write_results_to_csv(results, detailed_report, output_file=output_file)

if __name__ == "__main__":
    main()
