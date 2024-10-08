import argparse
import torch
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

"""_summary_
python3 src/seq-cls.py \
--model_name=microsoft/deberta-base
"""

def download_paraphrase_types_xml():
    """
    Downloads and parses the XML containing paraphrase types.

    Returns:
        tuple: Lists of paraphrase types, their IDs, and categories.
    """
    url = "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/paraphrase_types.xml"
    response = requests.get(url)
    root = ET.fromstring(response.text)

    paraphrase_types = [child.find("type_name").text.strip().lower() for child in root]
    paraphrase_type_ids = [int(child.find("type_id").text) for child in root]
    paraphrase_type_categories = [child.find("type_category").text for child in root]

    return paraphrase_types, paraphrase_type_ids, paraphrase_type_categories

def create_label_maps(etpc):
    """
    Creates label maps for the ETPC paraphrase types in a sequential order.

    Returns:
        tuple: Dictionaries for mapping between different paraphrase type representations.
    """
    paraphrase_types, paraphrase_type_ids, paraphrase_type_categories = download_paraphrase_types_xml()

    # Create mappings between paraphrase types, IDs, and categories
    paraphrase_type2cls_id = dict(zip(paraphrase_types, paraphrase_type_ids))
    paraphrase_id2cls_type = dict(zip(paraphrase_type_ids, paraphrase_types))
    paraphrase_type_to_category = dict(zip(paraphrase_types, paraphrase_type_categories))

    # Add "no paraphrase" to all mappings
    paraphrase_type2cls_id["no_paraphrase"] = 0
    paraphrase_id2cls_type[0] = "no_paraphrase"
    paraphrase_type_to_category["no_paraphrase"] = "no_paraphrase"

    # Create label2id and id2label mappings
    label2cls_id = {label: idx for idx, label in enumerate(paraphrase_types, start=1)}
    label2cls_id["no_paraphrase"] = 0
    cls_id2label = {idx: label for label, idx in label2cls_id.items()}

    # Create maps from class IDs to paraphrase type IDs and vice versa
    cls_id2paraphrase_type_id = {i: paraphrase_type2cls_id[cls_id2label[i]] for i in cls_id2label}
    paraphrase_type_id2cls_id = {v: k for k, v in cls_id2paraphrase_type_id.items()}

    # Debugging
    print("Keys in paraphrase_type2cls_id:", paraphrase_type2cls_id.keys())

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
    """
    Tokenizes input examples and creates multi-label binary labels.

    Returns:
        dict: The tokenized inputs with multi-label binary labels.
    """
    max_length = 512
    tokenized_inputs = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key] if sentence2_key else None,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    # Set the number of labels based on paraphrase_type2cls_id
    num_labels = len(paraphrase_type2cls_id)

    # Standardize the paraphrase type keys in paraphrase_type2cls_id for comparison
    standardized_paraphrase_type2cls_id = {k.strip().lower(): v for k, v in paraphrase_type2cls_id.items()}

    labels = []
    for paraphrase_type_list in examples["paraphrase_types"]:
        binary_labels = [0] * num_labels
        for paraphrase_type in paraphrase_type_list:
            paraphrase_type_standardized = paraphrase_type.strip().lower()
            cls_id = standardized_paraphrase_type2cls_id.get(paraphrase_type_standardized)
            if cls_id is not None and cls_id < num_labels:
                binary_labels[cls_id] = 1
            else:
                print(f"Warning: paraphrase type '{paraphrase_type}' not found after standardizing to '{paraphrase_type_standardized}'")
        labels.append(binary_labels)

    tokenized_inputs["labels"] = np.array(labels, dtype=np.float32)
    return tokenized_inputs

def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base", help="Name of the model to use")
    return parser.parse_args()

def compute_metrics(p, cls_id2label):
    """
    Computes evaluation metrics for multi-label classification.

    Args:
        p (EvalPrediction): An EvalPrediction object containing predictions and label_ids.
    Returns:
        dict: A dictionary containing computed metrics.
    """
    predictions, labels = p
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probs = sigmoid(predictions)
    preds = (probs > 0.5).astype(int)

    # Flatten predictions and labels for calculating overall metrics
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    accuracy = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat, average='macro', zero_division=0)
    recall = recall_score(labels_flat, preds_flat, average='macro', zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, average='macro', zero_division=0)

    # Compute metrics by paraphrase type
    report = classification_report(labels, preds, target_names=[cls_id2label[i] for i in range(len(cls_id2label))], output_dict=True)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detailed_report': report
    }

def main():
    args = parse_args()

    # Load the dataset and tokenizer
    dataset = load_dataset("jpwahle/etpc")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    sentence1_key, sentence2_key = "sentence1", "sentence2"
    dataset = dataset["train"]

    # Create label maps
    label_maps = create_label_maps(dataset)
    (
        label2cls_id,
        cls_id2label,
        paraphrase_type2cls_id,
        paraphrase_id2cls_type,
        paraphrase_type_to_category,
        cls_id2paraphrase_type_id,
        paraphrase_type_id2cls_id,
    ) = label_maps

    # Tokenize the dataset
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

    # Split dataset into train and test
    train, test = dataset_tokenized.train_test_split(train_size=0.8).values()

    # Define model configuration
    num_labels = len(paraphrase_type2cls_id)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config).to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{args.model_name.replace('/', '_')}_etpc_seq-cls",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, cls_id2label),
    )

    # Train and evaluate the model
    print("Starting training")
    trainer.train()

    print("Starting evaluation")
    results = trainer.evaluate()
    print("#" * 20)
    print(args.model_name)
    print(results)
    print("#" * 20)

    # Print detailed metrics by paraphrase type
    detailed_report = results.pop('detailed_report', None)
    if detailed_report:
        for label, metrics in detailed_report.items():
            print(f"Label: {label}")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")

    # Store results
    results_df = pd.DataFrame([results])  # Ensure results are passed as a list of dictionaries
    results_df.to_csv(
        f"./out/{args.model_name.replace('/', '_')}_etpc_seq-cls_results.csv",
        index=False
    )

if __name__ == "__main__":
    main()
