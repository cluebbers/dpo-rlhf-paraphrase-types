import argparse
import xml.etree.ElementTree as ET

import evaluate
import numpy as np
import pandas as pd
import requests
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

def create_label_maps(etpc):
    """
    Creates label maps for the ETPC paraphrase types.

    Returns:
        tuple: A tuple containing the following dictionaries:
            - label2cls_id: A dictionary mapping paraphrase types to class IDs.
            - cls_id2label: A dictionary mapping class IDs to paraphrase types.
            - paraphrase_type2cls_id: A dictionary mapping paraphrase types to class IDs.
            - paraphrase_id2cls_type: A dictionary mapping class IDs to paraphrase types.
            - paraphrase_type_to_category: A dictionary mapping paraphrase types to categories.
            - cls_id2paraphrase_type_id: A dictionary mapping class IDs to paraphrase type IDs.
            - paraphrase_type_id2cls_id: A dictionary mapping paraphrase type IDs to class IDs.

    Example:
        ```python
        label_maps = create_label_maps(etpc)
        print(label_maps)
        ```"""
    # Flatten paraphrase_types as list
    all_types = {el for sublist in etpc["paraphrase_types"] for el in sublist}

    # Download xml with paraphrase types to ids from url https://github.com/venelink/ETPC/blob/master/Corpus/paraphrase_types.xml
    url = "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/paraphrase_types.xml"
    r = requests.get(url)
    root = ET.fromstring(r.text)

    # Get paraphrase types, ids and categories
    paraphrase_types = [child.find("type_name").text for child in root]
    paraphrase_type_ids = [int(child.find("type_id").text) for child in root]
    paraphrase_type_categories = [child.find("type_category").text for child in root]

    # Create dictionary with paraphrase type as key and paraphrase type id as value
    paraphrase_type2cls_id = dict(zip(paraphrase_types, paraphrase_type_ids))
    paraphrase_id2cls_type = dict(zip(paraphrase_type_ids, paraphrase_types))

    # Create dictionary with paraphrase type as key and paraphrase type category as value
    paraphrase_type_to_category = dict(
        zip(paraphrase_types, paraphrase_type_categories)
    )

    # Add 0 for no paraphrase to all dictionaries
    paraphrase_type2cls_id["no_paraphrase"] = 0
    paraphrase_id2cls_type[0] = "no_paraphrase"
    paraphrase_type_to_category["no_paraphrase"] = "no_paraphrase"

    # Create label2id and id2label for etpc paraphrase_types
    label2cls_id = {label: i + 1 for i, label in enumerate(all_types)}
    cls_id2label = {i: label for label, i in label2cls_id.items()}

    # Add 0 for no paraphrase to all dictionaries
    label2cls_id["no_paraphrase"] = 0
    cls_id2label[0] = "no_paraphrase"

    # Create a map from ids to the ones in paraphrase_type_to_id and vice versa
    cls_id2paraphrase_type_id = {
        i: paraphrase_type2cls_id[cls_id2label[i]] for i in cls_id2label
    }
    paraphrase_type_id2cls_id = {
        paraphrase_type2cls_id[cls_id2label[i]]: i for i in cls_id2label
    }

    # Create a dictionary that maps ids from label2cls_id to the ones in paraphrase_type_to_id using the type label and vice versa
    cls_id2paraphrase_type_id = {
        i: paraphrase_type2cls_id[cls_id2label[i]] for i in cls_id2label
    }
    paraphrase_type_id2cls_id = {
        paraphrase_type2cls_id[cls_id2label[i]]: i for i in cls_id2label
    }
    print(paraphrase_type_id2cls_id)
    return (
        label2cls_id,
        cls_id2label,
        paraphrase_type2cls_id,
        paraphrase_id2cls_type,
        paraphrase_type_to_category,
        cls_id2paraphrase_type_id,
        paraphrase_type_id2cls_id,
    )


def tokenize_fn(examples, sentence1_key, sentence2_key, tokenizer, paraphrase_type_id2cls_id):
    """
    Tokenizes input examples using a tokenizer.

    Args:
        examples (dict): The input examples.
        sentence1_key (str): The key for the first sentence in the examples.
        sentence2_key (str, optional): The key for the second sentence in the examples.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        paraphrase_type_id2cls_id (dict): A dictionary mapping paraphrase type IDs to class IDs.

    Returns:
        dict: The tokenized inputs.
    """
    tokenized_inputs = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key] if sentence2_key else None,
        padding="max_length",
        truncation=True,
    )
    
    # Assign a single label for each input sequence using paraphrase_type_id2cls_id
    # Assuming examples["paraphrase_types"] is a list of paraphrase type IDs for each example
    tokenized_inputs["labels"] = [paraphrase_type_id2cls_id[paraphrase_type[0]] for paraphrase_type in examples["paraphrase_types"]]

    return tokenized_inputs



def split_dataset_by_type(dataset, train_percent=0.5):
    """
    Splits a dataset into training and testing sets based on paraphrase types.

    Args:
        dataset (dict): The dataset to split.
        train_percent (float, optional): The percentage of data to allocate for training.
        Defaults to 0.5.

    Returns:
        tuple: A tuple containing two lists: train and test. train contains the training examples,
        and test contains the testing examples.

    Example:
        ```python
        dataset = {
            "paraphrase_types": [["type1"], ["type2"]],
            "examples": [
                {"paraphrase_types": ["type1"]},
                {"paraphrase_types": ["type2"]},
                {"paraphrase_types": ["type1", "type2"]},
            ]
        }
        train, test = split_dataset_by_type(dataset, train_percent=0.7)
        print(train)
        print(test)
        ```
    """

    train = []
    test = []
    counts = {}
    for paraphrase_type in dataset["paraphrase_types"]:
        for par_type in paraphrase_type:
            if par_type not in counts:
                counts[par_type] = 1
            else:
                counts[par_type] += 1
    internal_counts = {key: 0 for key in counts}

    for example in dataset:
        types = example["paraphrase_types"]
        if len(types) > 0:
            # Get the type which has the lowest internal count
            par_type = min(types, key=lambda x: internal_counts[x])
            if internal_counts[par_type] < counts[par_type] * (1 - train_percent):
                test.append(example)
            else:
                train.append(example)
            for par_types in types:
                internal_counts[par_types] += 1

    return train, test


def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.

    Example:
        ```python
        args = parse_args()
        print(args.model_name)
        print(args.dataset_name)
        print(args.task_name)
        ```
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-large-uncased",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jpwahle/etpc",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="paraphrase-detection",
        help="Name of the task to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Hardware to use, either of 'cpu', 'cuda', or 'mps'",
    )
    return parser.parse_args()


def tokenize_fn(examples, sentence1_key, sentence2_key, tokenizer, paraphrase_type_id2cls_id):
    """
    Tokenizes input examples using a tokenizer.

    Args:
        examples (dict): The input examples.
        sentence1_key (str): The key for the first sentence in the examples.
        sentence2_key (str, optional): The key for the second sentence in the examples.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        paraphrase_type_id2cls_id (dict): A dictionary mapping paraphrase type IDs to class IDs.

    Returns:
        dict: The tokenized inputs.
    """
    tokenized_inputs = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key] if sentence2_key else None,
        padding="max_length",
        truncation=True,
    )
    
    # Assign a single label for each input sequence using paraphrase_type_id2cls_id
    # Assuming examples["paraphrase_types"] is a list of paraphrase type IDs for each example
    tokenized_inputs["labels"] = [paraphrase_type_id2cls_id[paraphrase_type[0]] for paraphrase_type in examples["paraphrase_types"]]

    return tokenized_inputs



def main():
    args = parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    if "etpc" in args.dataset_name:
        # ETPC dataset keys
        sentence1_key = "sentence1"
        sentence2_key = "sentence2"
        dataset = dataset["train"]

    # For the paraphrase_type_detection task, we need to create a data collator
    compute_metrics = None

    # Split the dataset into train and test
    if args.task_name == "paraphrase-type-detection":
        # Get label maps
        (
            label2cls_id,
            cls_id2label,
            paraphrase_type2cls_id,
            paraphrase_id2cls_type,
            paraphrase_type_to_category,
            cls_id2paraphrase_type_id,
            paraphrase_type_id2cls_id,
        ) = create_label_maps(dataset)

        # Tokenize the dataset
        dataset_tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            fn_kwargs={
                "sentence1_key": sentence1_key,
                "sentence2_key": sentence2_key,
                "tokenizer": tokenizer,
                "paraphrase_type_id2cls_id": paraphrase_type_id2cls_id,
            },
        )


        # Train test split
        train, test = split_dataset_by_type(dataset_tokenized)

        # Define metric
        metric = evaluate.load("seqeval")

        def compute_metrics_types(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [
                    cls_id2label[p]
                    for (p, lab) in zip(prediction, label)
                    if lab not in [-100, 0]
                ]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [
                    cls_id2label[lab]
                    for (p, lab) in zip(prediction, label)
                    if lab not in [-100, 0]
                ]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(
                predictions=true_predictions, references=true_labels
            )
            return results

        # Set metric
        compute_metrics = compute_metrics_types

        # Create a model that predicts the paraphrase_type of a sentence pair
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(cls_id2label),  # Ensure this matches the number of unique classes
            id2label=cls_id2label,
            label2id=label2cls_id,
        ).to(args.device)


    else:
        raise NotImplementedError(f"Task {args.task_name} not implemented.")

    # Replace '/' with '_' to avoid directory creation issues
    sanitized_dataset_name = args.dataset_name.replace('/', '-')
    sanitized_model_name = args.model_name.replace('/', '-')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{sanitized_model_name}_{sanitized_dataset_name}_{args.task_name}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
    )

    # Traininer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Training
    print("Starting training")
    trainer.train()

    # Evaluation
    print("Starting evaluation")
    results = trainer.evaluate()
    print("#" * 20)
    print(args.model_name)
    print(args.task_name)
    print(results)
    print("#" * 20)

    # Map results to lists in case of scalar values
    results = pd.DataFrame(results)

    # Store results
    
    results.to_csv(
        f"./out/{sanitized_model_name}_{sanitized_dataset_name}_{args.task_name}_results.csv",
    )

if __name__ == "__main__":
    main()
