import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

nlp = spacy.load("en_core_web_sm")

# Groups



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

    return parser.parse_args()

def load_and_preprocess_apty_dataset(dataset):
    """
    Load and preprocess the APTY-ranked dataset into Hugging Face Dataset format for DPOTrainer.
    
    Args:
        dataset: The raw dataset object containing 'train' data.
        
    Returns:
        train_dataset (Dataset): Training dataset as Hugging Face Dataset object.
        test_dataset (Dataset): Validation dataset as Hugging Face Dataset object.
    """
    
    # Convert dataset into a pandas DataFrame
    data = pd.DataFrame(dataset["train"])
    
    # Normalize the 'meta' column to create separate columns for 'id', 'annotators', and 'APT'
    meta_df = pd.json_normalize(data['meta'])
    data = data.drop(columns=['meta']).reset_index(drop=True)
    data = pd.concat([data, meta_df], axis=1)

    # Extract the text from the nested dictionaries for 'chosen' and 'rejected'
    data['original'] = data['original'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x))
    data['chosen'] = data['chosen'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x))
    data['rejected'] = data['rejected'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x))

    # Strip whitespace
    data['original'] = data['original'].str.strip()
    data['chosen'] = data['chosen'].str.strip()
    data['rejected'] = data['rejected'].str.strip()

    # Rename columns to match DPOTrainer's expected format
    data = data.rename(columns={'original': 'prompt'})

    # Split the dataset into training and test sets
    train_df, test_df = train_test_split(data, test_size=0.3, stratify=data["APT"], random_state=42)

    # Convert the pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset

def modify_last_character(text: str) -> str:
    """
    Modify the last character of a string based on specific rules.

    Args:
        text (str): The text to modify.

    Returns:
        str: The modified text.
    """
    if text.endswith('"'):
        text = text[:-1]  # Remove the last double quote
    elif text[-1].isalpha():
        text += '.'  # Add a '.' if the last character is a letter

    return text

def main():
    args = parse_args()
    
    fine_tuned_model_dir = f"./out/gen-models/{args.model_name}_paraphrase-type-generation"  # Path to the fine-tuned model
    output_dir = f"./out/gen-models/{args.model_name}_{args.task_name}"
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()  # Clear GPU cache before starting

    # Load training and evaluation datasets
    dataset = load_dataset("worta/apty", "APTY-ranked")
    train_dataset, eval_dataset = load_and_preprocess_apty_dataset(dataset)
    
    # Find the latest checkpoint from the fine-tuned model directory
    checkpoint_dir = None
    if os.path.exists(fine_tuned_model_dir) and os.listdir(fine_tuned_model_dir):
        checkpoint_dirs = [f for f in os.listdir(fine_tuned_model_dir) if f.startswith('checkpoint')]
        if checkpoint_dirs:
            checkpoint_dir = os.path.join(fine_tuned_model_dir, max(checkpoint_dirs, key=lambda x: os.path.getctime(os.path.join(fine_tuned_model_dir, x))))
            print(f"Loading from fine-tuned checkpoint: {checkpoint_dir}")
        else:
            print("No checkpoint found in fine-tuned model directory.")
            return

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    # For the paraphrase_type_detection task, we need to create a data collator
    data_collator = None
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
            tokenize_and_align_labels,
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

        # For Paraphrase Type Detection we need a data collator for token-level classifiction
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=27,
            id2label=cls_id2label,
            label2id=label2cls_id,
        ).to(args.device)

    elif args.task_name == "paraphrase-detection":
        # Tokenize the dataset
        dataset_tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            fn_kwargs={
                "sentence1_key": sentence1_key,
                "sentence2_key": sentence2_key,
                "tokenizer": tokenizer,
            },
        )

        # Train test split
        train, test = split_dataset_binary(dataset_tokenized)

        # Create model
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            id2label={0: "no_paraphrase", 1: "paraphrase"},
            label2id={"no_paraphrase": 0, "paraphrase": 1},
        ).to(args.device)

        # Define metric
        metric = evaluate.load("f1")

        def compute_metrics_binary(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(
                predictions=predictions, references=labels, average="micro"
            )

        # Set metric
        compute_metrics = compute_metrics_binary

    else:
        raise NotImplementedError(f"Task {args.task_name} not implemented.")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{args.model_name}-{args.dataset_name}-{args.task_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=None,
        use_mps_device=args.device == "mps",
    )

    # Traininer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()

    # Evaluation
    results = trainer.evaluate()
    print("#" * 20)
    print(args.model_name)
    print(args.task_name)
    print(results)
    print("#" * 20)

    # Map results to lists in case of scalar values
    if args.task_name == "paraphrase-detection":
        results = pd.DataFrame(results, index=[0])
    else:
        results = pd.DataFrame(results)

    # Store results
    results.to_csv(
        f"{args.model_name}-{args.dataset_name}-paraphrase-{args.task_name}-results.csv",
    )


if __name__ == "__main__":
    main()