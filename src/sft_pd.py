import argparse
from typing import Dict, List, Optional, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

# Global dictionary for all metrics
metrics = {
    "accuracy": evaluate.load("accuracy"),
    "precision": evaluate.load("precision"),
    "recall": evaluate.load("recall"),
    "f1": evaluate.load("f1"),
}


def tokenize_examples(examples, tokenizer):
    """
    Tokenizes input examples using the provided tokenizer.

    Args:
        examples (Dict[str, List[str]]): A dictionary with keys 'question1', 'question2', and 'label'.
            Each value is a list of strings.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for encoding the questions.

    Returns:
        Dict[str, List[int]]: Tokenized examples with the corresponding labels.
            The dictionary has the same keys as the input examples, but the values are lists of token IDs.
    """
    tokenized_examples = tokenizer(
        examples["question1"],
        examples["question2"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    tokenized_examples["label"] = examples["label"]

    return tokenized_examples


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a paraphrase detection model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-base",
        help="Model name from Hugging Face hub.",
    )
    return parser.parse_args()


def compute_binary_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray]
) -> Dict[str, float]:
    """
    Computes accuracy, precision, recall, and F1 score for binary classification.

    Parameters
    ----------
    eval_pred : Tuple[np.ndarray, np.ndarray]
        Model predictions and corresponding labels.
        The first element is a 2D array of shape (batch_size, num_classes) containing the
        model predictions. The second element is a 1D array of shape (batch_size,) containing
        the corresponding labels.

    Returns
    -------
    Dict[str, float]
        Computed accuracy, precision, recall, and F1 score.
        A dictionary containing the accuracy, precision, recall, and F1 score as floats.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Ensure labels are either 0 or 1, not -1 or 0
    labels = np.where(labels == -1, 0, labels)

    # Compute accuracy, precision, recall, and F1 score for binary classification
    accuracy = metrics["accuracy"].compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    precision = metrics["precision"].compute(
        predictions=predictions, references=labels, average="binary", pos_label=1
    )["precision"]
    recall = metrics["recall"].compute(
        predictions=predictions, references=labels, average="binary", pos_label=1
    )["recall"]
    f1 = metrics["f1"].compute(
        predictions=predictions, references=labels, average="binary", pos_label=1
    )["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_and_tokenize_dataset(
    tokenizer: PreTrainedTokenizerBase, subset_size: Optional[int] = None
) -> Dict[str, Dataset]:
    """
    Load and tokenize the QQP dataset using the provided tokenizer.

    Args:
        tokenizer: PreTrainedTokenizerBase
            The tokenizer used to encode questions.
        subset_size: Optional[int]
            If set, selects a smaller subset of the dataset.

    Returns:
        Dict[str, Dataset]
            Tokenized train and validation datasets.
    """
    dataset = load_dataset("glue", "qqp")

    if subset_size:
        # Select a smaller subset of the dataset and shuffle it
        dataset = {
            split: dataset[split].shuffle(seed=42).select(range(subset_size))
            for split in ["train", "validation"]
        }

    tokenized_dataset: Dict[str, Dataset] = {}

    for split in ["train", "validation"]:
        # Tokenize the dataset using the provided tokenizer
        tokenized_dataset[split] = dataset[split].map(
            tokenize_examples, batched=True, fn_kwargs={"tokenizer": tokenizer}
        )

    return tokenized_dataset


def main() -> None:
    """
    Main function to train and evaluate the paraphrase detection model.

    The model is trained on the QQP dataset and the best model is saved in the scratch filesystem.
    The evaluation results are saved in a CSV file in the current directory.

    Parameters:
        None

    Returns:
        None
    """
    args = parse_args()

    # Automatically set device to 'cuda' if available, else 'cpu'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "no_paraphrase", 1: "paraphrase"},
        label2id={"no_paraphrase": 0, "paraphrase": 1},
    ).to(device)

    # Load and tokenize dataset
    tokenized_datasets: Dict[str, Dataset] = load_and_tokenize_dataset(tokenizer)

    # Path to save models in the scratch filesystem
    output_dir: str = f"./out/cls-models/{args.model_name.split('/')[-1]}_qqp_pd"

    trainer: Trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=True,
            save_total_limit=1,
        ),
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_binary_metrics,
    )

    # Train and evaluate model
    trainer.train()
    print(f"Best model is saved at: {trainer.args.output_dir}")

    results: Dict[str, float] = trainer.evaluate(
        eval_dataset=tokenized_datasets["validation"]
    )

    # Save and print the location of the test set evaluation results
    results_file: str = (
        f"./out/cls-models/{args.model_name.split('/')[-1]}_qqp_pd_results.csv"
    )
    pd.DataFrame([results]).to_csv(results_file, index=False)
    print(f"Test set evaluation results are saved at: {results_file}")


if __name__ == "__main__":
    main()
