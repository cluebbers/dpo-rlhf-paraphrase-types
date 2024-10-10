import argparse
import numpy as np
import pandas as pd
import torch  
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate

# Global dictionary for all metrics
metrics = {
    "accuracy": evaluate.load("accuracy"),
    "precision": evaluate.load("precision"),
    "recall": evaluate.load("recall"),
    "f1": evaluate.load("f1")
}

def tokenize_examples(examples, tokenizer):
    """
    Tokenizes input examples using the provided tokenizer.

    Args:
        examples (dict): A dictionary with keys 'question1', 'question2', and 'label'.
        tokenizer (Tokenizer): The tokenizer used for encoding the questions.

    Returns:
        dict: Tokenized examples with the corresponding labels.
    """
    return {
        **tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True, max_length=256),
        "labels": examples["label"]
    }

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a paraphrase detection model.")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base", help="Model name from Hugging Face hub.")
    return parser.parse_args()

def compute_binary_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1 score for binary classification.

    Args:
        eval_pred (tuple): Model predictions and corresponding labels.

    Returns:
        dict: Computed accuracy, precision, recall, and F1 score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {metric: metrics[metric].compute(predictions=predictions, references=labels)[metric]
            for metric in ["accuracy", "precision", "recall", "f1"]}

def load_and_tokenize_dataset(tokenizer, subset_size=None):
    """
    Loads and tokenizes the QQP dataset.

    Args:
        tokenizer (Tokenizer): The tokenizer used to encode questions.
        subset_size (int, optional): If set, selects a smaller subset of the dataset.

    Returns:
        dict: Tokenized train, validation, and test datasets.
    """
    dataset = load_dataset("glue", "qqp")
    
    if subset_size:
        dataset = {split: dataset[split].shuffle(seed=42).select(range(subset_size)) for split in ["train", "validation", "test"]}
    
    return {
        split: dataset[split].map(tokenize_examples, batched=True, fn_kwargs={"tokenizer": tokenizer})
        for split in ["train", "validation", "test"]
    }

def main():
    """
    Main function to train and evaluate the paraphrase detection model.
    """
    args = parse_args()

    # Automatically set device to 'cuda' if available, else 'cpu'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label={0: "no_paraphrase", 1: "paraphrase"}, label2id={"no_paraphrase": 0, "paraphrase": 1}
    ).to(device)

    # Load and tokenize dataset
    tokenized_datasets = load_and_tokenize_dataset(tokenizer)

    # Define Trainer and training arguments
    output_dir = f"./out/{args.model_name.split('/')[-1]}_qqp_pd"
    trainer = Trainer(
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
            logging_dir="./logs",
            load_best_model_at_end=True,
            fp16=True,
        ),
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_binary_metrics,
    )

    # Train and evaluate model
    trainer.train()
    results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

    # Save results
    pd.DataFrame([results]).to_csv(f"{output_dir}_results.csv", index=False)

if __name__ == "__main__":
    main()
