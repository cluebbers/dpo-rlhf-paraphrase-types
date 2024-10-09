import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate

def tokenize_fn(examples, sentence1_key, sentence2_key, tokenizer):
    """
    Tokenizes input examples using a tokenizer.
    
    Args:
        examples (dict): The input examples.
        sentence1_key (str): The key for the first sentence in the examples.
        sentence2_key (str, optional): The key for the second sentence in the examples.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.

    Returns:
        dict: The tokenized inputs.
    """
    tokenized_inputs = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        padding="max_length",
        truncation=True,
        max_length=512 # Limit max length for faster training
    )
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

def parse_args():
    """
    Parses command line arguments.
    
    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-base",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Hardware to use, either of 'cpu', 'cuda', or 'mps'",
    )
    return parser.parse_args()

def compute_metrics_binary(eval_pred):
    """
    Computes accuracy, precision, recall, and F1 metrics for binary classification.

    Args:
        eval_pred: The predictions and labels from the model.

    Returns:
        dict: The calculated accuracy, precision, recall, and F1 score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    accuracy_score = accuracy.compute(predictions=predictions, references=labels)
    precision_score = precision.compute(predictions=predictions, references=labels, average="binary")
    recall_score = recall.compute(predictions=predictions, references=labels, average="binary")
    f1_score = f1.compute(predictions=predictions, references=labels, average="binary")
    
    return {
        "accuracy": accuracy_score["accuracy"],
        "precision": precision_score["precision"],
        "recall": recall_score["recall"],
        "f1": f1_score["f1"],
    }

def main():
    args = parse_args()

    # Load the QQP dataset
    dataset = load_dataset("glue", "qqp")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,  # Binary classification
        id2label={0: "no_paraphrase", 1: "paraphrase"},
        label2id={"no_paraphrase": 0, "paraphrase": 1},
    ).to(args.device)

    # Define sentence keys for QQP
    sentence1_key = "question1"
    sentence2_key = "question2"

    # Tokenize the dataset
    dataset_tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={"sentence1_key": sentence1_key, "sentence2_key": sentence2_key, "tokenizer": tokenizer},
    )

    # Define data collator with dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # Train/test split (QQP already has train/validation/test splits)
    train_dataset = dataset_tokenized["train"]
    val_dataset = dataset_tokenized["validation"]

    # Define training arguments
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{args.model_name}_qqp_pd",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model at the end of each epoch to match evaluation
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        logging_dir='./logs',
        load_best_model_at_end=True,  # Load the best model after training
        fp16=True,  # Use mixed precision training
    )


    # Create Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_binary,
    )

    # Training
    print("Starting training...")
    trainer.train()
    # Save the best model after training
    trainer.save_model(output_dir=f"./out/cls-models/{args.model_name}_qqp_pd_best")


    # Evaluation on validation set
    print("Evaluating on validation set...")
    results = trainer.evaluate()
    print("#" * 20)
    print(args.model_name)
    print("paraphrase-detection")
    print(results)
    print("#" * 20)

    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{args.model_name}-qqp-pd-results.csv", index=False)

if __name__ == "__main__":
    main()
