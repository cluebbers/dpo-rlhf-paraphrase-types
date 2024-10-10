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

def tokenize_fn(examples, tokenizer):
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
        examples["question1"],
        examples["question2"],
        padding="max_length",
        truncation=True,
        max_length=256 # Limit max length for faster training
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
    
    # Access global metrics
    accuracy_score = accuracy_metric.compute(predictions=predictions, references=labels)
    precision_score = precision_metric.compute(predictions=predictions, references=labels, average="binary")
    recall_score = recall_metric.compute(predictions=predictions, references=labels, average="binary")
    f1_score = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    
    return {
        "accuracy": accuracy_score["accuracy"],
        "precision": precision_score["precision"],
        "recall": recall_score["recall"],
        "f1": f1_score["f1"],
    }

def main():
    args = parse_args()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,  # Binary classification
        id2label={0: "no_paraphrase", 1: "paraphrase"},
        label2id={"no_paraphrase": 0, "paraphrase": 1},
    ).to(args.device)

    # Load the QQP dataset
    dataset = load_dataset("glue", "qqp")
        
    # Select a smaller subset of the dataset
    train_dataset = dataset["train"] #.shuffle(seed=42).select(range(100000))  # Select 10,000 samples for training
    val_dataset = dataset["validation"]#.shuffle(seed=42).select(range(10000))  # Select 1,000 samples for validation
    test_dataset = dataset["test"]#.shuffle(seed=42).select(range(10000))  # Select 1,000 samples for validation

    # Tokenize the subsetted datasets
    train_dataset_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )
    val_dataset_tokenized = val_dataset.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )
    test_dataset_tokenized = test_dataset.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # Define data collator with dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # Define training arguments
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{args.model_name.split('/')[-1]}_qqp_pd",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model at the end of each epoch to match evaluation
        learning_rate=2e-5,
        logging_steps=100000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        dataloader_num_workers=4,
        logging_dir='./logs',
        load_best_model_at_end=True,  # Load the best model after training
        fp16=True,  # Use mixed precision training
    )


    # Create Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_binary,
    )

    # Training
    print("Starting training...")
    trainer.train()

    # Evaluation on validation set
    print("Evaluating on validation set...")
    results = trainer.evaluate(eval_dataset=test_dataset_tokenized)
    print("#" * 20)
    print(args.model_name)
    print("paraphrase-detection")
    print(results)
    print("#" * 20)

    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"out/cls-models/{args.model_name.split('/')[-1]}-qqp-pd-results.csv", index=False)

if __name__ == "__main__":
    # Load metrics globally once
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    main()
