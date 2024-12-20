import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BartForConditionalGeneration
from trl import DPOConfig, DPOTrainer

from common import preprocess_apty_ranked_dataset


def parse_args():
    """
    Parses the command line arguments and returns the parsed arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    Example:
        ```python
        args = parse_args()
        print(args.model_name)
        ```"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="cluebbers/bart-large-paraphrase-type-generation-etpc",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="paraphrase-type-generation",
        help="paraphrase-type-generation or paraphrase-generation",
    )
    parser.add_argument(
        "--loss_type", type=str, default="sigmoid", help="sigmoid or ipo"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()  # Clear GPU cache before starting

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)  # Move model to GPU

    # Load training and evaluation datasets
    dataset = load_dataset("worta/apty", "APTY-ranked")
    datasets = preprocess_apty_ranked_dataset(dataset["train"])

    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]


    # Set up training arguments
    training_args = DPOConfig(
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        loss_type=args.loss_type,
        remove_unused_columns=False,  # Ensures unused columns aren't removed for DPOTrainer
        fp16=True,  # Enable mixed precision training
        output_dir=f"./out/gen-models/{args.model_name}_{args.task_name}_{args.loss_type}",
        max_length=1024,
        max_prompt_length=512,
        max_target_length=512,
    )

    # Set up trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
