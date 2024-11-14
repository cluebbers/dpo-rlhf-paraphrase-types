#!/usr/bin/env python3

import argparse
import logging
import os
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import DPOConfig, DPOTrainer

from common import login_to_huggingface, preprocess_apty_ranked_dataset


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for running DPO training.

    The following arguments can be specified:

    --model_name (str):     Path to the model to use for training. Defaults to "meta-llama/Llama-3.1-8B".
    --adapter_dir (str):    Name of the PEFT adapter to use for training. Defaults to "cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc".
    --loss_type (str):      Type of loss to use for training. Defaults to "sigmoid".
    """
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Path to the model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="out/gen-models/Llama-3.1-8B-etpc",
        help="Name of the PEFT adapter",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["sigmoid", "ipo"],
        default="sigmoid",
        help="Loss type",
    )
    args = parser.parse_args()

    if args.adapter_dir.lower() == "none":
        args.adapter_dir = None

    return args


def load_model_and_tokenizer(
    model_name: str, adapter_dir: Optional[str], device: torch.device
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a pre-trained model and tokenizer from Hugging Face Hub.

    Args:
        model_name (str): Name of the model to load.
        adapter_dir (str, optional): Path to the PEFT adapter directory.
        device (torch.device): Device to move the model to.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizerBase]: A tuple containing the loaded model and tokenizer.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        padding=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Loading {model_name} and adding PEFT adapter {adapter_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=True,
        )

        model.load_adapter(adapter_dir, adapter_name="reference")

    model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(device)

    return model, tokenizer


def main() -> None:
    """Main function to run the DPO training process.

    This function parses command-line arguments, loads the model and tokenizer, loads the datasets, sets up the DPO trainer, trains the model, and saves the trained model.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()

    sanitized_model_name = f"{args.model_name.split('/')[-1]}-paraphrase-type-generation-apty-{args.loss_type}"
    output_dir = f"./out/gen-models/{sanitized_model_name}"
    os.makedirs(output_dir, exist_ok=True)

    login_to_huggingface()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        logging.info(
            f"Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}"
        )
    else:
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.adapter_dir, device
    )

    logging.info("Loading and preprocessing APTY-ranked dataset")
    dataset = load_dataset("worta/apty", "APTY-ranked")
    datasets = preprocess_apty_ranked_dataset(dataset["train"])

    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]

    common_training_args = {
        "ref_adapter_name": "reference",
        "model_adapter_name": "default",
        "remove_unused_columns": False,
        "eval_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_rewards/accuracies",
        "greater_is_better": True,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "output_dir": output_dir,
        "max_length": 1024,
        "max_prompt_length": 512,
        "bf16": True,
        "report_to": "tensorboard",
        "optim": "adamw_8bit",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "beta": 0.2,
    }

    if args.loss_type == "sigmoid":
        specific_training_args = {
            "num_train_epochs": 3,
            "warmup_ratio": 0.3,
            "weight_decay": 2e-1,
            "learning_rate": 6e-5,
            "lr_scheduler_type": "linear",
        }
    else:  # IPO
        specific_training_args = {
            "num_train_epochs": 3,
            "warmup_ratio": 0.2,
            "weight_decay": 0.02,
            "learning_rate": 5e-6,
            "lr_scheduler_type": "reduce_lr_on_plateau",
        }

    training_args = DPOConfig(
        loss_type=args.loss_type,
        **common_training_args,
        **specific_training_args,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    torch.cuda.empty_cache()
    trainer.train()

    logging.info(f"Saving model to {output_dir}")
    # reference adapter is only needed for training
    model.delete_adapter("reference")

    trainer.save_model(output_dir)
    logging.info(f"Pushing model to {sanitized_model_name}")
    trainer.push_to_hub(sanitized_model_name)


if __name__ == "__main__":
    main()
