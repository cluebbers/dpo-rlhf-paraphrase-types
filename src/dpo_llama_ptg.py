import argparse
import logging
import os
from typing import Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for running DPO training.

    The following arguments can be specified:

    --model_name (str): Path to the model to use for training. Defaults to "out/gen-models/llama-3.1-8b-etpc".
    --adapter_dir (str): Name of the PEFT adapter to use for training. Defaults to None.
    --loss_type (str): Type of loss to use for training. Defaults to "sigmoid".

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="out/gen-models/llama-3.1-8b-etpc",
        help="Path to the model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
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

    if args.adapter_dir == "None":
        args.adapter_dir = None

    return args


def login_to_huggingface(token_path=None):
    """
    Login to the Hugging Face Hub using either the `HF_TOKEN` environment variable or a token file.

    Args:
        token_path (str, optional): Path to the file containing the Hugging Face token.

    Returns:
        None
    """
    # Check if the HF_TOKEN environment variable is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token and token_path:
        # If not, read the token from the file
        with open(token_path, "r") as token_file:
            hf_token = token_file.read().strip()

    # Login to the Hugging Face Hub
    login(token=hf_token)


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
    logging.info(f"Loading model and tokenizer for {model_name}")

    # Configure for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B", padding_side="left", padding="longest"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load adapter if specified, otherwise check or add LoRA adapter
    if adapter_dir:
        logging.info(f"Loading PEFT adapter from {adapter_dir}")
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        peft_config.base_model_name_or_path = model_name
        model = PeftModel.from_pretrained(
            model, adapter_dir, config=peft_config, is_trainable=True
        )
        model.load_adapter(adapter_dir, adapter_name="reference")
    elif not hasattr(model.config, "peft_type") or model.config.peft_type is None:
        logging.info(
            "Adding a new LoRA adapter as no adapter_dir is provided and no merged adapter found."
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
    else:
        logging.info("Model is already merged with the adapter configuration.")

    # If using 4-bit quantization, do not manually move the model to the device
    if not bnb_config.load_in_4bit:
        model.to(device)

    return model, tokenizer


def load_datasets(train_file: str, eval_file: str) -> Tuple[Dataset, Dataset]:
    """
    Load training and validation datasets from JSON files.

    Args:
        train_file (str): Path to the training dataset JSON file.
        eval_file (str): Path to the validation dataset JSON file.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
    """

    # Load the datasets from JSON files
    train_dataset = load_dataset("json", data_files={"train": train_file})["train"]
    validation_dataset = load_dataset("json", data_files={"validation": eval_file})[
        "validation"
    ]

    # Return the datasets as a tuple
    return train_dataset, validation_dataset


def setup_dpo_trainer(
    model, tokenizer, train_dataset, eval_dataset, output_dir, loss_type
):
    """
    Set up the DPO trainer with the given model, tokenizer, training dataset, evaluation dataset, output directory, and loss type.

    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        train_dataset (DatasetDict): The training dataset.
        eval_dataset (DatasetDict): The evaluation dataset.
        output_dir (str): The directory to save the model to.
        loss_type (str): The type of loss to use. Can be either 'kl' or 'js'.

    Returns:
        DPOTrainer: The set up DPO trainer.
    """
    # Set up the training arguments
    training_args = DPOConfig(
        eval_strategy="epoch",  # Evaluate the model at the end of each epoch
        per_device_train_batch_size=8,  # Number of samples per batch on each device
        gradient_accumulation_steps=4,  # Number of batches to accumulate gradients for
        output_dir=output_dir,  # Directory to save the model to
        max_length=1024,
        max_prompt_length=512,
        fp16=True,
        remove_unused_columns=False,
        loss_type=loss_type,  # The type of loss to use
        save_strategy="best",
        load_best_model_at_end=True,
        num_train_epochs=10,
    )

    # Set up the DPO trainer
    return DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )


def main() -> None:
    """Main function to run the DPO training process.

    This function parses command-line arguments, loads the model and tokenizer, loads the datasets, sets up the DPO trainer, trains the model, and saves the trained model.
    """
    args = parse_arguments()

    sanitized_model_name = args.model_name.replace("/", "-")
    output_dir = f"./out/gen-models/dpo_{sanitized_model_name}_{args.loss_type}"
    os.makedirs(output_dir, exist_ok=True)

    login_to_huggingface()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.adapter_dir, device
    )
    torch.cuda.empty_cache()

    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    train_dataset, eval_dataset = load_datasets(train_json_path, eval_json_path)
    torch.cuda.empty_cache()

    trainer = setup_dpo_trainer(
        model, tokenizer, train_dataset, eval_dataset, output_dir, args.loss_type
    )

    logging.info("Starting DPO training...")
    torch.cuda.empty_cache()
    trainer.train()
    torch.cuda.empty_cache()

    trainer.save_model(output_dir)
    logging.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
