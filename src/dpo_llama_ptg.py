import argparse
import logging
import os
from typing import Optional, Tuple

import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import DPOConfig, DPOTrainer
from sklearn.model_selection import train_test_split
import pandas as pd

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
        default="meta-llama/Llama-3.1-8B",
        help="Path to the model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc",
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
    login(token=hf_token, add_to_git_credential=True)


def clean_text(text):
    """
    Clean the text to fix encoding issues.

    Args:
        text (str): The original text that may contain encoding issues.

    Returns:
        str: The cleaned text.
    """
    replacements = {
        "Ãƒâ€šÃ‚Â½": "½",  # One-half
        "Ãƒâ€šÃ‚Â¼": "¼",  # One-quarter
        "Ãƒâ€šÃ‚Â©": "©",  # Copyright
        "Ãƒâ€šÃ‚Â¢": "¢",  # Cent
        "Ãƒâ€šÃ‚Â¡": "¡",  # Inverted exclamation
        "Ãƒâ€šÃ‚Â¿": "¿",  # Inverted question
        "ÃƒÂ¢Ã‚â‚¬": "–",  # En dash
        "ÃƒÂ¢Ã‚â„¢": "’",  # Right single quotation
        # Catch-all for any occurrence of Ãƒâ€¦
        "Ãƒâ€š": "",  # Remove leading misencoded sequences
        "Ãƒ": "",  # Remove leading misencoded sequences
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    return text


def preprocess_apty_ranked_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the APTY-ranked dataset by normalizing columns, extracting text, and cleaning data.

    Args:
        data (pd.DataFrame): The raw dataframe loaded from the dataset.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Normalize the 'meta' column to create separate columns for 'id', 'annotators', and 'APT'
    meta_df = pd.json_normalize(data["meta"])
    data = data.drop(columns=["meta"]).reset_index(drop=True)
    data = pd.concat([data, meta_df], axis=1)

    # Extract the text from the nested dictionaries for 'chosen' and 'rejected'
    data["original"] = data["original"].apply(
        lambda x: x["text"] if isinstance(x, dict) else str(x)
    )
    data["chosen"] = data["chosen"].apply(
        lambda x: x["text"] if isinstance(x, dict) else str(x)
    )
    data["rejected"] = data["rejected"].apply(
        lambda x: x["text"] if isinstance(x, dict) else str(x)
    )

    # Clean the text to fix encoding issues
    data["original"] = data["original"].apply(clean_text)
    data["chosen"] = data["chosen"].apply(clean_text)
    data["rejected"] = data["rejected"].apply(clean_text)

    # Strip whitespace
    data["original"] = data["original"].str.strip()
    data["chosen"] = data["chosen"].str.strip()
    data["rejected"] = data["rejected"].str.strip()

    # Modify the last character according to the rules
    data["original"] = data["original"].apply(modify_last_character)
    data["chosen"] = data["chosen"].apply(modify_last_character)
    data["rejected"] = data["rejected"].apply(modify_last_character)

    # Drop duplicates
    data = data.drop_duplicates(subset=["original", "chosen", "rejected"])
    # Generate train/eval splits
    train_df, eval_df = train_test_split(
        data, test_size=0.2, stratify=data["APT"], random_state=42
    )

    # Create prompt datasets
    def create_dataset(df):
        prompts = [
            {
                "prompt": (
                    f"Instruction: Given the following sentence, generate a paraphrase with the following types. "
                    f"Sentence: {row['original']} \n "
                    f"Paraphrase Types: {row['APT']}\n\n"
                    f"Answer: "
                ),
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for _, row in df.iterrows()
        ]
        return Dataset.from_list(prompts)

    datasets = {"train": create_dataset(train_df), "eval": create_dataset(eval_df)}

    return datasets


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
        text += "."  # Add a '.' if the last character is a letter

    return text


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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", padding=True, truncation=True
    )

    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    logging.info(f"Loading PEFT adapter from {adapter_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        quantization_config=bnb_config,
        is_trainable=True,
    )
    model.load_adapter(adapter_dir, adapter_name="reference")

    model = model.to(device)

    return model, tokenizer


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
        per_device_train_batch_size=1,  # Number of samples per batch on each device
        gradient_accumulation_steps=4,  # Number of batches to accumulate gradients for
        output_dir=output_dir,  # Directory to save the model to
        max_prompt_length=350,
        max_length=512,
        fp16=True,
        remove_unused_columns=False,
        loss_type=loss_type,  # The type of loss to use
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=3,
        save_total_limit=1,
        weight_decay=0.01,
        model_adapter_name="default",
        ref_adapter_name="reference",
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

    sanitized_model_name = args.model_name.split("/")[-1]
    output_dir = f"./out/gen-models/{sanitized_model_name}-paraphrase-type-generation-apty-{args.loss_type}"
    os.makedirs(output_dir, exist_ok=True)

    login_to_huggingface("token_file.txt")

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
    torch.cuda.empty_cache()

    # Load and preprocess APTY-ranked dataset
    logging.info("Loading and preprocessing APTY-ranked dataset")
    dataset = load_dataset("worta/apty", "APTY-ranked")
    datasets = preprocess_apty_ranked_dataset(pd.DataFrame(dataset["train"]))

    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]

    torch.cuda.empty_cache()

    trainer = setup_dpo_trainer(
        model, tokenizer, train_dataset, eval_dataset, output_dir, args.loss_type
    )

    torch.cuda.empty_cache()
    trainer.train()

    torch.cuda.empty_cache()

    model.delete_adapter("reference")

    trainer.push_to_hub(
        f"{sanitized_model_name}-paraphrase-type-generation-apty-{args.loss_type}"
    )


if __name__ == "__main__":
    main()
