import argparse
import logging
import torch
import os
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Constants
DEFAULT_OUTPUT_DIR = "./out/gen-models"

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for running DPO training.

    The following arguments can be specified:

    --model_name (str): Path to the model to use for training. Defaults to "meta-llama/Llama-2-7b-hf".
    --adapter_dir (str): Name of the PEFT adapter to use for training. Defaults to "out/gen-models/llama-7b-etpc".
    --loss_type (str): Type of loss to use for training. Defaults to "sigmoid".
    """
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to the model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="out/gen-models/llama-7b-etpc",
        help="Name of the PEFT adapter",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        help="Loss type",
    )
    return parser.parse_args()


def login_to_huggingface(token_path="token_file.txt"):
    """Login to Hugging Face using a token stored in a file.

    Args:
        token_path (str): Path to the file containing the Hugging Face token.

    Returns:
        None
    """
    logging.info(f"Logging into Hugging Face Hub using token from {token_path}")
    with open(token_path, "r") as token_file:
        hf_token = token_file.read().strip()
    login(token=hf_token)
    logging.info("Logged into Hugging Face Hub")   
    
     
def load_model_and_tokenizer(model_name, adapter_dir, device):
    """
    Load the model, tokenizer, and the PEFT adapter.

    This function loads the specified model and tokenizer, and then applies the PEFT adapter.
    It also sets the model to training mode and moves it to the specified device.

    Args:
        model_name (str): Name of the model to load.
        adapter_dir (str): Path to the PEFT adapter.
        device (torch.device): Device to move the model to.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    logging.info(f"Loading model and tokenizer for {model_name}")
    
    # Configure for 4-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and attach PEFT adapter
    logging.info(f"Loading PEFT adapter from {adapter_dir}")
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    peft_config.base_model_name_or_path = model_name
    model = PeftModel.from_pretrained(model, adapter_dir, config=peft_config)

    # Load the reference adapter
    model.load_adapter(adapter_dir, adapter_name="reference")

    # Set the model to training mode and move it to the specified device
    model.train()
    model.to(device)
    torch.cuda.empty_cache()

    return model, tokenizer


def load_datasets(train_file: str, eval_file: str) -> DatasetDict:
    """Load training and validation datasets from JSON files.

    Args:
        train_file (str): Path to the training dataset JSON file.
        eval_file (str): Path to the validation dataset JSON file.

    Returns:
        DatasetDict: A dictionary containing the training and validation datasets.
    """
    logging.info(f"Loading datasets: {train_file} and {eval_file}")
    
    # Load the datasets from JSON files
    train_dataset = load_dataset('json', data_files={'train': train_file})['train']
    validation_dataset = load_dataset('json', data_files={'validation': eval_file})['validation']
    
    # Log the number of samples in each dataset
    logging.info(f"Loaded {len(train_dataset)} training samples and {len(validation_dataset)} validation samples.")
    
    # Return the datasets as a dictionary
    return DatasetDict({'train': train_dataset, 'validation': validation_dataset})


def setup_dpo_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir, loss_type):
    """Set up the DPO trainer.

    This function sets up the DPO trainer with the given model, tokenizer, training dataset, evaluation dataset, output directory, and loss type.

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
    logging.info("Setting up the DPO trainer...")

    # Set up the training arguments
    training_args = DPOConfig(
        eval_strategy="epoch",  # Evaluate the model at the end of each epoch
        per_device_train_batch_size=8,  # Number of samples per batch on each device
        gradient_accumulation_steps=4,  # Number of batches to accumulate gradients for
        output_dir=output_dir,  # Directory to save the model to
        max_length=1024,  # Maximum length of the input sequences
        max_prompt_length=512,  # Maximum length of the prompts
        fp16=True,  # Use mixed precision training
        remove_unused_columns=False,  # Explicitly set this to avoid the warning
        loss_type=loss_type,  # The type of loss to use
    )

    # Set up the DPO trainer
    return DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    
def main():
    """Main function to run the DPO training process.

    This function parses command-line arguments, loads the model and tokenizer, loads the datasets, sets up the DPO trainer, trains the model, and saves the trained model.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Create the output directory if it does not exist
    sanitized_model_name = args.model_name.replace('/', '-')
    output_dir = f"{DEFAULT_OUTPUT_DIR}/dpo_{sanitized_model_name}_{args.loss_type}"
    os.makedirs(output_dir, exist_ok=True)

    # Login to Hugging Face Hub
    login_to_huggingface()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_dir, device)

    # Load datasets
    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    datasets = load_datasets(train_json_path, eval_json_path)
    train_dataset = datasets['train']
    eval_dataset = datasets['validation']

    # Setup DPO trainer
    trainer = setup_dpo_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir, args.loss_type)

    # Train the model
    logging.info("Starting DPO training...")
    trainer.train()

    # Save the trained model
    logging.info("Training completed. Saving the model...")
    trainer.save_model(output_dir)
    logging.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
