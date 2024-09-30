import argparse
import os
import logging
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Constants
DEFAULT_OUTPUT_DIR = "./out/gen-models"
DEFAULT_LOG_DIR = "slurm_files/my_app.log"

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the model")
    parser.add_argument("--adapter_dir", type=str, default="llama/llama-7b-etpc", help="Directory of the PEFT adapter")
    return parser.parse_args()

def login_to_huggingface(token_path="token_file.txt"):
    """Login to Hugging Face using a token stored in a file."""
    with open(token_path, "r") as token_file:
        hf_token = token_file.read().strip()
    login(token=hf_token)
    logging.info("Logged into Hugging Face Hub")

def load_model_and_tokenizer(model_name, adapter_dir, device):
    """Load the model, tokenizer, and the PEFT adapter."""
    logging.info(f"Loading model and tokenizer for {model_name}")
    
    # Configure for 4-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and attach PEFT adapter
    logging.info(f"Loading PEFT adapter from {adapter_dir}")
    adapter_dir_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), adapter_dir)
    peft_config = PeftConfig.from_pretrained(adapter_dir_abs)
    peft_config.base_model_name_or_path = model_name
    model = PeftModel.from_pretrained(model, adapter_dir_abs, config=peft_config)

    # Load the reference adapter
    model.load_adapter(adapter_dir_abs, adapter_name="reference")

    model.to(device)
    model.train()
    torch.cuda.empty_cache()

    return model, tokenizer

def load_datasets(train_file, eval_file):
    """Load training and validation datasets."""
    logging.info(f"Loading datasets: {train_file} and {eval_file}")
    
    train_dataset = load_dataset('json', data_files={'train': train_file})['train']
    validation_dataset = load_dataset('json', data_files={'validation': eval_file})['validation']
    
    logging.info(f"Loaded {len(train_dataset)} training samples and {len(validation_dataset)} validation samples.")
    
    return DatasetDict({'train': train_dataset, 'validation': validation_dataset})

def setup_dpo_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir):
    """Setup the DPO trainer."""
    logging.info("Setting up the DPO trainer...")
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_length=1024,
        max_prompt_length=512,
        fp16=True,
        optim="adamw_8bit",
        remove_unused_columns=False  # Explicitly set this to avoid the warning
    )
    
    return DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )


def main():
    """Main function to run the DPO training process."""
    # Parse command-line arguments
    args = parse_arguments()
    output_dir = f"{DEFAULT_OUTPUT_DIR}/dpo_{args.model_name}_{args.adapter_dir}"

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
    trainer = setup_dpo_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir)

    # Train the model
    logging.info("Starting DPO training...")
    trainer.train()

    # Save the trained model
    logging.info("Training completed. Saving the model...")
    trainer.save_model(output_dir)
    logging.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
