"""
python src/run_dpo.py \
 --model_name_or_path=meta-llama/Llama-2-7b-hf \
 --output_dir="dpo_llama_apty_output" 
"""

import argparse
import logging
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, PeftConfig
from trl import DPOTrainer, RichProgressCallback
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

# Patch the DPOTrainer with Unsloth optimizations
PatchDPOTrainer()

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login
login(new_session=False)

# Clear GPU cache before starting
torch.cuda.empty_cache()

# Enable rich logging if TRL_USE_RICH is set
from distutils.util import strtobool
TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))
if TRL_USE_RICH:
    from rich.console import Console
    from rich.logging import RichHandler
    logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)
    console = Console()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduling")
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    parser.add_argument("--sanity_check", action="store_true", help="Run a sanity check by limiting the dataset size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for DPOTrainer")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load base model and tokenizer using FastLanguageModel
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Load PEFT adapters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, "llama", "llama-7b-etpc")
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)

    # Load reference adapter for DPO
    model.load_adapter(adapter_dir, adapter_name="reference")

    # Apply additional LoRA weights
    model = FastLanguageModel.get_peft_model(model, **vars(peft_config))
    
    # Context manager setup for rich progress bars
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = nullcontext() if not TRL_USE_RICH else console.status(f"[bold green]Training completed! Saving the model to {args.output_dir}")

    # Load dataset
    dataset_path = os.path.join(script_dir, "data", "apty_dataset.parquet")
    ds = load_dataset("parquet", data_files=dataset_path)
    
    # Optionally limit dataset size for a sanity check
    if args.sanity_check:
        ds = ds.map(lambda x: x.select(range(50)))

    # Split dataset into train and validation sets
    train_test_split = ds['train'].train_test_split(test_size=0.2)
    ds = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test'],
    })

    # Prepare datasets for training
    train_dataset = ds['train']
    eval_dataset = ds['validation']

    # Training setup with DPOTrainer
    with init_context:
        trainer = DPOTrainer(
            model,
            args=TrainingArguments(
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_ratio=args.warmup_ratio,
                num_train_epochs=args.num_train_epochs,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=args.logging_steps,
                optim=args.optim,
                seed=args.seed,
                output_dir=args.output_dir,
            ),
            beta=args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    # Start training
    trainer.train()

    # Save the trained model
    with save_context:
        trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
