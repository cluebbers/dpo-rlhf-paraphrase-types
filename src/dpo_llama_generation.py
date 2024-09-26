"""
python src/dpo_generation.py \
 --model_name=meta-llama/Llama-2-7b-hf \
 --adapter_dir=llama/llama-7b-etpc
"""

import argparse
import os

import torch
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model
from trl.commands.cli_utils import DPOScriptArguments, TrlParser
from trl import (
    DPOTrainer, 
    DPOConfig,
    ModelConfig,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login
with open("token_file.txt", "r") as token_file:
    hf_token = token_file.read().strip()
login(token=hf_token)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the model")
    parser.add_argument("--adapter_dir", type=str, default="llama/llama-7b-etpc", help="Directory of the PEFT adapter")
    return parser.parse_args()   

def main():    
    args = parse_arguments()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
     
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT adapters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, args.adapter_dir)
    
    peft_config = PeftConfig.from_pretrained(adapter_dir, base_model_name_or_path=args.model_name)
    model = PeftModel.from_pretrained(model, adapter_dir)
    
    # Load reference adapter for DPO
    model.load_adapter(adapter_dir, adapter_name="reference")
    
    # Apply additional LoRA weights
    model = get_peft_model(model, peft_config=peft_config)
    
    # Move model to GPU again after loading PEFT model
    model = model.to(device)
    
    torch.cuda.empty_cache()  # Clear GPU cache before starting
                       
    # Load dataset from JSONL files
    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    
    train_dataset = load_dataset('json', data_files={'train': train_json_path})['train']
    validation_dataset = load_dataset('json', data_files={'validation': eval_json_path})['validation']
    ds=  DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
    })
    
    # Prepare datasets for training
    train_dataset = ds['train']
    eval_dataset = ds['validation']

    # Setup trainer    
    training_args=DPOConfig(output_dir=f"./out/gen-models/dpo_{args.model_name}",
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=1,
                            max_length=1024,
                            max_prompt_length=512,
                            fp16=True,
                            optim="adamw_8bit",        
                            )
    
    trainer = DPOTrainer(model,
                         args=training_args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         peft_config=peft_config,
                         )

    # Start training
    trainer.train()

    # Save the trained model
    trainer.save_model("./out/gen-models/dpo_{args.model_name}")

if __name__ == "__main__":
    main()
