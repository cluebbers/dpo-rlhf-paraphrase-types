# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python src/run_dpo.py `
    --dataset_name=data/apty_dataset.parquet `
    --model_name_or_path=llama-7b `
    --per_device_train_batch_size 4 `
    --learning_rate 1e-3 `
    --gradient_accumulation_steps 1 `
    --logging_steps 10 `
    --eval_steps 10 `
    --output_dir="dpo_llama_etpy_output" `
    --warmup_steps 10 `
    --report_to wandb `
    --bf16 `
    --logging_first_step `
    --no_remove_unused_columns

# peft:
python src/run_dpo.py \
    --dataset_name=/home/slim/Documents/06_DataScience/06_Projects/M.Inf.2901_Master/emnlp23-paraphrase-types-dpo/data/apty_dataset.parquet \
    --model_name_or_path=meta-llama/Llama-2-7b-hf \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_steps 10 \
    --output_dir="dpo_llama_apty_output" \
    --optim rmsprop \
    --warmup_steps 10 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16


"""

import logging
import multiprocessing
import os
import argparse
from contextlib import nullcontext

from trl.commands.cli_utils import init_zero_verbose, TrlParser
from distutils.util import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler



import torch
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments

from peft import PeftModel, PeftConfig

from trl import (
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()

from huggingface_hub import login
login(new_session=False)

torch.cuda.empty_cache()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
        # Argument parsing
        
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--dataset_name", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation steps")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--optim", type=str, default="rmsprop", help="Optimizer")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool")
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    parser.add_argument("--logging_first_step", action="store_true", help="Log first step")
    parser.add_argument("--remove_unused_columns", action="store_true", help="Do not remove unused columns")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    # long list because DPOConfig etc not working in trl<0.9
    parser.add_argument("--torch_dtype", type=str, default="auto", help="Torch dtype, e.g., 'auto', 'float16', 'bfloat16'")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model with 4-bit quantization") 
    parser.add_argument("--load_in_8bit", action="store_false", help="Load model with 8-bit quantization") 
    parser.add_argument("--model_revision", type=str, default="main", help="Model revision to use")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Attention implementation to use, if any")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow to trust remote code when loading models")
    parser.add_argument("--ignore_bias_buffers", action="store_true", help="Ignore bias buffers in the model")
    parser.add_argument("--sanity_check", action="store_true", help="Run a sanity check by limiting the dataset size")
    parser.add_argument("--batch_eval_metrics", action="store_true", help="Enable batch evaluation of metrics")
    parser.add_argument("--full_determinism", action="store_true", help="Enable full determinism for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--accelerator_config", type=str, help="Path to accelerate config file")

    
    args = parser.parse_args()
    
    # Force use our print callback
    if TRL_USE_RICH:
        args.disable_tqdm = True
        console = Console()

    ################
    # Regular Model & Tokenizer
    ################
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    
    # Set up quantization config with CPU offloading
    # Configuring quantization with CPU offload
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization
        bnb_4bit_quant_type='nf4',  # Quantization type, e.g., 'nf4'
        bnb_4bit_compute_dtype=torch.float32,  # Set computation dtype
        load_in_8bit_fp32_cpu_offload=True,  # Enable CPU offloading for modules
    )
    quantization_config = bnb_config
    
    model_kwargs = dict(
        revision=args.model_revision,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if args.gradient_checkpointing else True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    
    # Step 1: Load the base LLaMA2-7B model   
    max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number.
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name_or_path,
        max_seq_length = max_seq_length,
        dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
    )    
    
    # Step 2: Load the adapters
    # Get the absolute path to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, "llama", "llama-7b-etpc")

    peft_config = PeftConfig.from_pretrained(adapter_dir)

    model = PeftModel.from_pretrained(model, adapter_dir)
    
    # Load the adapter a second time, with a different name, which will be our reference model.
    model.load_adapter(adapter_dir, adapter_name="reference")    
   
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(model,**vars(peft_config))
    
    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {args.output_dir}")
    )

    ################
    # Dataset
    ################
    # Construct the path to the dataset
    dataset_path = os.path.join(script_dir, "data", "apty_dataset.parquet")
    
    file_extension = os.path.splitext(dataset_path)[-1].lower()

    if file_extension == ".parquet":
        ds = load_dataset("parquet", data_files=dataset_path)
    else:
        # Handle other formats as needed
        ds = load_dataset(dataset_path)
    
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    # Split the dataset into train and test
    train_test_split = ds['train'].train_test_split(test_size=0.2)  # 80% train, 20% validation
    ds = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test'],
    })

    train_dataset = ds['train']
    eval_dataset = ds['validation']

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            # model_adapter_name="train2",
            # ref_adapter_name="reference",
            args=TrainingArguments(
                per_device_train_batch_size = 4,
                gradient_accumulation_steps = 8,
                warmup_ratio = 0.1,
                num_train_epochs = 3,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                seed = 42,
                output_dir = "outputs",
            ),
            beta=0.1,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_length = 1024,
            max_prompt_length = 512,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(args.output_dir)