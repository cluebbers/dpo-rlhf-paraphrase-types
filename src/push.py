"""
python3 src/push.py \
    --model_name="meta-llama/Llama-3.1-8B" \
    --adapter_dir="cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid" \
    --output_dir="out/gen-models/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid"
    
python3 src/push.py \
    --model_name="meta-llama/Llama-3.1-8B" \
    --adapter_dir="cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo" \
    --output_dir="out/gen-models/Llama-3.1-8B-paraphrase-type-generation-apty-ipo"
    
python3 src/push.py \
    --model_name="meta-llama/Llama-3.1-8B" \
    --adapter_dir="cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc" \
    --output_dir="out/gen-models/Llama-3.1-8B-etpc"
"""

import argparse
import logging
import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from huggingface_hub import login

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge adapter with base model and push to hub")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Path to the base model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to the PEFT adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the merged model",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token, add_to_git_credential=True, new_session=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        padding=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Loading base model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    logging.info(f"Loading PEFT adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(
        model,
        args.adapter_dir,
        is_trainable=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id


    logging.info("Merging adapter with base model")
    model = model.merge_and_unload()
    
    model = model.to(torch.bfloat16)


    logging.info(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir,
                          safe_serialization=True,
                          torch_dtype=torch.bfloat16)
    
    tokenizer.save_pretrained(args.output_dir)

    logging.info(f"Pushing merged model to Hugging Face Hub at {args.adapter_dir}")
    model.push_to_hub(args.adapter_dir)
    tokenizer.push_to_hub(args.adapter_dir)

if __name__ == "__main__":
    main()
