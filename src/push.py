"""
python3 src/push.py \
    --model_name="meta-llama/Llama-3.1-8B" \
    --adapter_dir="cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid" \
    --output_dir="out/gen-models/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid"
    
python3 src/push.py \
    --model_name="meta-llama/Llama-3.1-8B" \
    --adapter_dir="cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo" \
    --output_dir="out/gen-models/Llama-3.1-8B-paraphrase-type-generation-apty-ipo"
"""

import argparse
import logging
from typing import Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from common import login_to_huggingface

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

def load_model_and_tokenizer(
    model_name: str, adapter_dir: str, device: torch.device
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
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

    logging.info(f"Loading base model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #quantization_config=bnb_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    logging.info(f"Loading PEFT adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    return model, tokenizer

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()
    login_to_huggingface()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.adapter_dir, device
    )

    logging.info("Merging adapter with base model")
    model = model.to(torch.float16)
    model = model.merge_and_unload()


    logging.info(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logging.info(f"Pushing merged model to Hugging Face Hub at {args.adapter_dir}")
    model.push_to_hub(args.adapter_dir)
    tokenizer.push_to_hub(args.adapter_dir)

if __name__ == "__main__":
    main()
