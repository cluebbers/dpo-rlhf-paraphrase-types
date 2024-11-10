"""
python3 src/dpo_llama_ptg.py \
    --model_name=meta-llama/Llama-2-7b-hf \
    --adapter_dir=out/gen-models/llama-2-7b-etpc \
    --loss_type=sigmoid
"""

import argparse
import logging
import os
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
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
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        padding=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Loading PEFT adapter from {adapter_dir}")
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
            low_cpu_mem_usage=True,
        )

        model.load_adapter(adapter_dir, adapter_name="reference")

    model.config.pad_token_id = tokenizer.pad_token_id

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
        loss_type (str): The type of loss to use. Can be either 'sigmoid' or 'ipo'.

    Returns:
        DPOTrainer: The set up DPO trainer.
    """

    training_args = DPOConfig(
        eval_strategy="epoch",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        output_dir=output_dir,
        max_prompt_length=512,
        optim="adamw_8bit", 
        warmup_ratio=0.1,  
        learning_rate=5e-5,
        max_length=1024,
        bf16=True,
        remove_unused_columns=False,
        loss_type=loss_type,  # The type of loss to use
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
    )

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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()

    sanitized_model_name = f"{args.model_name.split('/')[-1]}-paraphrase-type-generation-apty-{args.loss_type}"
    output_dir = f"./out/gen-models/{sanitized_model_name}"
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

    logging.info("Loading and preprocessing APTY-ranked dataset")
    dataset = load_dataset("worta/apty", "APTY-ranked")
    datasets = preprocess_apty_ranked_dataset(dataset["train"])

    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]

    trainer = setup_dpo_trainer(
        model, tokenizer, train_dataset, eval_dataset, output_dir, args.loss_type
    )

    torch.cuda.empty_cache()
    trainer.train()

    trainer.save_model(output_dir) 

    torch.cuda.empty_cache()
    # reference adapter is only needed for training
    model.delete_adapter("reference")

    #trainer.push_to_hub(sanitized_model_name)

    # model = model.to(torch.float16)
    # model = model.merge_and_unload()
    # model.save_pretrained(output_dir)
    # trainer.push_to_hub(sanitized_model_name)


if __name__ == "__main__":
    main()
