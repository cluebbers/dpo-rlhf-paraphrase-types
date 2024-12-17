import logging
import os
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import PPOConfig, PPOTrainer

from common import login_to_huggingface


def read_sentences_by_type(
    data_dir: str, num_examples: int
) -> Dict[str, Dict[str, List[str]]]:
    """
    Reads all .txt files in the given directory and creates a dictionary where the keys are the paraphrase type names
    (from the first line of each file) and the values are dictionaries containing the paraphrase type name and a list of
    sentences (from the rest of the file). The list of sentences is truncated to num_examples.

    Args:
        data_dir (str): The directory path containing the .txt files.
        num_examples (int): The maximum number of sentences to read per paraphrase type.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary where the keys are the paraphrase type names
            and the values are dictionaries containing the paraphrase type name and a list of sentences.
    """
    sentences_by_type: Dict[str, Dict[str, List[str]]] = {}

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8-sig") as file:
                    paraphrase_type = next(
                        file
                    ).strip()  # The first line is the paraphrase type name
                    sentences = [
                        line.strip() for line in file if line.strip()
                    ]  # The rest are the sentences
            except UnicodeDecodeError:
                logging.warning(
                    f"Failed to decode {file_name} with utf-8. Trying ISO-8859-1."
                )
                with open(file_path, "r", encoding="ISO-8859-1") as file:
                    paraphrase_type = next(
                        file
                    ).strip()  # The first line is the paraphrase type name
                    sentences = [
                        line.strip() for line in file if line.strip()
                    ]  # The rest are the sentences
            sentences = sentences[:num_examples]  # Truncate the list of sentences
            if paraphrase_type not in sentences_by_type:
                sentences_by_type[paraphrase_type] = {
                    "type": paraphrase_type,
                    "sentences": sentences[
                        :num_examples
                    ],  # Add the list of sentences to the dictionary
                }
            else:
                sentences_by_type[paraphrase_type]["sentences"].extend(
                    sentences[:num_examples]
                )
                # Ensure no duplicates if needed
                sentences_by_type[paraphrase_type]["sentences"] = list(
                    set(sentences_by_type[paraphrase_type]["sentences"])
                )

    return sentences_by_type


def process_dataset(apty_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:

    queries = [
        f"Instruction: Given the following sentence, generate a paraphrase with the following types. "
        f"Sentence: {sentence} \n "
        f"Paraphrase Types: {paraphrase_type}\n\n"
        f"Answer: "
        for paraphrase_type, details in apty_data.items()
        for sentence in details["sentences"]
    ]

    ppo_dataset_dict = {"query": queries}
    return Dataset.from_dict(ppo_dataset_dict)


def main():
    num_examples = 10
    mini_batch_size = 1
    gradient_accumulation_steps = 1
    batch_size = mini_batch_size * gradient_accumulation_steps

    policy_name = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc"
    reward_model_name = (
        "cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc-apty-reward"
    )

    login_to_huggingface()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        logging.info(
            f"Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}"
        )
    else:
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        use_fast=True,
        padding_side="left",
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    policy = AutoModelForCausalLM.from_pretrained(
        policy_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    policy.config.pad_token_id = tokenizer.pad_token_id

    if policy is None:
        raise ValueError("Failed to load policy model")

    ref_policy = AutoModelForCausalLM.from_pretrained(
        policy_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    ref_policy.config.pad_token_id = tokenizer.pad_token_id
    if ref_policy is None:
        raise ValueError("Failed to load ref_policy model")

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        num_labels=1,
    )

    reward_model.config.pad_token_id = tokenizer.pad_token_id
    if reward_model is None:
        raise ValueError("Failed to load reward model")

    torch.cuda.empty_cache()

    # Read sentences by type from the APTY dataset
    apty_data: Dict[str, Dict[str, Any]] = read_sentences_by_type(
        "out/basesentences", num_examples=num_examples
    )
    train_dataset = process_dataset(apty_data)

    torch.cuda.empty_cache()

    training_args = PPOConfig(
        seed=42,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir="./out/gen-models/Llama-3.1-8B-paraphrase-type-generation-etpc-apty-ppo",
        save_total_limit=1,
    )

    trainer = PPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
    )

    torch.cuda.empty_cache()

    trainer.train()

    # trainer.push_to_hub("Llama-3.1-8B-paraphrase-type-generation-etpc-apty-ppo")


if __name__ == "__main__":
    main()
