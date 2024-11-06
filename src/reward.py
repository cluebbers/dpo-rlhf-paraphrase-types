import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import optuna
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import PeftConfig, PeftModel, TaskType
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import RewardConfig, RewardTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def process_dataset_for_reward_model(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Processes the dataset for a reward model by tokenizing chosen and rejected responses in batches.

    Args:
        dataset (Dataset): The dataset containing prompts, chosen, and rejected responses.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding the text.

    Returns:
        Dataset: A processed dataset with tokenized inputs for chosen and rejected responses.
    """
    # Prepare lists of inputs for batch tokenization
    chosen_inputs = [
        f"<|start_header_id|>user<|end_header_id|>{ex['prompt']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>{ex['chosen']}<|eot_id|>"
        for ex in dataset
    ]
    rejected_inputs = [
        f"<|start_header_id|>user<|end_header_id|>{ex['prompt']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>{ex['rejected']}<|eot_id|>"
        for ex in dataset
    ]

    chosen_tokens = tokenizer(
        chosen_inputs, truncation=True, padding=True, return_tensors="pt"
    )

    rejected_tokens = tokenizer(
        rejected_inputs, truncation=True, padding=True, return_tensors="pt"
    )

    processed_data = {
        "input_ids_chosen": chosen_tokens["input_ids"].tolist(),
        "attention_mask_chosen": chosen_tokens["attention_mask"].tolist(),
        "input_ids_rejected": rejected_tokens["input_ids"].tolist(),
        "attention_mask_rejected": rejected_tokens["attention_mask"].tolist(),
    }

    # Create and return the final dataset
    return Dataset.from_dict(processed_data)


def hyperparameter_space(trial: optuna.trial.Trial) -> Dict[str, Union[float, int]]:
    """
    Define the hyperparameter space to be searched by Optuna.

    This function returns a dictionary of hyperparameters to be searched by Optuna.
    The hyperparameters are:

    - learning_rate (float): The learning rate of the optimizer, sampled from a log-uniform distribution between 1e-5 and 1e-1.
    - weight_decay (float): The weight decay of the optimizer, sampled from a log-uniform distribution between 1e-6 and 1e-2.
    - per_device_train_batch_size (int): The batch size per device for training, sampled from a categorical distribution of [16, 32].

    The hyperparameters are sampled by Optuna and passed to the Trainer for training.

    Args:
        trial (Trial): The Optuna trial to sample hyperparameters from.

    Returns:
        Dict[str, Union[float, int]]: A dictionary of sampled hyperparameters.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [2, 4, 8]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [4, 8, 16]
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.2),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type",
            ["linear", "cosine", "polynomial"],
        ),
        "label_smoothing_factor": trial.suggest_float(
            "label_smoothing_factor", 0.0, 0.2
        ),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 0.1, log=True),
        "meaningful_gap_threshold": trial.suggest_float(
            "meaningful_gap_threshold", 0.05, 0.3
        ),  # Custom metric
    }


def model_init(trial: Optional[optuna.trial.Trial] = None) -> PreTrainedModel:
    global tokenizer
    global model_cache
    if model_cache is not None:
        logging.info("Using cached model.")
        return model_cache
    logging.info("Initializing model...")
    model_name = "meta-llama/Llama-3.1-8B"
    adapter_name = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        ignore_mismatched_sizes=True,
        num_labels=1,
    )

    # 2. Get original PEFT config but modify for classification
    peft_config = PeftConfig.from_pretrained(adapter_name)
    peft_config.task_type = TaskType.SEQ_CLS

    # 3. Create new PEFT model with classification config
    model = PeftModel.from_pretrained(model, adapter_name, config=peft_config)

    model.config.pad_token_id = tokenizer.pad_token_id

    model_cache = model

    return model


tokenizer = None
model_cache = None


def main() -> None:
    """
    Main function to train and evaluate the reward model.

    The model is trained on the IMDB dataset and the best model is saved in the scratch filesystem.
    The evaluation results are saved in a CSV file in the current directory.

    Parameters:
        None

    Returns:
        None
    """
    global tokenizer

    login_to_huggingface("token_file.txt")

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B"
    adapter_name = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
    )
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    torch.cuda.empty_cache()

    # Load datasets
    # Load and preprocess APTY-ranked dataset
    logging.info("Loading and preprocessing APTY-ranked dataset")
    dataset = load_dataset("worta/apty", "APTY-ranked")
    datasets = preprocess_apty_ranked_dataset(pd.DataFrame(dataset["train"]))

    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]

    train_dataset = process_dataset_for_reward_model(train_dataset, tokenizer)
    eval_dataset = process_dataset_for_reward_model(eval_dataset, tokenizer)

    torch.cuda.empty_cache()

    # Define and train the reward model
    training_args = RewardConfig(
        output_dir=f"./out/cls-models/{adapter_name.split('/')[-1]}-apty-reward",
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        load_best_model_at_end=True,
        bf16=True,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        warmup_ratio=0.1,
        max_length=512,
        greater_is_better=True,
    )

    trainer = RewardTrainer(
        model_init=lambda trial: model_init(trial),
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # # Perform the hyperparameter search using Optuna
    best_run = trainer.hyperparameter_search(
        direction="minimize",
        hp_space=hyperparameter_space,
        n_trials=200,  # Number of trials for hyperparameter search
        backend="optuna",
    )

    # # Save the best parameters after the search using a DataFrame
    best_hyperparameters: Dict[str, Union[float, int]] = best_run.hyperparameters
    best_params_df = pd.DataFrame([best_hyperparameters])

    output_params = (
        f"out/cls-models/{adapter_name.split('/')[-1]}_hyperparameters_reward.csv"
    )
    best_params_df.to_csv(output_params, index=False)
    print(f"Results written to {output_params}")

    torch.cuda.empty_cache()

    #trainer.train()

    # trainer.push_to_hub(
    #     f"{adapter_name.split('/')[-1]}-apty-reward"
    # )


if __name__ == "__main__":
    main()
