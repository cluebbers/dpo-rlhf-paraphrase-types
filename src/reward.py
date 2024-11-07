import logging

import torch
from datasets import Dataset, load_dataset
from peft import PeftConfig, PeftModel, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)
from trl import RewardConfig, RewardTrainer

from common import login_to_huggingface, preprocess_apty_ranked_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    login_to_huggingface("token_file.txt")

    torch.cuda.empty_cache()

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B"
    adapter_name = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
    )
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. Get original PEFT config but modify for classification
    peft_config = PeftConfig.from_pretrained(adapter_name)
    peft_config.task_type = TaskType.SEQ_CLS

    # def model_init(trial: Optional[optuna.trial.Trial] = None) -> PreTrainedModel:

    torch.cuda.empty_cache()
    logging.info("Initializing model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        ignore_mismatched_sizes=True,
        num_labels=1,
    )

    # 3. Create new PEFT model with classification config
    model = PeftModel.from_pretrained(model, adapter_name, config=peft_config)

    model.config.pad_token_id = tokenizer.pad_token_id

    # Load datasets
    # Load and preprocess APTY-ranked dataset
    logging.info("Loading and preprocessing APTY-ranked dataset")
    dataset = load_dataset("worta/apty", "APTY-ranked")
    datasets = preprocess_apty_ranked_dataset(dataset["train"])

    train_dataset = process_dataset_for_reward_model(datasets["train"], tokenizer)
    eval_dataset = process_dataset_for_reward_model(datasets["eval"], tokenizer)

    torch.cuda.empty_cache()

    # Define and train the reward model
    training_args = RewardConfig(
        output_dir=f"./out/cls-models/{adapter_name.split('/')[-1]}-apty-reward",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        max_length=512,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    torch.cuda.empty_cache()

    trainer.train()

    trainer.push_to_hub(f"{adapter_name.split('/')[-1]}-apty-reward")


if __name__ == "__main__":
    main()
