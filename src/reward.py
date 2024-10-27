from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel,BitsAndBytesConfig
from trl import RewardTrainer, RewardConfig
import torch
import argparse
from datasets import Dataset, load_dataset
from typing import Tuple
from transformers  import get_scheduler



def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a paraphrase detection model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="out/cls-models/deberta-base_qqp_pd",
        help="Model name from Hugging Face hub."
    )
    return parser.parse_args()

def load_datasets(train_file: str, eval_file: str) -> Tuple[Dataset, Dataset]:
    """
    Load training and validation datasets from JSON files.

    Args:
        train_file (str): Path to the training dataset JSON file.
        eval_file (str): Path to the validation dataset JSON file.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
    """
    
    # Load the datasets from JSON files
    train_dataset = load_dataset('json', data_files={'train': train_file})['train']
    validation_dataset = load_dataset('json', data_files={'validation': eval_file})['validation']
    
    # Return the datasets as a tuple
    return train_dataset, validation_dataset

def process_dataset_for_reward_model(dataset, tokenizer):
    processed_data = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": []
    }

    for example in dataset:
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Concatenate prompt with chosen and tokenize
        chosen_input = prompt + " " + chosen
        chosen_tokens = tokenizer(chosen_input, 
                                  truncation=True, 
                                  padding="longest", 
                                  max_length=512, 
                                  return_tensors="pt")
        processed_data["input_ids_chosen"].append(chosen_tokens["input_ids"][0])
        processed_data["attention_mask_chosen"].append(chosen_tokens["attention_mask"][0])

        # Concatenate prompt with rejected and tokenize
        rejected_input = prompt + " " + rejected
        rejected_tokens = tokenizer(rejected_input, 
                                    truncation=True, 
                                    padding="longest", 
                                    max_length=512, return_tensors="pt")
        processed_data["input_ids_rejected"].append(rejected_tokens["input_ids"][0])
        processed_data["attention_mask_rejected"].append(rejected_tokens["attention_mask"][0])

    # Convert lists to tensor-friendly format
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
    args = parse_args()
        
    # Automatically set device to 'cuda' if available, else 'cpu'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load in 4-bit precision
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computations
    )

    # Load tokenizer and model
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, 
        num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1}
    ).to(device)
    
    peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
            r=8,  
            lora_alpha=32,  
            lora_dropout=0.05,
            bias="none"
        )
    
    model = get_peft_model(model, peft_config)
    
    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    train_dataset, eval_dataset = load_datasets(train_json_path, eval_json_path)
    train_dataset = process_dataset_for_reward_model(train_dataset, tokenizer)
    eval_dataset = process_dataset_for_reward_model(eval_dataset, tokenizer)

    torch.cuda.empty_cache()
    
    training_args = RewardConfig(
            output_dir=f"./out/cls-models/reward_{args.model_name.split('/')[-1]}",
            max_length=512,
            remove_unused_columns=False,   
            bf16=True,     
            fp16=False,
            num_train_epochs=20,
            eval_strategy="epoch",
            save_total_limit=1,
            lr_scheduler_type="reduce_lr_on_plateau",

    )
    
    trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,


)

    trainer.train()
    # Load and tokenize dataset

if __name__ == "__main__":
    main()