import os
import argparse
import torch

import pandas as pd

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

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
        text += '.'  # Add a '.' if the last character is a letter

    return text

def load_and_preprocess_apty_dataset(dataset):
    """
    Load and preprocess the APTY-ranked dataset into Hugging Face Dataset format for DPOTrainer.
    
    Args:
        dataset: The raw dataset object containing 'train' data.
        
    Returns:
        train_dataset (Dataset): Training dataset as Hugging Face Dataset object.
        test_dataset (Dataset): Validation dataset as Hugging Face Dataset object.
    """
    
    # Convert dataset into a pandas DataFrame
    data = pd.DataFrame(dataset["train"])
    
    # Normalize the 'meta' column to create separate columns for 'id', 'annotators', and 'APT'
    meta_df = pd.json_normalize(data['meta'])
    data = data.drop(columns=['meta']).reset_index(drop=True)
    data = pd.concat([data, meta_df], axis=1)

    # Extract the text from the nested dictionaries for 'chosen' and 'rejected'
    data['original'] = data['original'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x))
    data['chosen'] = data['chosen'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x))
    data['rejected'] = data['rejected'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x))

    # Strip whitespace
    data['original'] = data['original'].str.strip()
    data['chosen'] = data['chosen'].str.strip()
    data['rejected'] = data['rejected'].str.strip()

    # Rename columns to match DPOTrainer's expected format
    data = data.rename(columns={'original': 'prompt'})

    # Split the dataset into training and test sets
    train_df, test_df = train_test_split(data, test_size=0.3, stratify=data["APT"], random_state=42)

    # Convert the pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset

def parse_args():
    """
    Parses the command line arguments and returns the parsed arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    Example:
        ```python
        args = parse_args()
        print(args.model_name)
        ```"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large")
    parser.add_argument(
        "--task_name",
        type=str,
        default="paraphrase-type-generation",
        help="paraphrase-type-generation or paraphrase-generation",
    )
    parser.add_argument("--loss_type", type=str, default="sigmoid", help="sigmoid or ipo")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()  # Clear GPU cache before starting

    fine_tuned_model_dir = f"./out/gen-models/{args.model_name}_{args.task_name}"  # Path to the fine-tuned model
    
    # Find the latest checkpoint from the fine-tuned model directory
    checkpoint_dir = None
    if os.path.exists(fine_tuned_model_dir) and os.listdir(fine_tuned_model_dir):
        checkpoint_dirs = [f for f in os.listdir(fine_tuned_model_dir) if f.startswith('checkpoint')]
        if checkpoint_dirs:
            checkpoint_dir = os.path.join(fine_tuned_model_dir, max(checkpoint_dirs, key=lambda x: os.path.getctime(os.path.join(fine_tuned_model_dir, x))))
            print(f"Loading from fine-tuned checkpoint: {checkpoint_dir}")
        else:
            print("No checkpoint found in fine-tuned model directory.")
            return
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained(checkpoint_dir)
    model = model.to(device)  # Move model to GPU

    # Load training and evaluation datasets
    dataset = load_dataset("worta/apty", "APTY-ranked")
    train_dataset, eval_dataset = load_and_preprocess_apty_dataset(dataset)

    # Set up training arguments  
    training_args = DPOConfig(
        eval_strategy="epoch",
        per_device_train_batch_size=16,  
        gradient_accumulation_steps=2,        
        loss_type=args.loss_type,
        remove_unused_columns=False,  # Ensures unused columns aren't removed for DPOTrainer
        fp16=True, # Enable mixed precision training
        output_dir = f"./out/gen-models/{args.model_name}_{args.task_name}_{args.loss_type}",
        max_length=1024,
        max_prompt_length=512,
        max_target_length=512,
    )
            
    # Set up trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model 
    trainer.train() 

if __name__ == "__main__":
    main()
