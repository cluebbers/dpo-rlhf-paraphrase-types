from peft import LoraConfig, TaskType, get_peft_model
import os
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer, 
                          PreTrainedTokenizerBase, 
                          PreTrainedModel,
                          BitsAndBytesConfig,
                          )
from trl import RewardTrainer, RewardConfig
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login

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
    login(token=hf_token)

def process_dataset_for_reward_model(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """
    Processes the dataset for a reward model by tokenizing chosen and rejected responses in batches.

    Args:
        dataset (Dataset): The dataset containing prompts, chosen, and rejected responses.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding the text.

    Returns:
        Dataset: A processed dataset with tokenized inputs for chosen and rejected responses.
    """
    # Prepare lists of inputs for batch tokenization
    chosen_inputs = [f"<|start_header_id|>user<|end_header_id|>{ex['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{ex['chosen']}<|eot_id|>" for ex in dataset]
    rejected_inputs = [f"<|start_header_id|>user<|end_header_id|>{ex['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{ex['rejected']}<|eot_id|>" for ex in dataset]

    # Tokenize all chosen responses in a single batch
    chosen_tokens = tokenizer(
        chosen_inputs,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    # Tokenize all rejected responses in a single batch
    rejected_tokens = tokenizer(
        rejected_inputs,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    # Convert tokenized data to lists for Dataset compatibility
    processed_data = {
        "input_ids_chosen": chosen_tokens["input_ids"].tolist(),
        "attention_mask_chosen": chosen_tokens["attention_mask"].tolist(),
        "input_ids_rejected": rejected_tokens["input_ids"].tolist(),
        "attention_mask_rejected": rejected_tokens["attention_mask"].tolist()
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
    
    # Load model and tokenizer
            
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model_name = "out/gen-models/llama-3.1-8b-etpc"
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        ignore_mismatched_sizes=True,
        num_labels=1,
        )
    
    if not hasattr(model.config, "peft_type") or model.config.peft_type is None:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,  
            lora_alpha=32,  
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
    

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", 
                                                                           padding_side="left",)      
    
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id or model.config.eos_token_id 
    
    torch.cuda.empty_cache()
    
    # Load datasets
    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    train_dataset = load_dataset('json', data_files={'train': train_json_path})['train']
    eval_dataset = load_dataset('json', data_files={'validation': eval_json_path})['validation']
    train_dataset = process_dataset_for_reward_model(train_dataset, tokenizer)
    eval_dataset = process_dataset_for_reward_model(eval_dataset, tokenizer)

    torch.cuda.empty_cache()
    
    # Define and train the reward model    
    training_args = RewardConfig(
            output_dir=f"./out/cls-models/reward_{model_name.split('/')[-1]}",
            max_length=512,
            remove_unused_columns=False, 
            gradient_accumulation_steps=4,  
            per_device_train_batch_size=8, 
            load_best_model_at_end=True,
            bf16=True,     
            fp16=False,
            num_train_epochs=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            lr_scheduler_type="reduce_lr_on_plateau",
            weight_decay=0.01,
            )
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        )
    
    torch.cuda.empty_cache()
    
    trainer.train()

if __name__ == "__main__":
    main()