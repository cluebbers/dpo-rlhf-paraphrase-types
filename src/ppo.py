from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers import pipeline, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from typing import Tuple, Dict, List, Any
from datasets import Dataset
import torch
import logging
from tqdm import tqdm
from huggingface_hub import login
import os
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, PeftModel, get_peft_model

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
    
def setup_model_and_tokenizer(
    model_name: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Setup the model and tokenizer for the paraphrase detection task.

    Args:
        model_name (str): The name of the model to use.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizerBase]: The model and tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Load in 4-bit precision
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computations
    )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B",
                                              padding_side="left",  # Set padding side to left
                                              use_fast=True  # Use fast tokenizer implementation
                                              )
    
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,  # Load in bfloat16 precision
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            ) 
    
    return model, tokenizer

def setup_ppo_trainer(model, tokenizer, train_dataset, batch_size, mini_batch_size, gradient_accumulation_steps) -> PPOTrainer:
    """
    Set up the PPO trainer with the given model, tokenizer, training dataset, and evaluation dataset.

    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        train_dataset (DatasetDict): The training dataset.
        eval_dataset (DatasetDict): The evaluation dataset.

    Returns:
        PPOTrainer: The set up PPO trainer.
    """
    # Set up the training arguments
    training_args = PPOConfig(
        seed=42,
        batch_size=batch_size,
        mini_batch_size =mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    return PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_args,
        dataset=train_dataset,
    )
    

def read_sentences_by_type(
    data_dir: str, 
    num_examples: int
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
                    paraphrase_type = next(file).strip()  # The first line is the paraphrase type name
                    sentences = [line.strip() for line in file if line.strip()]  # The rest are the sentences
            except UnicodeDecodeError:
                logging.warning(f"Failed to decode {file_name} with utf-8. Trying ISO-8859-1.")
                with open(file_path, "r", encoding="ISO-8859-1") as file:
                    paraphrase_type = next(file).strip()  # The first line is the paraphrase type name
                    sentences = [line.strip() for line in file if line.strip()]  # The rest are the sentences
            sentences = sentences[:num_examples]  # Truncate the list of sentences
            if paraphrase_type not in sentences_by_type:
                sentences_by_type[paraphrase_type] = {
                    "type": paraphrase_type,
                    "sentences": sentences[:num_examples]  # Add the list of sentences to the dictionary
                }
            else:
                sentences_by_type[paraphrase_type]["sentences"].extend(sentences[:num_examples])
                # Ensure no duplicates if needed
                sentences_by_type[paraphrase_type]["sentences"] = list(set(sentences_by_type[paraphrase_type]["sentences"]))

    return sentences_by_type

def tokenize_data(
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompts using the provided tokenizer.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        prompts (List[str]): The list of prompts to tokenize.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the tokenized input IDs and attention mask.
    """
    # Tokenize the prompts
    tokenized_inputs: Dict[str, torch.Tensor] = tokenizer(
        prompts,
        return_tensors="pt",  # Return PyTorch tensors
        padding=True,  # Pad the inputs to the longest sequence
        truncation=True,  # Truncate long sequences
    )

    return tokenized_inputs



def main():
    num_examples= 10    
    mini_batch_size=4
    gradient_accumulation_steps=4
    batch_size=mini_batch_size * gradient_accumulation_steps
    
    login_to_huggingface("/home/slim/dpo-rhlf-paraphrase-types/token_file.txt")
    
    output_dir= "out/gen-models"
    
    # Setup device
    if torch.cuda.is_available():        
        device = torch.device("cuda")
        logging.info(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")
    
    
    torch.cuda.empty_cache()
    
    model_name="/home/slim/dpo-rhlf-paraphrase-types/out/gen-models/llama-3.1-8b-etpc"        
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Read sentences by type from the APTY dataset
    apty_data: Dict[str, Dict[str, Any]] = read_sentences_by_type("out/basesentences", num_examples=num_examples)  
    
    queries = [
        f"Instruction: Given the following sentence, generate a paraphrase with the following type. "
        f"Sentence: {sentence} Paraphrase Type: {paraphrase_type}. Generated Paraphrase: "
        for paraphrase_type, details in apty_data.items()
        for sentence in details["sentences"]
    ]
    
    # Create the dataset with a single "query" column
    ppo_dataset_dict = {"query": queries}
    train_dataset = Dataset.from_dict(ppo_dataset_dict)
    
    def collate_fn(batch):
        # Extract the queries from the batch
        queries = [b["query"] for b in batch]

        # Tokenize the batch with padding and truncation handled automatically
        tokenized_batch = tokenizer(
            queries,
            padding=True,  # Pad to the longest sequence in the batch
            truncation=True,  # Truncate if necessary
            return_tensors="pt"  # Return as PyTorch tensors
        )

        # Convert input_ids and attention_mask to bfloat16
        tokenized_batch["input_ids"] = tokenized_batch["input_ids"].to(torch.long)
        tokenized_batch["attention_mask"] = tokenized_batch["attention_mask"].to(torch.bfloat16)

        return {
            "input_ids": tokenized_batch["input_ids"],
            "attention_mask": tokenized_batch["attention_mask"],
            "query": queries  # Keep the original queries
        }
  
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    reward_model_name = "/home/slim/dpo-rhlf-paraphrase-types/out/cls-models/deberta-base_qqp_pd" 
      
    reward_model_name = "microsoft/deberta-base" 
    reward_model = pipeline("text-classification", model=reward_model_name, device=device, torch_dtype=torch.bfloat16)
    
    torch.cuda.empty_cache()
    
    ppo_trainer = setup_ppo_trainer(model, tokenizer, train_dataset, 
                                    batch_size=batch_size, 
                                    mini_batch_size=mini_batch_size, 
                                    gradient_accumulation_steps=gradient_accumulation_steps)
        
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens":50,
    }
        
    epochs=3
    
    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(train_loader):
            if batch is None:
                logging.error("Received a None batch from the dataloader")
                continue
            query_tensors = batch["input_ids"]
            # Assuming query_tensors is a batch tensor of shape [batch_size, seq_len]
            query_tensors = [query for query in batch["input_ids"]]
        
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
            #### Compute reward score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model(texts)
            rewards = [torch.tensor(output["score"], dtype=torch.bfloat16) for output in pipe_outputs]
        
            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    ppo_trainer.save_pretrained(f"./out/gen-models/ppo_model")
    
    torch.cuda.empty_cache()  # Clear GPU cache before starting  

if __name__ == "__main__":
    main()