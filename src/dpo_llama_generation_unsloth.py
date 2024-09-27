"""
python src/dpo_generation.py \
 --model_name=meta-llama/Llama-2-7b-hf \
 --adapter_dir=llama/llama-7b-etpc
"""

import argparse
import logging
import os
from contextlib import nullcontext
import json
from tqdm import tqdm
from typing import List, Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

import torch
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments
from peft import PeftModel, PeftConfig
from trl import DPOTrainer, RichProgressCallback
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from distutils.util import strtobool

from llama.tokenizer import Tokenizer

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login
login(new_session=False)

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

def initialize_logging():
    """Initialize logging settings for rich logging.""" 
    if TRL_USE_RICH:
        from rich.console import Console
        from rich.logging import RichHandler
        logging.basicConfig(
            format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
        )
        return Console()
    return None

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduling")
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for DPOTrainer")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--adapter_dir", type=str, default="llama/llama-7b-etpc", help="Directory of the PEFT adapter")
    return parser.parse_args()

def load_and_prepare_model(model_name, adapter_dir):
    """Load the base model, tokenizer, and apply PEFT adapters."""
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Load PEFT adapters
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    
    # Load reference adapter for DPO
    model.load_adapter(adapter_dir, adapter_name="reference")
    
    # Apply additional LoRA weights
    model = FastLanguageModel.get_peft_model(model, **vars(peft_config))
    
    return model, tokenizer

def load_datasets(train_json_path, eval_json_path):
    """Load the datasets from JSONL files."""
    train_dataset = load_dataset('json', data_files={'train': train_json_path})['train']
    validation_dataset = load_dataset('json', data_files={'validation': eval_json_path})['validation']
    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
    })
    
def setup_trainer(args, model, train_dataset, eval_dataset, tokenizer):
    """Setup the DPOTrainer with the given arguments and datasets."""
    return DPOTrainer(
        model,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            seed=args.seed,
            output_dir=f"./out/gen-models/dpo_{args.model_name}",
        ),
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
    )
    
def load_data(filename):
    """Loads data from a file in JSON format.

    Args:
        filename (str): The path to the file to load.

    Returns:
        list: The loaded data as a list of dictionaries.

    Example:
        ```python
        filename = "data.json"
        data = load_data(filename)
        print(data)
        ```
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def text_completion(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    logprobs: bool = False,
    echo: bool = False,
) -> List[dict]:
    """
    Perform text completion for a list of prompts using the language generation model.

    Args:
        model: The model used for text generation.
        tokenizer: The tokenizer used to encode/decode text.
        prompts (List[str]): List of text prompts for completion.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
        If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Whether to include prompt tokens in the output. Defaults to False.

    Returns:
        List[dict]: List of dictionaries containing generated text and optionally log probabilities.
    """
    if max_gen_len is None:
        max_gen_len = model.config.max_position_embeddings - 1

    # Tokenize prompts and generate an attention mask
    if hasattr(tokenizer, 'encode'):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    else:
        # Assuming the custom tokenizer outputs a list of tokens
        inputs = {"input_ids": [tokenizer.tokenize(p) for p in prompts]}

    # Add attention mask if available (typically generated during tokenization)
    attention_mask = inputs.get('attention_mask', None)
    
    # Ensure input_ids are on the correct device
    input_ids = inputs['input_ids'].to(model.device)

    # Ensure attention_mask is also on the correct device, if available
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # Generate tokens using the model
    generation_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass the attention mask to prevent the warning
        max_length=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )

    return [{"generation": tokenizer.decode(g)} for g in generation_tokens]


def generate_paraphrases(
    data, model, tokenizer, max_gen_len, temperature, top_p, max_batch_size, num_examples
):
    paraphrases = []

    for i in tqdm(range(0, num_examples, max_batch_size)):
        batch = data[i : i + max_batch_size]
        user_messages = [instance["messages"][0]["content"] for instance in batch]

        # Use the modified text_completion function
        results = text_completion(
            model=model,
            tokenizer=tokenizer,
            prompts=user_messages,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        paraphrases.extend([result["generation"] for result in results])
        print("generate_paraphrases: ", len(paraphrases))

    return paraphrases


def evaluate(paraphrases, references):
    """
    Evaluates the quality of paraphrases compared to reference texts.

    Args:
        paraphrases (list): The generated paraphrases.
        references (list): The reference texts.

    Returns:
        dict: A dictionary containing the evaluation scores.

    Example:
        ```python
        paraphrases = ["Paraphrase 1", "Paraphrase 2"]
        references = ["Reference 1", "Reference 2"]

        scores = evaluate(paraphrases, references)
        print(scores)
        ```
    """

    rouge = Rouge()

    # ROUGE scores
    rouge_scores = rouge.get_scores(paraphrases, references, avg=True)

    # BLEU scores
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], paraphrase, smoothing_function=smoothie)
        for ref, paraphrase in zip(references, paraphrases)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-2": rouge_scores["rouge-2"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],
        "BLEU": avg_bleu,
    }
    
def main(
    num_examples: int = 1000,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 1024,
    max_batch_size: int = 4,
):
    # Initialize everything
    args = parse_arguments()
    output_dir = f"./out/gen-models/dpo_{args.model_name}"
    
    console = initialize_logging()
    PatchDPOTrainer()  # Patch the DPOTrainer with Unsloth optimizations
    torch.cuda.empty_cache()  # Clear GPU cache before starting

    # Load model and tokenizer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, args.adapter_dir)
    model, tokenizer = load_and_prepare_model(args.model_name, adapter_dir)
                                              
    # Load dataset from JSONL files
    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    ds = load_datasets(train_json_path, eval_json_path)
    
    # Prepare datasets for training
    train_dataset = ds['train']
    eval_dataset = ds['validation']

    # Context manager setup for rich progress bars
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = nullcontext() if not TRL_USE_RICH else console.status(f"[bold green]Training completed! Saving the model to {output_dir}")

    # Setup trainer
    with init_context:
        trainer = setup_trainer(args, model, train_dataset, eval_dataset, tokenizer)

    # print("Training started...")  # Start training
    # trainer.train()

    # # Save the trained model
    # with save_context:
    #     trainer.save_model(output_dir)
        
    # Load data and predict
    FastLanguageModel.for_inference(model)
    data_file= "out/generation_etpc_test.jsonl"
    test_data = load_data(data_file)
    
    print("Generating paraphrases...")
    generated_paraphrases = generate_paraphrases(
        test_data,
        model,
        tokenizer=tokenizer,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        max_batch_size=max_batch_size,
        num_examples=num_examples,
    )
    references = [item["messages"][1]["content"] for item in test_data[:num_examples]]
    print(f"Number of paraphrases: {len(generated_paraphrases)}")
    print(f"Number of references: {len(references)}")

    scores = evaluate(generated_paraphrases, references)
    print(f"Model: {args.model_name}")
    print(f"Adapter: {args.adapter_dir}")
    print(scores)

if __name__ == "__main__":
    main()
