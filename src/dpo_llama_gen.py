import argparse
import os
import json
import logging
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from typing import List, Optional
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login
with open("token_file.txt", "r") as token_file:
    hf_token = token_file.read().strip()
login(token=hf_token)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the model")
    parser.add_argument("--adapter_dir", type=str, default="llama/llama-7b-etpc", help="Directory of the PEFT adapter")
    return parser.parse_args()

def load_data(filename):
    """Loads data from a file in JSON format."""
    logging.info(f"Loading data from {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    logging.info(f"Loaded {len(data)} records from {filename}")
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

    return [{"generation": tokenizer.decode(g, skip_special_tokens=True)} for g in generation_tokens]

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
    max_batch_size: int = 8,
):    
    logging.basicConfig(
    filename='slurm_files/my_app.log',  # Specify the log file
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format with timestamp
)
    logging.info("Parsing arguments...")
    args = parse_arguments()
    output_dir = f"./out/gen-models/dpo_{args.model_name}_{args.adapter_dir}"
    
    logging.info(f"Loading model and tokenizer: {args.model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        )    
   
    logging.info("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    logging.info(f"Loading PEFT adapter from {args.adapter_dir}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, args.adapter_dir)
    
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    peft_config.base_model_name_or_path=args.model_name
    model = PeftModel.from_pretrained(model, adapter_dir, config=peft_config)
    
    logging.info("Loading reference adapter for DPO...")
    model.load_adapter(adapter_dir, adapter_name="reference")
    
    logging.info("Model moved to GPU")
    model = model.to(device)
    model.train()    
    
    torch.cuda.empty_cache()  # Clear GPU cache
    
    # Load dataset
    logging.info("Loading datasets...")
    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    
    train_dataset = load_dataset('json', data_files={'train': train_json_path})['train']
    validation_dataset = load_dataset('json', data_files={'validation': eval_json_path})['validation']
    logging.info(f"Loaded {len(train_dataset)} training samples and {len(validation_dataset)} validation samples.")
    
    ds = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
    })
    
    logging.info("Preparing datasets for training")
    train_dataset = ds['train']
    eval_dataset = ds['validation']

    logging.info("Setting up the DPO trainer...")
    training_args = DPOConfig(
        output_dir=f"./out/gen-models/dpo_{args.model_name}_{args.adapter_dir}",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_length=1024,
        max_prompt_length=512,
        fp16=True,
        optim="adamw_8bit",
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    logging.info("Starting training...")    
    trainer.train()

    logging.info("Training completed. Saving the model...")
    trainer.save_model(f"./out/gen-models/dpo_{args.model_name}_{args.adapter_dir}")
    
    logging.info("Loading test data for paraphrase generation...")
    data_file = "out/generation_etpc_test.jsonl"
    test_data = load_data(data_file)

    logging.info("Generating paraphrases...")
    model.eval()
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
    logging.info(f"Generated {len(generated_paraphrases)} paraphrases, and found {len(references)} references.")

    logging.info("Evaluating paraphrases...")
    scores = evaluate(generated_paraphrases, references)

    logging.info(f"Model: {args.model_name}")
    logging.info(f"Adapter: {args.adapter_dir}")
    logging.info(f"Scores: {scores}")

if __name__ == "__main__":
    main()
