import os
import torch
from transformers import TextStreamer
from peft import PeftModel, PeftConfig
from unsloth import FastLanguageModel
import chardet

import random

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_model_and_tokenizer(model_name_or_path, adapter_dir=None):
    """
    Load a model and tokenizer, applying PEFT adapters if specified.
    """
    # Load base model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        load_in_4bit=True, # Enables 4-bit quantization to run on consumer hardware
        dtype=torch.float16 
    )

    # Apply PEFT adapters if specified
    if adapter_dir:
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = FastLanguageModel.get_peft_model(model, **vars(peft_config))
    
    # Set up model for inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def generate_paraphrases(model, tokenizer, sentences, paraphrase_type, batch_size=8):
    """
    Generate paraphrases for a list of sentences using the provided model and tokenizer.
    """
    paraphrases = []
    device = next(model.parameters()).device

    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize input batch
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate paraphrases
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_return_sequences=1)  # Adjust max_length as needed

        # Decode generated outputs
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            paraphrases.append((paraphrase_type, generated_text))
    
    return paraphrases

def read_sentences_from_files(data_dir):
    """
    Read base sentences and their paraphrase types from text files in the specified directory.
    """
    sentences_by_type = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)

            # Detect file encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # Read file with the detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                paraphrase_type = lines[0].strip()  # First line specifies the paraphrase type
                sentences = [line.strip() for line in lines[1:]]
                sentences_by_type[paraphrase_type] = sentences
                
    return sentences_by_type

def save_paraphrases(paraphrases, output_dir, model_name, paraphrase_type):
    """
    Save generated paraphrases to a text file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(output_dir, f"{model_name}_{paraphrase_type}.txt")

    # Write paraphrases to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        for pt, text in paraphrases:
            file.write(f"{pt}: {text}\n")

    print(f"Paraphrases for type {paraphrase_type} saved.\n")
            
def main():
    # Load models
    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_model_name_or_path = "meta-llama/Llama-2-7b-hf"
    etpc_adapter_path = os.path.join(script_dir, "llama", "llama-7b-etpc")
    dpo_adapter_path = os.path.join(script_dir, "llama", "dpo_llama_apty_output")
    
    # Read sentences from files
    data_dir = os.path.join(script_dir, "basesentences")
    sentences_by_type = read_sentences_from_files(data_dir)
    
    # Directory to save paraphrases
    output_dir = os.path.join(script_dir, "generated_paraphrases")
           
    # Generate paraphrases for base model
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name_or_path)
    for paraphrase_type, sentences in sentences_by_type.items():
        print(f"Generating paraphrases for type: {paraphrase_type} using Llama-2-7b Model")
        base_paraphrases = generate_paraphrases(base_model, base_tokenizer, sentences, paraphrase_type)
        save_paraphrases(base_paraphrases, output_dir, "base_model", paraphrase_type)
        print(f"Paraphrases for type {paraphrase_type} saved.\n")
    del base_model, base_tokenizer  # Free up memory
    torch.cuda.empty_cache()
       
    # Generate paraphrases for etpc finetuned model
    etpc_model, etpc_tokenizer = load_model_and_tokenizer(base_model_name_or_path, etpc_adapter_path)
    for paraphrase_type, sentences in sentences_by_type.items():
        print(f"Generating paraphrases for type: {paraphrase_type} using ETPC Fine-Tuned Llama-2-7b")
        finetuned_paraphrases = generate_paraphrases(etpc_model, etpc_tokenizer, sentences, paraphrase_type)
        save_paraphrases(finetuned_paraphrases, output_dir, "etpc_model", paraphrase_type)
        print(f"Paraphrases for type {paraphrase_type} saved.\n")
    del etpc_model, etpc_tokenizer  # Free up memory
    torch.cuda.empty_cache()

    # Generate paraphrases for DPO fine-tuned model
    dpo_model, dpo_tokenizer = load_model_and_tokenizer(base_model_name_or_path, dpo_adapter_path)
    for paraphrase_type, sentences in sentences_by_type.items():
        print(f"Generating paraphrases for type: {paraphrase_type} using DPO + ETPC Fine-Tuned Llama-2-7b")
        dpo_paraphrases = generate_paraphrases(dpo_model, dpo_tokenizer, sentences, paraphrase_type)
        save_paraphrases(dpo_paraphrases, output_dir, "dpo_model", paraphrase_type)
        print(f"Paraphrases for type {paraphrase_type} saved.\n")
    del dpo_model, dpo_tokenizer  # Free up memory
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
