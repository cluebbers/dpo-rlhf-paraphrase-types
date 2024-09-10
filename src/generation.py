import os
import random
import torch
import chardet
from peft import PeftModel, PeftConfig
from unsloth import FastLanguageModel
import json

def set_seed(seed):
    """
    Set the seed for random number generators for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def load_model_and_tokenizer(model_name_or_path, adapter_dir=None):
    """
    Load a model and tokenizer, applying PEFT adapters if specified.

    Args:
        model_name_or_path (str): The name or path of the model to load.
        adapter_dir (str, optional): The path to the PEFT adapter to apply. Defaults to None.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        load_in_4bit=True,
        dtype=torch.float16 
    )

    if adapter_dir:
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = FastLanguageModel.get_peft_model(model, **vars(peft_config))

    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def generate_paraphrases(model, tokenizer, sentences, paraphrase_type, batch_size=8):
    """
    Generate paraphrases for a list of sentences using the provided model and tokenizer.

    Args:
        model (FastLanguageModel): The model to use for generating paraphrases.
        tokenizer (FastTokenizer): The tokenizer to use for encoding the input sentences.
        sentences (list[str]): The list of sentences to generate paraphrases for.
        paraphrase_type (str): The type of paraphrase to generate.
        batch_size (int, optional): The number of sentences to process in each batch. Defaults to 8.

    Returns:
        list[tuple]: A list of tuples, where each tuple contains the paraphrase type and the generated paraphrase.
    """
    paraphrases = []
    device = next(model.parameters()).device

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        prompts = [
            f"Instruction: Given the following sentence, generate a paraphrase with the following type. "
            f"Sentence: {sentence} Paraphrase Type: {paraphrase_type}. Generated Paraphrase: "
            for sentence in batch_sentences
        ]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                top_p=0.9,
                temperature=0.6
            )

        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            
            # Find the start of the generated paraphrase after "Generated Paraphrase: "
            paraphrase_start = "Generated Paraphrase: "
            paraphrase_index = generated_text.find(paraphrase_start)
            
            # If "Generated Paraphrase: " is found, remove everything before it
            if paraphrase_index != -1:
                generated_paraphrase = generated_text[paraphrase_index + len(paraphrase_start):].strip()
            else:
                generated_paraphrase = generated_text.strip()  # Default to the full generated text if not found

            paraphrases.append((generated_paraphrase))
    
    return paraphrases

def read_sentences_from_files(data_dir):
    """
    Reads base sentences and their paraphrase types from text files in the specified directory.
    Only the first 10 sentences are returned for each file.

    Args:
        data_dir (str): The directory containing the text files to read from.

    Returns:
        dict[str, list[str]]: A dictionary where the keys are the paraphrase types and the values are lists of sentences.
    """
    sentences_by_type = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)

            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                paraphrase_type = lines[0].strip()
                sentences = [line.strip() for line in lines[1:11]]  # Only take the first 10 sentences
                sentences_by_type[paraphrase_type] = sentences
                
    return sentences_by_type

def save_paraphrases_to_json(paraphrases, output_file):
    """
    Saves the generated paraphrases to a JSON file in the specified format.

    Args:
        paraphrases (list[dict]): A list of dictionaries containing paraphrase information.
        output_file (str): The file path to save the JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(paraphrases, file, ensure_ascii=False, indent=4)

    print(f"Paraphrases saved to {output_file}.\n")


def process_model_generation(model_name_or_path, adapter_path, sentences_by_type, model_suffix):
    """
    Generates paraphrases using the specified model and returns them in a list.

    Args:
        model_name_or_path (str): The base model path or name.
        adapter_path (str): The adapter directory path.
        sentences_by_type (dict): Dictionary of sentences categorized by paraphrase type.
        model_suffix (str): Suffix to append to the model name in saved files.

    Returns:
        list[dict]: A list of dictionaries containing generated paraphrases information.
    """
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, adapter_path)
    all_paraphrases = []

    for paraphrase_type, sentences in sentences_by_type.items():
        print(f"Generating paraphrases for type: {paraphrase_type} using {model_suffix} Model")
        paraphrases = generate_paraphrases(model, tokenizer, sentences, paraphrase_type)
        
        for index, paraphrase in enumerate(paraphrases, start=1):
            entry = {
                "data": {
                    "Original": sentences[index - 1],
                    "APT": paraphrase_type,
                    "Paraphrase": paraphrase,
                    "Kind": model_suffix,
                    "Index": index
                }
            }
            all_paraphrases.append(entry)

    del model, tokenizer
    torch.cuda.empty_cache()

    return all_paraphrases
           
def main():
    """
    The main entry point of the script. It runs the DPO paraphrase generation process on a set of input sentences.
    """
    set_seed(42)
    
    # Load models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_name_or_path = "meta-llama/Llama-2-7b-hf"
    etpc_adapter_path = os.path.join(script_dir, "llama", "llama-7b-etpc")
    dpo_adapter_path = "out/dpo_llama-7b_apty"
    data_dir = os.path.join(script_dir, "basesentences")
    output_file = "out/generated_paraphrases.json"

    sentences_by_type = read_sentences_from_files(data_dir)
    
    # Generate paraphrases for all models and aggregate in one JSON file
    all_paraphrases = []
    all_paraphrases.extend(process_model_generation(base_model_name_or_path, None, sentences_by_type, "base_model"))
    all_paraphrases.extend(process_model_generation(base_model_name_or_path, etpc_adapter_path, sentences_by_type, "etpc_model"))
    all_paraphrases.extend(process_model_generation(base_model_name_or_path, dpo_adapter_path, sentences_by_type, "dpo_model"))

    save_paraphrases_to_json(all_paraphrases, output_file)
    
if __name__ == "__main__":
    main()
