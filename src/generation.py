import os
import random
import torch
import chardet
from peft import PeftModel, PeftConfig
from unsloth import FastLanguageModel

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
            paraphrases.append((paraphrase_type, generated_text))
    
    return paraphrases

def read_sentences_from_files(data_dir):
    """
    Reads base sentences and their paraphrase types from text files in the specified directory.

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
                sentences = [line.strip() for line in lines[1:]]
                sentences_by_type[paraphrase_type] = sentences
                
    return sentences_by_type

def save_paraphrases(paraphrases, output_dir, model_name, paraphrase_type):
    """
    Saves the generated paraphrases to a text file.

    Args:
        paraphrases (list[tuple]): A list of tuples, where each tuple contains the paraphrase type and the generated paraphrase.
        output_dir (str): The directory where the output file should be saved.
        model_name (str): The name of the model used to generate the paraphrases.
        paraphrase_type (str): The type of paraphrase that was generated.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name}_{paraphrase_type}.txt")

    with open(file_path, 'w', encoding='utf-8') as file:
        for pt, text in paraphrases:
            file.write(f"{pt}: {text}\n")

    print(f"Paraphrases for type {paraphrase_type} saved to {file_path}.\n")


def process_model_generation(model_name_or_path, adapter_path, sentences_by_type, output_dir, model_suffix):
    """
    Generates paraphrases using the specified model and saves them.

    Args:
        model_name_or_path (str): The base model path or name.
        adapter_path (str): The adapter directory path.
        sentences_by_type (dict): Dictionary of sentences categorized by paraphrase type.
        output_dir (str): Directory to save the paraphrases.
        model_suffix (str): Suffix to append to the model name in saved files.
    """
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, adapter_path)
    for paraphrase_type, sentences in sentences_by_type.items():
        print(f"Generating paraphrases for type: {paraphrase_type} using {model_suffix} Model")
        paraphrases = generate_paraphrases(model, tokenizer, sentences, paraphrase_type)
        save_paraphrases(paraphrases, output_dir, model_suffix, paraphrase_type)
    del model, tokenizer
    torch.cuda.empty_cache()
           
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
    output_dir = "out/generated_paraphrases"

    sentences_by_type = read_sentences_from_files(data_dir)
    
    process_model_generation(base_model_name_or_path, None, sentences_by_type, output_dir, "base_model")
    process_model_generation(base_model_name_or_path, etpc_adapter_path, sentences_by_type, output_dir, "etpc_model")
    process_model_generation(base_model_name_or_path, dpo_adapter_path, sentences_by_type, output_dir, "dpo_model")
    
if __name__ == "__main__":
    main()
