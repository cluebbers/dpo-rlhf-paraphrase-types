import os
from typing import Dict

import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from sklearn.model_selection import train_test_split

TOP_10_PARAPHRASE_TYPES = [
    "addition/deletion",
    "change of order",
    "derivational changes",
    "inflectional changes",
    "punctuation changes",
    "same polarity substitution (contextual)",
    "semantic based",
    "spelling changes",
    "subordination and nesting changes",
    "synthetic/analytic substitution",
]


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
    login(token=hf_token, add_to_git_credential=True)


def clean_text(text):
    """
    Clean the text to fix encoding issues.

    Args:
        text (str): The original text that may contain encoding issues.

    Returns:
        str: The cleaned text.
    """
    replacements = {
        "Ãƒâ€šÃ‚Â½": "½",  # One-half
        "Ãƒâ€šÃ‚Â¼": "¼",  # One-quarter
        "Ãƒâ€šÃ‚Â©": "©",  # Copyright
        "Ãƒâ€šÃ‚Â¢": "¢",  # Cent
        "Ãƒâ€šÃ‚Â¡": "¡",  # Inverted exclamation
        "Ãƒâ€šÃ‚Â¿": "¿",  # Inverted question
        "ÃƒÂ¢Ã‚â‚¬": "–",  # En dash
        "ÃƒÂ¢Ã‚â„¢": "’",  # Right single quotation
        # Catch-all for any occurrence of Ãƒâ€¦
        "Ãƒâ€š": "",  # Remove leading misencoded sequences
        "Ãƒ": "",  # Remove leading misencoded sequences
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    return text


def preprocess_apty_ranked_dataset(dataset: Dataset) -> Dict[str, Dataset]:
    """
    Preprocess the APTY-ranked dataset by normalizing columns, extracting text, and cleaning data.

    Args:
        data (pd.DataFrame): The raw dataframe loaded from the dataset.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """

    data = pd.DataFrame(dataset)

    # Normalize the 'meta' column to create separate columns for 'id', 'annotators', and 'APT'
    meta_df = pd.json_normalize(data["meta"])
    data = data.drop(columns=["meta"]).reset_index(drop=True)
    data = pd.concat([data, meta_df], axis=1)

    # Extract the text from the nested dictionaries for 'chosen' and 'rejected'
    data["original"] = data["original"].apply(
        lambda x: x["text"] if isinstance(x, dict) else str(x)
    )
    data["chosen"] = data["chosen"].apply(
        lambda x: x["text"] if isinstance(x, dict) else str(x)
    )
    data["rejected"] = data["rejected"].apply(
        lambda x: x["text"] if isinstance(x, dict) else str(x)
    )

    # Clean the text to fix encoding issues
    data["original"] = data["original"].apply(clean_text)
    data["chosen"] = data["chosen"].apply(clean_text)
    data["rejected"] = data["rejected"].apply(clean_text)

    # Strip whitespace
    data["original"] = data["original"].str.strip()
    data["chosen"] = data["chosen"].str.strip()
    data["rejected"] = data["rejected"].str.strip()

    # Modify the last character according to the rules
    data["original"] = data["original"].apply(modify_last_character)
    data["chosen"] = data["chosen"].apply(modify_last_character)
    data["rejected"] = data["rejected"].apply(modify_last_character)

    # Drop duplicates
    data = data.drop_duplicates(subset=["original", "chosen", "rejected"])
    # Generate train/eval splits
    train_df, eval_df = train_test_split(
        data, test_size=0.2, stratify=data["APT"], random_state=42
    )

    # Create prompt datasets
    def create_dataset(df):
        prompts = [
            {
                "prompt": (
                    f"Instruction: Given the following sentence, generate a paraphrase with the following types. "
                    f"Sentence: {row['original']} \n "
                    f"Paraphrase Types: {row['APT']}\n\n"
                    f"Answer: "
                ),
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for _, row in df.iterrows()
        ]
        return Dataset.from_list(prompts)

    datasets = {"train": create_dataset(train_df), "eval": create_dataset(eval_df)}

    return datasets


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
        text += "."  # Add a '.' if the last character is a letter

    return text
