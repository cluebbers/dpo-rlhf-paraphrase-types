import json
import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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


def preprocess_apty_ranked_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the APTY-ranked dataset by normalizing columns, extracting text, and cleaning data.

    Args:
        data (pd.DataFrame): The raw dataframe loaded from the dataset.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
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

    return data


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


def write_to_jsonl(data, filename):
    # Full list of paraphrase types
    all_paraphrase_types = [
        "Derivational Changes",
        "Inflectional Changes",
        "Modal Verb Changes",
        "Spelling changes",
        "Change of format",
        "Same Polarity Substitution (contextual)",
        "Same Polarity Substitution (habitual)",
        "Same Polarity Substitution (named ent.)",
        "Converse substitution",
        "Opposite polarity substitution (contextual)",
        "Opposite polarity substitution (habitual)",
        "Synthetic/analytic substitution",
        "Coordination changes",
        "Diathesis alternation",
        "Ellipsis",
        "Negation switching",
        "Subordination and nesting changes",
        "Direct/indirect style alternations",
        "Punctuation changes",
        "Syntax/discourse structure changes",
        "Entailment",
        "Identity",
        "Non-paraphrase",
        "Addition/Deletion",
        "Change of order",
        "Semantic-based",
    ]

    with open(filename, "w", encoding="utf-8") as file:
        for instance in tqdm(data):
            # Check if there are any paraphrase types in instance["APT"]
            # otherwise skip
            if not instance["APT"]:
                continue

            # Construct generation entry
            generation_entry = {
                "prompt": (
                    "Given the following sentence, generate a paraphrase with the following type. "
                    f"Sentence: {instance['original']} "
                    f"Paraphrase Type: {instance['APT']}. "
                    "Generated Paraphrase: "
                ),
                "chosen": f"{instance['chosen']}",
                "rejected": f"{instance['rejected']}",
            }

            # Write entries to the respective files
            file.write(json.dumps(generation_entry, ensure_ascii=False) + "\n")


def main():
    """Create JSONL files for the APTY-ranked dataset."""
    # Load dataset
    dataset = load_dataset("worta/apty", "APTY-ranked")

    # Preprocess dataset
    df = preprocess_apty_ranked_dataset(pd.DataFrame(dataset["train"]))

    # Split dataset into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["APT"], random_state=42
    )

    # Convert DataFrames to lists of dictionaries
    train_data = train_df.to_dict(orient="records")
    test_data = test_df.to_dict(orient="records")

    # Create output directory if it doesn't exist
    output_dir = "out"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write to JSONL files
    write_to_jsonl(
        train_data, os.path.join(output_dir, "generation_apty_ranked_train.jsonl")
    )
    write_to_jsonl(
        test_data, os.path.join(output_dir, "generation_apty_ranked_test.jsonl")
    )


if __name__ == "__main__":
    main()
