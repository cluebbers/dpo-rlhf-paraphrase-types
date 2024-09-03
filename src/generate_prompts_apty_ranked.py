import json
import os
import random

from datasets import load_dataset
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset
data_apty_ranked = load_dataset("worta/apty", "APTY-ranked")

# Convert to DataFrames
df_apty_ranked = pd.DataFrame(data_apty_ranked['train'])

# Normalize the 'meta' column to create separate columns for 'id', 'annotators', and 'APT'
meta_df = pd.json_normalize(df_apty_ranked['meta'])
# Drop the original 'meta' column and concatenate the new columns
df_apty_ranked = df_apty_ranked.drop(columns=['meta'])
df_apty_ranked = pd.concat([df_apty_ranked, meta_df], axis=1)

# Extract the text from the nested dictionaries for 'chosen' and 'rejected'
df_apty_ranked['original'] = df_apty_ranked['original'].apply(str)
df_apty_ranked['chosen'] = df_apty_ranked['chosen'].apply(lambda x: x['text'] if isinstance(x, dict) else str(x))
df_apty_ranked['rejected'] = df_apty_ranked['rejected'].apply(lambda x: x['text'] if isinstance(x, dict) else str(x))

# Check for Leading or Trailing Whitespace
# Strip whitespace and compare to detect issues
df_apty_ranked['original'] = df_apty_ranked['original'].str.strip()
df_apty_ranked['chosen'] = df_apty_ranked['chosen'].str.strip()
df_apty_ranked['rejected'] = df_apty_ranked['rejected'].str.strip()

# Function to modify the last character according to the rules
def modify_last_character(text):
    if text.endswith('"'):
        # Remove the last double quote
        text = text[:-1]
    elif text[-1].isalpha():
        # Add a '.' if the last character is a letter
        text += '.'
    return text

# Apply the function to each column
df_apty_ranked['original'] = df_apty_ranked['original'].apply(modify_last_character)
df_apty_ranked['chosen'] = df_apty_ranked['chosen'].apply(modify_last_character)
df_apty_ranked['rejected'] = df_apty_ranked['rejected'].apply(modify_last_character)

# Drop duplicates
df_unique_specific_columns = df_apty_ranked.drop_duplicates(subset=['original', 'chosen', "rejected"])

# Stratified train-test split
train_df, test_df = train_test_split(
    df_unique_specific_columns, test_size=0.3, stratify=df_unique_specific_columns['APT'], random_state=42
)

# Convert DataFrames to lists of dictionaries
train_data = train_df.to_dict(orient='records')
test_data = test_df.to_dict(orient='records')

def write_to_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        for instance in tqdm(data):
            # Check if there are any paraphrase types in instance["APT"]
            # otherwise skip
            if not instance["APT"]:
                continue

            # Construct detection entry
            detection_entry = {
                "prompt": (
                            "Given the following two sentences, which of the"
                            " paraphrase types are changed between them?"
                            f" Sentence 1: {instance['original']} Sentence 2:"
                            f" {instance['chosen']} Paraphrase Types:"
                            " Derivational Changes, Inflectional Changes,"
                            " Modal Verb Changes, Spelling changes, Change of"
                            " format, Same Polarity Substitution"
                            " (contextual), Same Polarity Substitution"
                            " (habitual), Same Polarity Substitution (named"
                            " ent.), Converse substitution, Opposite polarity"
                            " substitution (contextual), Opposite polarity"
                            " substitution (habitual), Synthetic/analytic"
                            " substitution, Coordination changes, Diathesis"
                            " alternation, Ellipsis, Negation switching,"
                            " Subordination and nesting changes,"
                            " Direct/indirect style alternations, Punctuation"
                            " changes, Syntax/discourse structure changes,"
                            " Entailment, Identity, Non-paraphrase,"
                            " Addition/Deletion, Change of order,"
                            " Semantic-based"
                        ),
                        "chosen": f" {instance['APT']}",
                        "rejected": f"{instance['rejected']}",
                    },

            # Construct generation entry
            generation_entry = {
                "prompt": (
                            "Given the following sentence, generate a"
                            " paraphrase with the following type. Sentence:"
                            f" {instance['original']} Paraphrase Type:"
                            f" {instance['APT']}."
                        ),
                    "chosen": f"{instance['chosen']}", 
                    "rejected": f"{instance['rejected']}",
            }

            # Write entries to the respective files
            if "detection" in filename:
                file.write(json.dumps(detection_entry) + "\n")
            else:
                file.write(json.dumps(generation_entry) + "\n")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists("out"):
        os.makedirs("out")

    # Write to JSONL files in the 'out' directory
    write_to_jsonl(train_data, "out/detection_train.jsonl")
    write_to_jsonl(test_data, "out/detection_test.jsonl")
    write_to_jsonl(train_data, "out/generation_train.jsonl")
    write_to_jsonl(test_data, "out/generation_test.jsonl")

    print("JSONL files created in 'out' directory successfully!")
