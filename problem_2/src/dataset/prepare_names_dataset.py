"""
prepare_names_dataset.py
-----------------------------------------
This script prepares the dataset for Problem 2:
Character-Level Name Generation.

Steps performed:
1. Load raw names from TrainingNames.txt
2. Clean each name:
   - Convert to lowercase
   - Remove non-alphabetic characters
   - Remove extra spaces
3. Filter invalid/short names
4. Remove duplicates
5. Save cleaned names to processed_names.txt

This ensures a clean and consistent dataset
for training character-level models.
-----------------------------------------
"""

import os
import re

# Paths (relative to project root)
RAW_FILE = "problem_2/data/TrainingNames.txt"
PROCESSED_FILE = "problem_2/data/processed_names.txt"

#  Function to clean a single name string
def clean_name(name: str) -> str:
    """
    Clean a single name.

    Steps:
    - Strip leading/trailing whitespace
    - Convert to lowercase
    - Remove all characters except a-z and space
    - Replace multiple spaces with a single space

    Args:
        name (str): Raw name string

    Returns:
        str: Cleaned name
    """
    # Remove leading/trailing spaces
    name = name.strip()

    # Convert to lowercase
    name = name.lower()

    # Remove anything that is not a letter or space
    name = re.sub(r"[^a-z\s]", "", name)

    # Replace multiple spaces with single space
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def load_and_clean_names(file_path: str):
    """
    Load raw names and clean them.

    Args:
        file_path (str): Path to raw dataset

    Returns:
        list: Cleaned names
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    cleaned_names = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = clean_name(line)

            # Filter conditions:
            # - name should not be empty
            # - name length should be >= 2 (avoid noise)
            if cleaned and len(cleaned) >= 2:
                cleaned_names.append(cleaned)

    return cleaned_names


def remove_duplicates(names: list):
    """
    Remove duplicate names while preserving order.

    Args:
        names (list): List of names

    Returns:
        list: Unique names
    """
    return list(dict.fromkeys(names))


def save_names(names: list, file_path: str):
    """
    Save cleaned names to file.

    Args:
        names (list): List of names
        file_path (str): Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")


def main():
    print("=== Dataset Preparation Started ===")

    # Step 1: Load and clean raw names
    raw_names = load_and_clean_names(RAW_FILE)
    print(f"Total raw cleaned names     : {len(raw_names)}")

    # Step 2: Remove duplicates
    unique_names = remove_duplicates(raw_names)
    print(f"Total unique names          : {len(unique_names)}")

    # Step 3: Save processed dataset
    save_names(unique_names, PROCESSED_FILE)
    print(f"Saved processed dataset to  : {PROCESSED_FILE}")

    print("=== Dataset Preparation Completed ===")


if __name__ == "__main__":
    main()