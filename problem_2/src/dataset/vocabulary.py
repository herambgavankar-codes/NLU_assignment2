"""
vocabulary.py
-----------------------------------------
Builds character-level vocabulary for the
name generation task.

This file:
1. Loads processed names
2. Extracts all unique characters
3. Creates character-to-index mapping
4. Creates index-to-character mapping
5. Adds special tokens:
   - <PAD>
   - <SOS>
   - <EOS>

The vocabulary is used to convert names
into integer sequences for model training.
-----------------------------------------
"""

from typing import List, Dict, Tuple


class Vocabulary:
    """
    Character-level vocabulary manager.
    """

    def __init__(self):
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"

        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]

        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

    def build_vocab(self, names: List[str]) -> None:
        """
        Build vocabulary from a list of processed names.

        Args:
            names (List[str]): List of cleaned names
        """
        unique_chars = set()

        for name in names:
            for ch in name:
                unique_chars.add(ch)

        # Sort characters for consistent ordering
        unique_chars = sorted(list(unique_chars))

        all_tokens = self.special_tokens + unique_chars

        self.char2idx = {ch: idx for idx, ch in enumerate(all_tokens)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

    def encode_name(self, name: str) -> List[int]:
        """
        Convert a name into a sequence of token indices.

        Format:
        <SOS> + characters + <EOS>

        Args:
            name (str): Input name

        Returns:
            List[int]: Encoded sequence
        """
        sequence = [self.char2idx[self.SOS_TOKEN]]

        for ch in name:
            sequence.append(self.char2idx[ch])

        sequence.append(self.char2idx[self.EOS_TOKEN])

        return sequence

    def decode_indices(self, indices: List[int]) -> str:
        """
        Convert index sequence back into string.

        Ignores special tokens.

        Args:
            indices (List[int]): List of token indices

        Returns:
            str: Decoded name
        """
        chars = []

        for idx in indices:
            ch = self.idx2char[idx]
            if ch not in self.special_tokens:
                chars.append(ch)

        return "".join(chars)

    def vocab_size(self) -> int:
        """
        Return total vocabulary size.
        """
        return len(self.char2idx)


def load_processed_names(file_path: str) -> List[str]:
    """
    Load processed names from file.

    Args:
        file_path (str): Path to processed names file

    Returns:
        List[str]: List of names
    """
    with open(file_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


if __name__ == "__main__":
    file_path = "problem_2/data/processed_names.txt"
    names = load_processed_names(file_path)

    vocab = Vocabulary()
    vocab.build_vocab(names)

    print("=== Vocabulary Statistics ===")
    print(f"Total names loaded   : {len(names)}")
    print(f"Vocabulary size      : {vocab.vocab_size()}")
    print(f"Character mapping    : {vocab.char2idx}")

    sample_name = names[0]
    encoded = vocab.encode_name(sample_name)
    decoded = vocab.decode_indices(encoded)

    print(f"\nSample name          : {sample_name}")
    print(f"Encoded sequence     : {encoded}")
    print(f"Decoded name         : {decoded}")