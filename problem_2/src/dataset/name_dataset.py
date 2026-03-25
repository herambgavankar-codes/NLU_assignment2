"""
name_dataset.py
-----------------------------------------
PyTorch dataset for character-level name generation.
-----------------------------------------
"""
# Standard library imports
from typing import List, Tuple
import os
import sys
import torch
from torch.utils.data import Dataset

# Add project root to Python path for direct script execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# Ensure project root is in sys.path for imports
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# Local imports
from src.dataset.vocabulary import Vocabulary, load_processed_names

# Dataset class for character-level name generation
class NameDataset(Dataset):
    def __init__(self, file_path: str):
        # Load processed names from the specified file
        self.names: List[str] = load_processed_names(file_path)
        # Build vocabulary from the names
        self.vocab = Vocabulary()
        # Build the vocabulary based on the loaded names
        self.vocab.build_vocab(self.names)
        # Encode names as sequences of indices
        self.encoded_names: List[List[int]] = [
            # Encode each name using the vocabulary
            self.vocab.encode_name(name) for name in self.names
        ]
    # Return the total number of samples in the dataset
    def __len__(self) -> int:
        return len(self.encoded_names)
    # Return the input and target tensors for a given index
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the encoded name sequence for the specified index
        sequence = self.encoded_names[idx]
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        # Convert input and target sequences to PyTorch tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        
        # Return the input and target tensors as a tuple
        return input_tensor, target_tensor

# Test the NameDataset class
if __name__ == "__main__":
    file_path = "problem_2/data/processed_names.txt"
    if not os.path.exists(file_path):
        file_path = "data/processed_names.txt"
    
    dataset = NameDataset(file_path)
    
    # Print dataset statistics and a sample input-target pair
    print("=== Dataset Statistics ===")
    print(f"Total samples        : {len(dataset)}")
    print(f"Vocabulary size      : {dataset.vocab.vocab_size()}")
    
    # Retrieve the first sample from the dataset
    sample_input, sample_target = dataset[0]
    # Print the input and target tensors for the first sample
    print("\nFirst sample tensors:")
    print(f"Input tensor         : {sample_input}")
    print(f"Target tensor        : {sample_target}")
    
    # Decode the input and target tensors back to strings for verification 
    decoded_input = dataset.vocab.decode_indices(sample_input.tolist())
    decoded_target = dataset.vocab.decode_indices(sample_target.tolist())
    
    # Print the decoded input and target strings to verify correctness
    print(f"\nDecoded input        : {decoded_input}")
    print(f"Decoded target       : {decoded_target}")