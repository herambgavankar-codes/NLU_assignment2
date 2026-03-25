"""
helpers.py
-----------------------------------------
Helper script to test dataset + dataloader.
-----------------------------------------
"""

import os
import sys
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset.name_dataset import NameDataset
from src.utils.collate import collate_fn

# Test function to verify that the dataset and dataloader are working correctly by loading a batch of data and printing its shape and contents. This helps ensure that the data preprocessing and batching are functioning as expected before training the models.
def test_dataloader():
    file_path = "problem_2/data/processed_names.txt"
    if not os.path.exists(file_path):
        file_path = "data/processed_names.txt"

    dataset = NameDataset(file_path)
    pad_idx = dataset.vocab.char2idx[dataset.vocab.PAD_TOKEN]

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )
    # Load a single batch of data from the dataloader to inspect its structure and contents, including the input sequences, target sequences, and their lengths
    batch_inputs, batch_targets, lengths = next(iter(loader))

    print("=== DataLoader Test ===")
    print(f"Batch input shape    : {batch_inputs.shape}")
    print(f"Batch target shape   : {batch_targets.shape}")
    print(f"Lengths              : {lengths}")

    print("\nBatch input tensor:")
    print(batch_inputs)

    print("\nBatch target tensor:")
    print(batch_targets)


if __name__ == "__main__":
    test_dataloader()