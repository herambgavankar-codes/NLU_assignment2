"""
collate.py
-----------------------------------------
Custom collate function for batching variable-length
name sequences.
-----------------------------------------
"""

from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int):
    """
    Pads input and target sequences in a batch.

    Args:
        batch: list of (input_tensor, target_tensor)
        pad_idx: padding token index

    Returns:
        padded_inputs: [batch_size, max_seq_len]
        padded_targets: [batch_size, max_seq_len]
        lengths: original sequence lengths
    """
    inputs, targets = zip(*batch)

    lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)

    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=pad_idx
    )

    padded_targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_idx
    )
    # Return the padded input and target tensors along with their original lengths for use in training the models, allowing for proper handling of variable-length sequences during batching
    return padded_inputs, padded_targets, lengths