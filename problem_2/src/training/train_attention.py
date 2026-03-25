"""
train_attention.py
-----------------------------------------
Training script for RNN with attention
for character-level Indian name generation.
-----------------------------------------
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset.name_dataset import NameDataset
from src.utils.collate import collate_fn
from src.models.attention_rnn_model import CharAttentionRNN

# Training function for the attention-based RNN model   
def train():
    data_path = "problem_2/data/processed_names.txt"
    save_path = "problem_2/models/attention_model.pth"

    if not os.path.exists(data_path):
        data_path = "data/processed_names.txt"
    
    os.makedirs("problem_2/models", exist_ok=True)
    # Hyperparameters
    embedding_dim = 32
    hidden_dim = 128
    num_layers = 1
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    # Check for GPU availability and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device         : {device}")
    # Load dataset and create dataloader
    dataset = NameDataset(data_path)
    pad_idx = dataset.vocab.char2idx[dataset.vocab.PAD_TOKEN]
    vocab_size = dataset.vocab.vocab_size()
    # Create DataLoader with custom collate function to handle variable-length sequences
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )   
    # Initialize the attention-based RNN model and move it to the appropriate device
    model = CharAttentionRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_idx=pad_idx
    ).to(device)
    # Print the number of trainable parameters in the model for reference
    print(f"Trainable parameters : {model.count_parameters()}")
    # Define the loss function (CrossEntropyLoss) with padding index ignored and the optimizer (Adam)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop for the specified number of epochs, iterating over batches of data
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        # Iterate over batches of input and target sequences along with their lengths from the dataloader   
        for batch_inputs, batch_targets, lengths in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # Zero the gradients before the backward pass
            optimizer.zero_grad()
            # Forward pass through the model to get the output logits for the current batch of inputs
            logits, _, _ = model(batch_inputs)
            # Reshape logits and targets to match the expected input shape for the loss function
            logits = logits.reshape(-1, vocab_size)
            batch_targets = batch_targets.reshape(-1)
            # Compute the loss between the predicted logits and the true targets, then perform backpropagation and update the model parameters
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            # Accumulate the loss for the current batch to compute the average loss for the epoch
            total_loss += loss.item()
        # Compute and print the average loss for the current epoch after processing all batches
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    # Save the trained model's state dictionary to the specified path for later use in inference or further training
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to       : {save_path}")


if __name__ == "__main__":
    train()