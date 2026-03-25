"""
train_rnn.py
-----------------------------------------
Training script for Vanilla RNN model
for character-level Indian name generation.
-----------------------------------------
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Set up paths to ensure the project root is in the system path for importing modules from src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset.name_dataset import NameDataset
from src.utils.collate import collate_fn
from src.models.rnn_model import CharRNN

# Training function for the RNN model
def train():
    # File paths
    data_path = "problem_2/data/processed_names.txt"
    save_path = "problem_2/models/rnn_model.pth"
    # Check if the data file exists at the specified path, if not, try an alternative path. Also, ensure that the directory for saving models exists, creating it if necessary.
    if not os.path.exists(data_path):
        data_path = "data/processed_names.txt"
    if not os.path.exists("problem_2/models"):
        os.makedirs("problem_2/models", exist_ok=True)
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)

    # Hyperparameters
    embedding_dim = 32
    hidden_dim = 128
    num_layers = 1
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device         : {device}")

    # Dataset and DataLoader
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

    # Model initialization and move to device   
    model = CharRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_idx=pad_idx
    ).to(device)

    print(f"Trainable parameters : {model.count_parameters()}")

    # Loss and optimizer definition, using CrossEntropyLoss with padding index ignored and Adam optimizer for training the RNN model
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        # Iterate over batches of input and target sequences along with their lengths from the dataloader
        for batch_inputs, batch_targets, lengths in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # Zero the gradients before the backward pass to prevent accumulation of gradients from previous iterations, then perform a forward pass through the model to get the output logits for the current batch of inputs
            optimizer.zero_grad()
            # Forward pass through the model to get the output logits for the current batch of inputs
            logits, _ = model(batch_inputs)

            # Reshape for CrossEntropyLoss
            # logits: [B, T, V] -> [B*T, V]
            # targets: [B, T] -> [B*T]
            logits = logits.reshape(-1, vocab_size)
            batch_targets = batch_targets.reshape(-1)
            # Compute the loss between the predicted logits and the true targets, then perform backpropagation and update the model parameters using the optimizer. Accumulate the loss for the current batch to compute the average loss for the epoch.
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            # Accumulate the loss for the current batch to compute the average loss for the epoch
            total_loss += loss.item()
        # Compute and print the average loss for the current epoch after processing all batches in the dataloader
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    # Save model state dictionary to the specified path for later use in inference or further training, checking if the directory exists before saving the model, and providing feedback on where the model is saved. If the specified directory does not exist
    if os.path.exists("problem_2/models"):
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to       : {save_path}")
    else:
        torch.save(model.state_dict(), "models/rnn_model.pth")
        print("Model saved to       : models/rnn_model.pth")

# Entry point for the script, calling the train function to start the training process when the script is executed
if __name__ == "__main__":
    train()