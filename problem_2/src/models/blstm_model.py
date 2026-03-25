"""
blstm_model.py
-----------------------------------------
Bidirectional LSTM model for character-level
name generation.

Architecture:
1. Character embedding layer
2. Bidirectional LSTM
3. Fully connected output layer

Because the LSTM is bidirectional, the hidden
representation size becomes 2 * hidden_dim.
-----------------------------------------
"""

import torch
import torch.nn as nn

# BLSTM model definition
class CharBLSTM(nn.Module):
    """
    Bidirectional LSTM model for character-level generation.
    """
    # The constructor initializes the embedding layer, bidirectional LSTM, and fully connected output layer based on the provided hyperparameters.
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 1,
        pad_idx: int = 0
    ):
        super().__init__()
        # Store hyperparameters for referenceand potential use in generation
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        # Embedding layer to convert character indices to dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        # Bidirectional LSTM layer to process the embedded input sequence, allowing the model to capture context from both directions.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Fully connected output layer that maps the concatenated hidden states from both directions of the LSTM to the vocabulary size for character prediction.
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    # Forward pass that computes the output logits and hidden states for a given input sequence. The output logits are computed from the concatenated hidden states of the bidirectional LSTM.  
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)            # [B, T, E]
        output, hidden = self.lstm(embedded, hidden)   # [B, T, 2H]
        logits = self.fc(output)               # [B, T, V]
        return logits, hidden
    # Utility function to count the number of trainable parameters in the model for reference and comparison.
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Test the model with dummy input to verify dimensions and parameter count
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    vocab_size = 29
    # Initialize the BLSTM model with specified hyperparameters and test it with random input to verify the output shapes and parameter count.
    model = CharBLSTM(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=128,
        num_layers=1,
        pad_idx=0
    )
    # Create a dummy input tensor of shape [B, T] with random character indices for testing the model's forward pass.
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, hidden = model(x)
    # Since the LSTM is bidirectional, the hidden state is a tuple of (h_n, c_n) where each has shape [num_layers * 2, B, hidden_dim] due to the bidirectionality. We will unpack the hidden state to verify the shapes of h_n and c_n separately.
    h_n, c_n = hidden

    print("=== BLSTM Model Test ===")
    print(f"Input shape          : {x.shape}")
    print(f"Logits shape         : {logits.shape}")
    print(f"Hidden h_n shape     : {h_n.shape}")
    print(f"Hidden c_n shape     : {c_n.shape}")
    print(f"Trainable parameters : {model.count_parameters()}")