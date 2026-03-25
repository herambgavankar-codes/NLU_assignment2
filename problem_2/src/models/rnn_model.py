"""
rnn_model.py
-----------------------------------------
Vanilla character-level RNN model for name generation.

Architecture:
1. Character embedding layer
2. Vanilla RNN layer
3. Fully connected output layer

The model predicts the next character at each time step.
-----------------------------------------
"""

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """
    Vanilla character-level RNN model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 1,
        pad_idx: int = 0
    ):
        """
        Initialize model.

        Args:
            vocab_size: total vocabulary size
            embedding_dim: size of character embeddings
            hidden_dim: size of hidden state
            num_layers: number of recurrent layers
            pad_idx: padding token index
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # Character embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        # Vanilla RNN
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer to predict next character
        self.fc = nn.Linear(hidden_dim, vocab_size)
# Forward pass that computes the output logits and hidden states for a given input sequence. The output logits are computed from the hidden states of the RNN at each time step.
    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: input tensor of shape [batch_size, seq_len]
            hidden: optional hidden state

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden: final hidden state
        """
        embedded = self.embedding(x)                 # [B, T, E]
        output, hidden = self.rnn(embedded, hidden) # [B, T, H]
        logits = self.fc(output)                    # [B, T, V]

        return logits, hidden

    def count_parameters(self):
        """
        Return total trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Small standalone test
    batch_size = 4
    seq_len = 10
    vocab_size = 29
# Initialize the RNN model with specified hyperparameters and test it with random input to verify the output shapes and parameter count.
    model = CharRNN(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=128,
        num_layers=1,
        pad_idx=0
    )
# Create a dummy input tensor of shape [B, T] with random character indices for testing the model's forward pass.
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, hidden = model(x)

    print("=== RNN Model Test ===")
    print(f"Input shape          : {x.shape}")
    print(f"Logits shape         : {logits.shape}")
    print(f"Hidden shape         : {hidden.shape}")
    print(f"Trainable parameters : {model.count_parameters()}")