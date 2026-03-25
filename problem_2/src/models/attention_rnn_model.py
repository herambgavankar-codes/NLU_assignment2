"""
attention_rnn_model.py
-----------------------------------------
RNN with a basic attention mechanism for
character-level name generation.

Architecture:
1. Character embedding
2. GRU encoder
3. Attention over all time-step outputs
4. Context-aware output prediction

This is a simple attention-based recurrent model
suitable for assignment comparison.
-----------------------------------------
"""

import torch
import torch.nn as nn

# Attention-based RNN model definition
class CharAttentionRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 1,
        pad_idx: int = 0
    ):
        super().__init__()
        # Store hyperparameters for reference
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
        # GRU layer to process the embedded input sequence
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Attention score layer
        self.attn = nn.Linear(hidden_dim, 1)

        # Output layer uses concatenation of RNN output and context
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    # Forward pass that computes the output logits, hidden states, and attention weights for a given input sequence.
    def forward(self, x, hidden=None):
        """
        x: [B, T]
        """
        embedded = self.embedding(x)              # [B, T, E]
        outputs, hidden = self.gru(embedded, hidden)   # [B, T, H]

        # Compute attention weights over time steps
        attn_scores = self.attn(outputs).squeeze(-1)   # [B, T]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T]

        # Weighted sum of outputs -> context vector
        context = torch.bmm(attn_weights.unsqueeze(1), outputs)  # [B, 1, H]
        context = context.repeat(1, outputs.size(1), 1)          # [B, T, H]
        # Combine RNN outputs with context for final prediction
        combined = torch.cat([outputs, context], dim=-1)         # [B, T, 2H]
        logits = self.fc(combined)                               # [B, T, V]
        # Return logits, hidden states, and attention weights for analysis
        return logits, hidden, attn_weights
    # Utility function to count the number of trainable parameters in the model for reference and comparison.
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Test the model with dummy input to verify dimensions and parameter count
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    vocab_size = 29
    # Initialize the attention-based RNN model with specified hyperparameters and test it with random input to verify the output shapes and parameter count.
    model = CharAttentionRNN(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=128,
        num_layers=1,
        pad_idx=0
    )
    # Create a dummy input tensor of shape [B, T] with random character indices for testing the model's forward pass.
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, hidden, attn_weights = model(x)
    # Print the shapes of the input, output logits, hidden states, and attention weights, as well as the total number of trainable parameters in the model for verification.
    print("=== Attention RNN Model Test ===")
    print(f"Input shape          : {x.shape}")
    print(f"Logits shape         : {logits.shape}")
    print(f"Hidden shape         : {hidden.shape}")
    print(f"Attention shape      : {attn_weights.shape}")
    print(f"Trainable parameters : {model.count_parameters()}")