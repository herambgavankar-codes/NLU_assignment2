"""
generate_samples.py
-----------------------------------------
Generate sample names from trained models.

Supports:
1. Vanilla RNN
2. BLSTM
3. Attention RNN
-----------------------------------------
"""

import os
import sys
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset.name_dataset import NameDataset
from src.models.rnn_model import CharRNN
from src.models.blstm_model import CharBLSTM
from src.models.attention_rnn_model import CharAttentionRNN

# Sample generation function that takes a trained model and generates a name character by character until an end token is produced or max length is reached.
def sample_from_model(model, vocab, device, max_length=12, temperature=0.8, model_type="rnn"):
    model.eval()

    sos_idx = vocab.char2idx[vocab.SOS_TOKEN]
    eos_idx = vocab.char2idx[vocab.EOS_TOKEN]

    input_tensor = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
    hidden = None
    generated_indices = []

    with torch.no_grad():
        for _ in range(max_length):
            if model_type == "attention":
                logits, hidden, _ = model(input_tensor, hidden)
            else:
                logits, hidden = model(input_tensor, hidden)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            if next_idx == eos_idx:
                break

            generated_indices.append(next_idx)
            input_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return vocab.decode_indices(generated_indices).strip()
# Function to generate names from a given model and save them to a specified output file. It generates a specified number of samples and prints some of them to the console for verification. The function handles different model types (RNN, BLSTM, Attention) and uses temperature sampling to control the randomness of the generated names. 
def generate_names(model, vocab, device, output_path, num_samples=100, model_type="rnn"):
    generated_names = []

    for _ in range(num_samples):
        name = sample_from_model(
            model, vocab, device,
            max_length=12,
            temperature=0.8,
            model_type=model_type
        )
        if len(name) >= 2:
            generated_names.append(name)

    with open(output_path, "w", encoding="utf-8") as f:
        for name in generated_names:
            f.write(name + "\n")

    print(f"Generated names      : {len(generated_names)}")
    print(f"Saved to             : {output_path}")

    print("\nSample generated names:")
    for name in generated_names[:20]:
        print(name)

# Main function to load the dataset, initialize models, load trained weights, and generate names for each model type. It checks for the existence of model files and generates names accordingly, saving them to the results directory.
def main():
    data_path = "problem_2/data/processed_names.txt"
    # Paths to trained model files and output files for generated names
    rnn_model_path = "problem_2/models/rnn_model.pth"
    blstm_model_path = "problem_2/models/blstm_model.pth"
    attention_model_path = "problem_2/models/attention_model.pth"
    # Output files for generated names from each model
    rnn_output = "problem_2/results/rnn_generated_names.txt"
    blstm_output = "problem_2/results/blstm_generated_names.txt"
    attention_output = "problem_2/results/attention_generated_names.txt"

    if not os.path.exists(data_path):
        data_path = "data/processed_names.txt"

    os.makedirs("problem_2/results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device         : {device}")

    dataset = NameDataset(data_path)
    vocab = dataset.vocab
    pad_idx = vocab.char2idx[vocab.PAD_TOKEN]
    vocab_size = vocab.vocab_size()
    # Check if the Vanilla RNN model file exists, load it, and generate names using the RNN model. The generated names are saved to the specified output file and a sample of generated names is printed to the console for verification.
    if os.path.exists(rnn_model_path):
        print("\n=== Generating from Vanilla RNN ===")
        rnn_model = CharRNN(
            vocab_size=vocab_size,
            embedding_dim=32,
            hidden_dim=128,
            num_layers=1,
            pad_idx=pad_idx
        ).to(device)
    
        rnn_model.load_state_dict(torch.load(rnn_model_path, map_location=device))
        print(f"Loaded model from    : {rnn_model_path}")
        generate_names(rnn_model, vocab, device, rnn_output, num_samples=100, model_type="rnn")
    # Check if the BLSTM model file exists, load it, and generate names using the BLSTM model. The generated names are saved to the specified output file and a sample of generated names is printed to the console for verification.
    if os.path.exists(blstm_model_path):
        print("\n=== Generating from BLSTM ===")
        blstm_model = CharBLSTM(
            vocab_size=vocab_size,
            embedding_dim=32,
            hidden_dim=128,
            num_layers=1,
            pad_idx=pad_idx
        ).to(device)

        blstm_model.load_state_dict(torch.load(blstm_model_path, map_location=device))
        print(f"Loaded model from    : {blstm_model_path}")
        generate_names(blstm_model, vocab, device, blstm_output, num_samples=100, model_type="blstm")
    # Check if the attention model file exists, load it, and generate names using the attention-based RNN model. The generated names are saved to the specified output file and a sample of generated names is printed to the console for verification.
    if os.path.exists(attention_model_path):
        print("\n=== Generating from Attention RNN ===")
        attention_model = CharAttentionRNN(
            vocab_size=vocab_size,
            embedding_dim=32,
            hidden_dim=128,
            num_layers=1,
            pad_idx=pad_idx
        ).to(device)

        attention_model.load_state_dict(torch.load(attention_model_path, map_location=device))
        print(f"Loaded model from    : {attention_model_path}")
        generate_names(attention_model, vocab, device, attention_output, num_samples=100, model_type="attention")

# Run the main generation function 
if __name__ == "__main__":
    main()