"""
evaluate_generation.py
-----------------------------------------
Evaluate generated names for Problem 2.

Evaluates:
1. Vanilla RNN
2. BLSTM
3. Attention RNN

Metrics:
- Novelty Rate
- Diversity
-----------------------------------------
"""

import os
import sys

# Add project root to Python path for direct script execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# Ensure project root is in sys.path for imports
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.evaluation.metrics import compute_novelty_rate, compute_diversity

# This script evaluates the generated names from different models against the training dataset.
# It computes the novelty rate (percentage of generated names not in training set) and diversity (a measure of variety in the generated names).
# The results are saved to a text file and also printed to the console for easy reference.      
def load_names(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def evaluate_file(model_name, generated_file, training_names):
    generated_names = load_names(generated_file)

    novelty_rate = compute_novelty_rate(generated_names, training_names)
    diversity = compute_diversity(generated_names)

    return {
        "model": model_name,
        "count": len(generated_names),
        "novelty_rate": novelty_rate,
        "diversity": diversity,
    }

# Main evaluation function that loads the training names, evaluates each model's generated names, and saves/prints the results.
def main():
    train_file = "problem_2/data/processed_names.txt"
    rnn_file = "problem_2/results/rnn_generated_names.txt"
    blstm_file = "problem_2/results/blstm_generated_names.txt"
    attention_file = "problem_2/results/attention_generated_names.txt"
    output_file = "problem_2/results/evaluation_results.txt"

    if not os.path.exists(train_file):
        train_file = "data/processed_names.txt"
    # Load training names and evaluate each model's generated names against the training set, computing novelty and diversity metrics.
    training_names = load_names(train_file)
    results = []

    if os.path.exists(rnn_file):
        results.append(evaluate_file("Vanilla RNN", rnn_file, training_names))

    if os.path.exists(blstm_file):
        results.append(evaluate_file("BLSTM", blstm_file, training_names))

    if os.path.exists(attention_file):
        results.append(evaluate_file("Attention RNN", attention_file, training_names))

    os.makedirs("problem_2/results", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Results ===\n\n")

        for result in results:
            f.write(f"Model          : {result['model']}\n")
            f.write(f"Generated Count: {result['count']}\n")
            f.write(f"Novelty Rate   : {result['novelty_rate']:.2f}%\n")
            f.write(f"Diversity      : {result['diversity']:.4f}\n")
            f.write("-" * 40 + "\n")
    # Print the evaluation results to the console for quick reference.
    print("=== Evaluation Results ===")
    for result in results:
        print(f"Model          : {result['model']}")
        print(f"Generated Count: {result['count']}")
        print(f"Novelty Rate   : {result['novelty_rate']:.2f}%")
        print(f"Diversity      : {result['diversity']:.4f}")
        print("-" * 40)

    print(f"Saved to       : {output_file}")

# Run the main evaluation function when this script is executed directly.
if __name__ == "__main__":
    main()
    
    