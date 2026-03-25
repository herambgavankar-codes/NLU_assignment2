from preprocess import load_and_process
from train import train_all
from evaluate import evaluate
from visualise import plot_tsne

print("Step 1: Processing data...")
sentences = load_and_process()

print("Step 2: Training models...")
cbow, skipgram = train_all(sentences)

print("Step 3: Evaluating...")
evaluate(skipgram)

print("Step 4: Visualizing...")
plot_tsne(skipgram)

print("DONE ✅")