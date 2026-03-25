# NLU Assignment 2  
**Course:** Natural Language Understanding  
**Name:** Heramb Gavankar  

---

## Overview

This assignment consists of two main problems:

- **Problem 1:** Learning Word Embeddings using Word2Vec on IIT Jodhpur data  
- **Problem 2:** Character-Level Name Generation using RNN-based models  

---

## Problem 1: Word Embeddings

### Objective
To learn semantic word representations from IIT Jodhpur textual data using Word2Vec.

### Approach
- Collected data from:
  - IITJ website
  - Academic regulations
  - Course syllabus
- Applied preprocessing:
  - Lowercasing
  - Removing special characters
  - Sentence and word tokenization
  - Stopword removal
- Trained:
  - CBOW model
  - Skip-gram model

### Outputs
- Cleaned corpus  
- Trained models  
- Nearest neighbor analysis  
- Analogy experiments  
- t-SNE visualization  

### Observation
Skip-gram performs better than CBOW for capturing semantic relationships, though results are limited by dataset size.

---

## Problem 2: Character-Level Name Generation

### Objective
To generate Indian names using sequence models and compare their performance.

### Models Implemented
- Vanilla RNN  
- Bidirectional LSTM (BLSTM)  
- RNN with Attention  

### Dataset
- 1000+ Indian names  
- Character-level encoding  

### Evaluation Metrics
- **Novelty Rate:** Percentage of generated names not present in training data  
- **Diversity:** Ratio of unique generated names  

### Results

| Model          | Novelty | Diversity |
|---------------|--------|----------|
| RNN           | 98.00% | 1.0000   |
| BLSTM         | 100.00%| 0.8333   |
| Attention RNN | 100.00%| 0.9592   |

### Observation
- RNN generates the most realistic names  
- BLSTM shows repetition despite high novelty  
- Attention model produces diverse but sometimes unrealistic outputs  

---

## How to Run

**Problem 1**  

---

## 📌 Overview

This project focuses on learning word embeddings from IIT Jodhpur textual data using Word2Vec models. The goal is to capture semantic relationships between words and analyze how well the embeddings represent domain-specific concepts.

---

## 🎯 Objective

- Train Word2Vec models:
  - Continuous Bag of Words (CBOW)  
  - Skip-gram with Negative Sampling  

- Analyze embeddings using:
  - Cosine similarity  
  - Word analogies  

- Visualize embeddings using dimensionality reduction  

---

## 📂 Folder Structure


problem1_word2vec/
│
├── data/
│ ├── raw/ # PDFs / text files
│ └── processed/ # Cleaned corpus
│
├── outputs/
│ ├── models/ # Trained models
│ ├── plots/ # t-SNE, WordCloud
│ └── results/ # neighbors, analogies
│
├── preprocess.py
├── train.py
├── evaluate.py
├── visualize.py
└── run.py


---

## ⚙️ Setup Instructions

### 1. Create virtual environment

```bash
python -m venv .venv
2. Activate environment
.venv\Scripts\activate
3. Install dependencies
pip install gensim nltk matplotlib scikit-learn wordcloud PyPDF2
📊 Dataset
Text collected from IIT Jodhpur sources:
Official website
Academic regulations
Course syllabus
Stored in: data/raw/
Processed into: data/processed/corpus.txt
🔧 Preprocessing
Remove special characters and unwanted text
Convert to lowercase
Sentence tokenization
Word tokenization
Remove stopwords
Filter short tokens
🚀 How to Run
Step 1: Preprocess data
python preprocess.py
Step 2: Train models
python train.py
Step 3: Evaluate embeddings
python evaluate.py
Step 4: Visualize embeddings
python visualize.py
(Optional) Run full pipeline
python run.py

**Problem 2**  

This project focuses on generating Indian names at the **character level** using different recurrent neural network architectures. The goal is to design, implement, and compare multiple sequence models and evaluate their ability to produce realistic names.

---

## 🎯 Objective

- Implement and compare:
  - Vanilla RNN  
  - Bidirectional LSTM (BLSTM)  
  - RNN with Attention  

- Evaluate models using:
  - **Novelty Rate**  
  - **Diversity**  

- Perform qualitative analysis of generated names  

---
problem_2/
│
├── data/
│ ├── TrainingNames.txt
│ ├── processed_names.txt
│
├── models/
│ ├── rnn_model.pth
│ ├── blstm_model.pth
│ ├── attention_model.pth
│
├── results/
│ ├── rnn_generated_names.txt
│ ├── blstm_generated_names.txt
│ ├── attention_generated_names.txt
│ ├── evaluation_results.txt
│
├── src/
│ ├── dataset/
│ ├── models/
│ ├── training/
│ ├── generation/
│ ├── evaluation/
│ ├── utils/
│
└── report/


---

## ⚙️ Setup Instructions

### 1. Create virtual environment

```bash
python -m venv .venv
2. Activate environment
.venv\Scripts\activate
3. Install dependencies
pip install torch numpy
📊 Dataset
1000+ Indian names generated using an LLM
Stored in: TrainingNames.txt
Processed into: processed_names.txt
Preprocessing
Convert to lowercase
Remove duplicates
Filter invalid characters
Apply character-level tokenization
🚀 How to Run
Step 1: Prepare Dataset
python problem_2/src/dataset/prepare_names_dataset.py
Step 2: Build Vocabulary
python problem_2/src/dataset/vocabulary.py
Step 3: Test Dataset Loader
python problem_2/src/dataset/name_dataset.py
Step 4: Train Models
Vanilla RNN
python problem_2/src/training/train_rnn.py
BLSTM
python problem_2/src/training/train_blstm.py
Attention RNN
python problem_2/src/training/train_attention.py
Step 5: Generate Names
python problem_2/src/generation/generate_samples.py
Step 6: Evaluate Models
python problem_2/src/evaluation/evaluate_generation.py
📈 Evaluation Metrics
Novelty Rate

Percentage of generated names not present in training data

Diversity

Ratio of unique generated names to total generated names

📊 Results Summary
Model	Novelty Rate	Diversity
Vanilla RNN	92.00%	1.0000
BLSTM	100.00%	0.9785
Attention RNN	100.00%	0.6842
