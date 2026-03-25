import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
PROCESSED_DIR = "../data/processed/"
PROCESSED_FILE = PROCESSED_DIR + "corpus.txt"
RAW_PATH = "../data/raw/"



def save_corpus(sentences):
    os.makedirs(PROCESSED_DIR, exist_ok=True)   # ✅ create folder automatically

    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")
def extract_text_from_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)

    for page in reader.pages:
        text += page.extract_text() + " "

    return text


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text


def preprocess_text(text):
    text = clean_text(text)
    sentences = sent_tokenize(text)

    processed = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
        if len(tokens) > 2:
            processed.append(tokens)

    return processed


def load_and_process():
    all_sentences = []

    for file in os.listdir(RAW_PATH):
        file_path = os.path.join(RAW_PATH, file)

        if file.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)

        elif file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue

        processed = preprocess_text(text)
        all_sentences.extend(processed)

    return all_sentences
if __name__ == "__main__":
    sentences = load_and_process()
    save_corpus(sentences)
    print("Corpus created successfully")