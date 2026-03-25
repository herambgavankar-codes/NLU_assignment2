from gensim.models import Word2Vec
import os

MODEL_PATH = "../outputs/models/"


def train_cbow(sentences, vector_size=100, window=5, negative=5):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=2,
        sg=0,
        negative=negative
    )
    return model

def save_models(cbow, skipgram):
    os.makedirs(MODEL_PATH, exist_ok=True)   # ✅ auto create

    cbow.save(MODEL_PATH + "cbow.model")
    skipgram.save(MODEL_PATH + "skipgram.model")

def train_all(sentences):
    cbow = train_cbow(sentences)
    skipgram = train_skipgram(sentences)

    save_models(cbow, skipgram)

    return cbow, skipgram
def train_skipgram(sentences, vector_size=100, window=5, negative=5):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=2,
        sg=1,
        negative=negative
    )
    return model


def save_models(cbow, skipgram):
    os.makedirs(MODEL_PATH, exist_ok=True)
    cbow.save(MODEL_PATH + "cbow.model")
    skipgram.save(MODEL_PATH + "skipgram.model")


# def train_all(sentences):
#     cbow = train_cbow(sentences)
#     skipgram = train_skipgram(sentences)

#     save_models(cbow, skipgram)

#     return cbow, skipgram