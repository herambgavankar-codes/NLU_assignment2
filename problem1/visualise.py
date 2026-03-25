import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import numpy as np

PLOT_PATH = "../outputs/plots/"



def plot_wordcloud(tokens):
    os.makedirs(PLOT_PATH, exist_ok=True)

    text = " ".join(tokens)
    wc = WordCloud(width=800, height=400).generate(text)

    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(PLOT_PATH + "wordcloud.png")
    plt.show()


# 🔥 THIS FUNCTION MUST EXIST
def plot_tsne(model):
    os.makedirs(PLOT_PATH, exist_ok=True)

    words = list(model.wv.index_to_key)[:100]   # Define words

    vectors = [model.wv[w] for w in words]
    vectors = np.array(vectors)

    tsne = TSNE(n_components=2, perplexity=min(30, len(vectors)-1))
    reduced = tsne.fit_transform(vectors)
    x = reduced[:, 0]
    y = reduced[:, 1]

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)

    for i, word in enumerate(words):
        plt.annotate(word, (x[i], y[i]))

    plt.savefig(PLOT_PATH + "tsne.png")
    plt.show()