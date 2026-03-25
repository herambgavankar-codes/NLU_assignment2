import os

RESULT_PATH = "../outputs/results/"


def nearest_neighbors(model, words):
    results = {}

    for word in words:
        if word in model.wv:
            results[word] = model.wv.most_similar(word, topn=5)
        else:
            results[word] = "Word not in vocab"

    return results


def analogy_tests(model):
    analogies = {}

    try:
        analogies["UG:BTech :: PG:?"] = model.wv.most_similar(
            positive=["pg", "btech"], negative=["ug"], topn=3
        )
    except:
        analogies["UG:BTech :: PG:?"] = "Error"

    return analogies


def save_results(neighbors, analogies):
    os.makedirs(RESULT_PATH, exist_ok=True)

    with open(RESULT_PATH + "neighbors.txt", "w") as f:
        for word, vals in neighbors.items():
            f.write(f"{word}: {vals}\n")

    with open(RESULT_PATH + "analogies.txt", "w") as f:
        for k, v in analogies.items():
            f.write(f"{k}: {v}\n")


# 🔥 THIS IS THE IMPORTANT FUNCTION
def evaluate(model):
    words = ["research", "student", "phd", "exam"]

    neighbors = nearest_neighbors(model, words)
    analogies = analogy_tests(model)

    save_results(neighbors, analogies)

    return neighbors, analogies