from train import train_cbow, train_skipgram


def run_experiments(sentences):
    dimensions = [50, 100]
    windows = [3, 5]
    negatives = [5, 10]

    results = []

    for dim in dimensions:
        for win in windows:
            for neg in negatives:
                cbow = train_cbow(sentences, dim, win, neg)
                skip = train_skipgram(sentences, dim, win, neg)

                results.append({
                    "dim": dim,
                    "window": win,
                    "negative": neg,
                    "cbow_vocab": len(cbow.wv),
                    "skip_vocab": len(skip.wv)
                })

                print(f"Done: dim={dim}, win={win}, neg={neg}")

    return results