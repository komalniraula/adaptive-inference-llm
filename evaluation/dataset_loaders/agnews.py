from datasets import load_dataset
import random

def load_agnews(fraction=1.0, seed=42):
    ds = load_dataset("ag_news")
    data = ds["test"]   # standard AGNews evaluation split

    n = int(len(data) * fraction)
    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    def preprocess(batch):
        return {
            "text": batch["text"],
            "label": batch["label"]
        }

    return sampled.map(preprocess)
