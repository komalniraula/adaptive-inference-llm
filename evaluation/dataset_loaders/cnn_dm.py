from datasets import load_dataset
import random

def load_cnndm(fraction=1.0, seed=42):
    ds = load_dataset("cnn_dailymail", "3.0.0")
    data = ds["validation"]

    n = int(len(data) * fraction)
    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    def preprocess(batch):
        return {
            "article": batch["article"],
            "summary": batch["highlights"]
        }

    return sampled.map(preprocess)
