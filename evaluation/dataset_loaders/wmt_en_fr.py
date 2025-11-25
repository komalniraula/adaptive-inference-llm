from datasets import load_dataset
import random

def load_wmt_enfr(fraction=1.0, seed=42):
    ds = load_dataset("wmt14", "fr-en")
    data = ds["validation"]

    n = int(len(data) * fraction)
    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    def preprocess(batch):
        return {
            "src": batch["translation"]["en"],
            "tgt": batch["translation"]["fr"]
        }

    return sampled.map(preprocess)
