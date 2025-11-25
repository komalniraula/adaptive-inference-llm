from datasets import load_dataset
import random

def load_sst2(fraction=1.0, seed=42):
    ds = load_dataset("glue", "sst2")
    data = ds["validation"]  # SST2 validation is standard for evaluation

    n = int(len(data) * fraction)
    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    def preprocess(batch):
        return {
            "text": batch["sentence"],
            "label": batch["label"]
        }

    return sampled.map(preprocess)
