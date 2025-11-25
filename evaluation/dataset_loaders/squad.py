from datasets import load_dataset
import random

def load_squad(fraction=1.0, seed=42):
    ds = load_dataset("squad")
    data = ds["validation"]

    n = int(len(data) * fraction)
    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    def preprocess(batch):
        return {
            "context": batch["context"],
            "question": batch["question"],
            "answers": batch["answers"]
        }

    return sampled.map(preprocess)
