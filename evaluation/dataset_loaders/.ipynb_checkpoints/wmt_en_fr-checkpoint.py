from datasets import load_dataset
import random

def load_wmt_enfr(fraction=1.0, number=None, task="test", seed=42):
    ds = load_dataset("wmt14", "en-fr")

    if task == "train":
        data = ds["train"]
    elif task == "test":
        data = ds["test"]
    else:
        raise ValueError("task must be 'train' or 'test'")

    if number is None:
        n = int(len(data) * fraction)
    else:
        n = min(number, len(data))
        
    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    def preprocess(batch):
        return {
            "src": batch["translation"]["en"],
            "tgt": batch["translation"]["fr"]
        }

    return sampled.map(preprocess)
