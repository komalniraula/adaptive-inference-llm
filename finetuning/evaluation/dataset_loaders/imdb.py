from datasets import load_dataset
import random

def load_imdb(fraction=1.0, number=None, task='test', seed=42):
    ds = load_dataset("imdb")

    # train or test split
    if task == "train":
        data = ds["train"]
    elif task == "test":
        data = ds["test"]
    else:
        raise ValueError("task must be 'train' or 'test'")

    # sampling logic
    if number is None:
        n = int(len(data) * fraction)
    else:
        n = min(number, len(data))

    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    # preprocess (IMDB already has text + label)
    def preprocess(batch):
        return {
            "text": batch["text"].strip(),
            "label": batch["label"]
        }

    return sampled.map(preprocess)