from datasets import load_dataset
import random

def load_dbpedia(fraction=1.0, number=None, task='test', seed=42):
    # DBPedia has only train & test splits
    ds = load_dataset("dbpedia_14")

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

    # DBPedia fields: title + content → we use content (like normal classification)
    def preprocess(batch):
        return {
            "text": batch["content"],
            "label": batch["label"]
        }

    return sampled.map(preprocess)
