from datasets import load_dataset
import random

def load_agnews(fraction=1.0, number=None, task='test', seed=42):
    ds = load_dataset("ag_news")
    
    if task == "train":
        data = ds["train"]
    elif task == "test":
        data = ds["test"]   # standard AGNews evaluation split
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
            "text": batch["text"],
            "label": batch["label"]
        }

    return sampled.map(preprocess)
