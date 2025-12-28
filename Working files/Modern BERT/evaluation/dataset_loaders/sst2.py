from datasets import load_dataset
import random

def load_sst2(fraction=1.0, number=None, task='test', seed=42):
    ds = load_dataset("glue", "sst2")

    if task == "train":
        data = ds["train"]
    elif task == "test":
        data = ds["validation"]     # GLUE SST2 has no labeled test set
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
            "text": batch["sentence"],
            "label": batch["label"]
        }

    return sampled.map(preprocess)
