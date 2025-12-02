from datasets import load_dataset
import random

def load_sst2(fraction=1.0, number=None, task='test', seed=42):
    ds = load_dataset("glue", "sst2")

    if task == 'train':
        data = ds['train']
    else:
        data = ds["validation"]  # SST2 validation is standard for evaluation

    if not number:
        n = int(len(data) * fraction)
    else:
        n = number
        
    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    def preprocess(batch):
        return {
            "text": batch["sentence"],
            "label": batch["label"]
        }

    return sampled.map(preprocess)
