from datasets import load_dataset
import random

def load_cnndm(fraction=1.0, number=None, task='test', seed=42):
    ds = load_dataset("cnn_dailymail", "3.0.0")
    
    if task == "train":
        data = ds["train"] + ds["validation"] # combined train and val both as train set
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
            "article": batch["article"],
            "summary": batch["highlights"]
        }

    return sampled.map(preprocess)
