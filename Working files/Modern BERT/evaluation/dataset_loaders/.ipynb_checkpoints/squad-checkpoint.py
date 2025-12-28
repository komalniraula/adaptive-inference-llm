from datasets import load_dataset
import random

def load_squad(fraction=1.0, number=None, task='test', seed=42):
    ds = load_dataset("squad")

    if task == "train":
        data = ds["train"]
    elif task == "test":
        data = ds["validation"]

    if number is None:
        n = int(len(data) * fraction)
    else:
        n = min(number, len(data))
        
    random.seed(seed)
    sampled = data.select(random.sample(range(len(data)), n))

    def preprocess(batch):
        return {
            "context": batch["context"],
            "question": batch["question"],
            "answers": batch["answers"]
        }

    return sampled.map(preprocess)
