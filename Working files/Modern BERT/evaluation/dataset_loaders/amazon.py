from datasets import load_dataset
import random

def load_amazon_polarity(fraction=1.0, number=None, task='test', seed=42):
    ds = load_dataset("amazon_polarity")

    # Only train/test available
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

    # preprocess
    def preprocess(batch):
        # Convert Amazon labels: 1 → negative (0), 2 → positive (1)
        label = 0 if batch["label"] == 1 else 1

        return {
            "text": batch["content"].strip(),
            "label": label
        }

    return sampled.map(preprocess)