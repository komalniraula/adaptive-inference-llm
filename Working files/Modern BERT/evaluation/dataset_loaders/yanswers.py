from datasets import load_dataset
import random

def load_yahoo(fraction=1.0, number=None, task='test', seed=42):
    ds = load_dataset("yahoo_answers_topics")

    # Yahoo provides only train/test
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

    # combine fields into a single text
    def preprocess(batch):
        combined_text = (batch.get("question_title", "") or "") + " " + \
                        (batch.get("question_content", "") or "")
        return {
            "text": combined_text.strip(),
            "label": batch["topic"]  # label name in HF dataset is "topic"
        }

    return sampled.map(preprocess)
