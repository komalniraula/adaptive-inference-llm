from datasets import load_dataset
import random

def load_rotten_tomatoes(fraction=1.0, number=None, task='test', seed=42):
    """
    Loads the Rotten Tomatoes movie review dataset for binary sentiment classification.

    Args:
        fraction (float): Fraction of the dataset to load (0.0 to 1.0).
        number (int, optional): Exact number of samples to load. Overrides fraction.
        task (str): Which split to load ('train', 'validation', or 'test').
        seed (int): Random seed for consistent sampling.
    """
    
    # 1. Load the dataset from Hugging Face
    # Rotten Tomatoes is loaded directly by its name
    ds = load_dataset("rotten_tomatoes")

    if task == "train":
        data = ds["train"]
    elif task == "validation": # Rotten Tomatoes has explicit validation and test splits
        data = ds["validation"]
    elif task == "test":
        data = ds["test"]
    else:
        raise ValueError("task must be 'train', 'validation', or 'test'")

    # 2. Determine the number of samples to select
    if number is None:
        n = int(len(data) * fraction)
    else:
        n = min(number, len(data))
    
    # 3. Sample the data randomly
    random.seed(seed)
    # The .select() method requires indices for sampling
    sampled = data.select(random.sample(range(len(data)), n))

    # 4. Define the preprocessing function
    def preprocess(batch):
        # Rotten Tomatoes uses the key 'text' for the review and 'label' for the sentiment
        return {
            "text": batch["text"],
            "label": batch["label"]
        }

    # 5. Apply preprocessing and return the final dataset
    return sampled.map(preprocess)

# Example Usage:
# train_data = load_rotten_tomatoes(task='train', number=1000)
# test_data = load_rotten_tomatoes(task='test', fraction=0.1)