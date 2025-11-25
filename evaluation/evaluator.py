import time
from tqdm import tqdm
import numpy as np

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from evaluate import load as hf_load_metric

# For QA
import numpy as np

# -----------------------------
# UTILITY METRICS
# -----------------------------

def compute_classification_accuracy(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)['rougeL'].fmeasure
        scores.append(score)
    return np.mean(scores)


def compute_bleu(predictions, references):
    smoothie = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        ref_split = [ref.split()]   # expects list of reference lists
        pred_split = pred.split()
        score = sentence_bleu(ref_split, pred_split, smoothing_function=smoothie)
        scores.append(score)
    return np.mean(scores)


def compute_token_f1(preds, refs):
    """
    Token-level F1 for QA (SQuAD-style)
    """
    scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        common = set(pred_tokens) & set(ref_tokens)
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            scores.append(0)
            continue
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return np.mean(scores)



# ---------------------------------------------------------
#                EVALUATOR CLASS
# ---------------------------------------------------------

class EarlyExitEvaluator:
    """
    Evaluates any early-exit model + strategy combination on any dataset.

    Supports:
        - classification (accuracy)
        - summarization (ROUGE-L)
        - translation (BLEU)
        - QA span extraction (Token-F1)
        - custom percentage of dataset (already sampled by your loader)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # =====================================================
    # Main evaluation method
    # =====================================================
    def evaluate(self, model, strategy, dataset, task_type, max_samples=None):
        """
        model      → GPT2WithEarlyExit or T5WithEarlyExit
        strategy   → ConfidenceExitStrategy / ContinuousConfidenceExit
        dataset    → preprocessed dataset object
        task_type  → ["classification", "summarization", "translation", "qa"]
        max_samples→ optional subset (overrides loader’s fraction)

        Returns dict with:
            dataset_name, metric, score, avg_latency_sec, tokens_per_sec, avg_layers_used
        """

        predictions = []
        references = []
        latencies = []
        layers_used_list = []

        # If user wants absolute number of samples
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        for example in tqdm(dataset, desc="Evaluating"):
            strategy.reset()

            # ----------------------------------
            # Prepare input text
            # ----------------------------------
            if task_type == "classification":
                inp = example["text"]
                reference = example["label"]
            elif task_type == "summarization":
                inp = example["article"]
                reference = example["summary"]
            elif task_type == "translation":
                inp = example["src"]
                reference = example["tgt"]
            elif task_type == "qa":
                inp = f"question: {example['question']} context: {example['context']}"
                reference = example["answers"]["text"][0]
            else:
                raise ValueError("Invalid task type")

            # ----------------------------------
            # Run model with early exit
            # ----------------------------------
            t0 = time.time()
            pred_text, layers_used = model.generate_with_early_exit(inp)
            latency = time.time() - t0

            predictions.append(pred_text)
            references.append(reference)
            latencies.append(latency)
            layers_used_list.append(layers_used)

        # =====================================================
        # Compute metrics
        # =====================================================
        if task_type == "classification":
            metric_name = "accuracy"
            score = compute_classification_accuracy(predictions, references)

        elif task_type == "summarization":
            metric_name = "rougeL"
            score = compute_rouge_l(predictions, references)

        elif task_type == "translation":
            metric_name = "bleu"
            score = compute_bleu(predictions, references)

        elif task_type == "qa":
            metric_name = "token_f1"
            score = compute_token_f1(predictions, references)

        else:
            raise ValueError("Invalid task type")

        # =====================================================
        # Efficiency metrics
        # =====================================================
        avg_latency = np.mean(latencies)
        throughput = len(dataset) / sum(latencies)
        avg_layers = np.mean(layers_used_list)

        return {
            "metric": metric_name,
            "score": score,
            "avg_latency_sec": avg_latency,
            "tokens_per_sec": throughput,
            "avg_layers_used": avg_layers,
            "num_samples": len(dataset),
        }
