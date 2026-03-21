"""
MCQ Evaluator: loads MedQA / MedMCQA datasets and runs pipeline evaluation.
"""

import ast
import csv
import json
import random
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default dataset paths
DEFAULT_MEDQA_PATH   = str(PROJECT_ROOT / "data" / "external" / "medqa" / "medqa_form2.csv")
DEFAULT_MEDMCQA_PATH = str(PROJECT_ROOT / "data" / "processed" / "medmcqa" / "medmcqa.json")

LETTERS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class MCQItem:
    question: str          # formatted question + options string
    gold:     str          # correct letter A–E


@dataclass
class ItemResult:
    question:  str
    gold:      str
    predicted: str
    correct:   bool


@dataclass
class EvalResult:
    dataset:   str
    n_total:   int
    n_correct: int
    accuracy:  float
    precision: dict = field(default_factory=dict)   # {A: float, ...}
    recall:    dict = field(default_factory=dict)
    f1:        dict = field(default_factory=dict)
    per_item:  List[ItemResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_medqa(path: str = DEFAULT_MEDQA_PATH, n: Optional[int] = None, seed: int = 42) -> List[MCQItem]:
    """
    Load MedQA Form2 CSV.
    Columns: question, option_A, option_B, option_C, option_D, option_E, answer, metamap
    answer column: letter A–D
    """
    items: List[MCQItem] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q    = row["question"].strip()
            opts = {
                "A": row["option_A"].strip(),
                "B": row["option_B"].strip(),
                "C": row["option_C"].strip(),
                "D": row["option_D"].strip(),
            }
            # option_E is always None/empty in Form2 — skip
            gold = row["answer"].strip().upper()
            if gold not in LETTERS:
                continue

            question_str = (
                f"{q}\n"
                f"A. {opts['A']}\n"
                f"B. {opts['B']}\n"
                f"C. {opts['C']}\n"
                f"D. {opts['D']}"
            )
            items.append(MCQItem(question=question_str, gold=gold))

    if n is not None and n < len(items):
        random.seed(seed)
        items = random.sample(items, n)

    return items

def load_medmcqa(path: str = DEFAULT_MEDMCQA_PATH, n: Optional[int] = None, seed: int = 42) -> List[MCQItem]:
    """
    Load pre-processed MedMCQA JSON.
    Format: [{"question": "...\n\nA. ...\nB. ...", "answer": "A"}, ...]
    Options are embedded in the question string already.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[MCQItem] = []
    for entry in raw:
        gold = entry["answer"].strip().upper()
        if gold not in LETTERS:
            continue
        items.append(MCQItem(question=entry["question"].strip(), gold=gold))

    if n is not None and n < len(items):
        random.seed(seed)
        items = random.sample(items, n)

    return items

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_metrics(results: List[ItemResult], labels: List[str] = None) -> EvalResult:
    """Compute accuracy, per-class precision, recall, F1."""
    labels = labels or LETTERS
    n      = len(results)
    if n == 0:
        return EvalResult(dataset="", n_total=0, n_correct=0, accuracy=0.0)

    n_correct = sum(r.correct for r in results)
    accuracy  = n_correct / n

    # Per-class TP, FP, FN
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}

    for r in results:
        g, p = r.gold, r.predicted
        if g == p:
            tp[g] += 1
        else:
            fn[g] = fn.get(g, 0) + 1
            fp[p] = fp.get(p, 0) + 1

    precision = {}
    recall    = {}
    f1        = {}
    for l in labels:
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
        rec  = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
        f1_  = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precision[l] = round(prec, 4)
        recall[l]    = round(rec, 4)
        f1[l]        = round(f1_, 4)

    return EvalResult(
        dataset="",
        n_total=n,
        n_correct=n_correct,
        accuracy=round(accuracy, 4),
        precision=precision,
        recall=recall,
        f1=f1,
        per_item=results,
    )


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
def evaluate(
    pipeline,
    items: List[MCQItem],
    dataset_name: str,
    progress_callback=None,
) -> EvalResult:
    """
    Run pipeline on each item and collect results.

    Args:
        pipeline:          RAGPipeline instance (thinking=False for speed).
        items:             List of MCQItems.
        dataset_name:      Label for EvalResult.
        progress_callback: Optional callable(i, item, result) for tqdm update.

    Returns:
        EvalResult with all metrics.
    """
    item_results: List[ItemResult] = []

    for i, item in enumerate(items):
        gen    = pipeline.run(item.question)
        pred   = gen["answer"]
        correct = (pred.upper() == item.gold.upper())

        item_results.append(ItemResult(
            question=item.question,
            gold=item.gold,
            predicted=pred,
            correct=correct,
        ))

        # Free fragmented KV cache between questions to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(i, item, item_results[-1])

    result = _compute_metrics(item_results)
    result.dataset = dataset_name
    return result
