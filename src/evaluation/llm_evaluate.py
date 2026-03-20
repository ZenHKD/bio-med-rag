import json
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.llm import QwenMCQ
from src.llm.utils import build_medmcqa_prompt

MEDMCQA_PATH = os.path.join(ROOT_DIR, "data", "processed", "medmcqa", "medmcqa.json")
OUTPUT_PATH = os.path.join(ROOT_DIR, "src", "evaluation", "llm_eval.json")

device = "cuda"

PROMPT_TEMPLATE = """
You are answering a medical multiple choice question.

Choose the correct answer from A, B, C, or D.

Return ONLY ONE LETTER: A, B, C, or D.
Do not explain.

Question:
{question}

Answer:
"""


if __name__ == "__main__":

    # load model
    llm = QwenMCQ(device=device)

    # load dataset
    with open(MEDMCQA_PATH) as f:
        data = json.load(f)

    # random 1000 samples with seed
    random.seed(42)
    data_medmcqa = random.sample(data, min(1000, len(data)))

    preds_medmcqa = []
    gts_medmcqa = []
    results_medmcqa = []

    for item in tqdm(data_medmcqa):

        prompt = build_medmcqa_prompt(item, PROMPT_TEMPLATE)

        pred = llm.predict(prompt)
        gt = item["answer"]

        preds_medmcqa.append(pred)
        gts_medmcqa.append(gt)

        results_medmcqa.append({
            "question": item["question"],
            "prediction": pred,
            "groundtruth": gt
        })

    # metrics
    accuracy_medmcqa = accuracy_score(gts_medmcqa, preds_medmcqa)

    precision_medmcqa = precision_score(
        gts_medmcqa,
        preds_medmcqa,
        labels=["A", "B", "C", "D"],
        average="macro",
        zero_division=0
    )

    recall_medmcqa = recall_score(
        gts_medmcqa,
        preds_medmcqa,
        labels=["A", "B", "C", "D"],
        average="macro",
        zero_division=0
    )

    print("Accuracy (MedMCQA):", accuracy_medmcqa)
    print("Precision (MedMCQA):", precision_medmcqa)
    print("Recall (MedMCQA):", recall_medmcqa)

    # save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy_medmcqa,
                "precision": precision_medmcqa,
                "recall": recall_medmcqa
            },
            "predictions": results_medmcqa
        }, f, indent=2)