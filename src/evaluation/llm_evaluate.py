import json
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
from pathlib import Path
import sys
import pandas as pd 

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.llm import QwenMCQ
from src.llm.utils import build_medmcqa_prompt, build_medqa_prompt

MEDMCQA_PATH = os.path.join(ROOT_DIR, "data", "processed", "medmcqa", "medmcqa.json")
MEDQA_PATH = os.path.join(ROOT_DIR, "data", "external", "medqa", "medqa_form2.csv")
OUTPUT_PATH_MEDMCQA = os.path.join(ROOT_DIR, "src", "evaluation", "llm_eval_medmcqa.json")
OUTPUT_PATH_MEDQA = os.path.join(ROOT_DIR, "src", "evaluation", "llm_eval_medqa.json")

device = "cuda"

PROMPT_TEMPLATE_MEDMCQA = """
You are answering a medical multiple choice question.

Choose the correct answer from A, B, C, or D.

Return ONLY ONE LETTER: A, B, C, or D.
Do not explain.

Question:
{question}

Answer:
"""

PROMPT_TEMPLATE_MEDQA = """
You are answering a medical multiple choice question.

Select the single best answer.

Return ONLY ONE LETTER (A, B, C, D, or E).
Do not explain your reasoning.
IF YOUR ANSWER IS UNKNOWN, CHANGE IT TO E

Question:
{question}

Options:
{options}

Answer:
"""


if __name__ == "__main__":

    # load model
    llm = QwenMCQ(device=device)

    # load dataset
    with open(MEDMCQA_PATH) as f:
        data = json.load(f)


    df= pd.read_csv(MEDQA_PATH)
    data1 = df.to_dict(orient="records")

    # random 1000 samples with seed
    random.seed(42)
    data_medmcqa = random.sample(data, min(1000, len(data)))
    data_medqa= random.sample(data1, min(1000, len(data1)))

    preds_medmcqa = []
    gts_medmcqa = []
    results_medmcqa = []


    preds_medqa = []
    gts_medqa = []
    results_medqa = []



    for item in tqdm(data_medqa, desc="Evaluating MedQA"):

        prompt = build_medqa_prompt(item, PROMPT_TEMPLATE_MEDQA)

        pred = llm.predict(prompt)
        if pred == "UNKNOWN":
            pred= "E"
        gt = item["answer"]

        preds_medqa.append(pred)
        gts_medqa.append(gt)

        results_medqa.append({
            "question": item["question"],
            "prediction": pred,
            "groundtruth": gt
        })

    for item in tqdm(data_medmcqa, desc= "Evaluating MedMCQA"):

        prompt = build_medmcqa_prompt(item, PROMPT_TEMPLATE_MEDMCQA)

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


    accuracy_medqa = accuracy_score(gts_medqa, preds_medqa)

    precision_medqa = precision_score(
        gts_medqa,
        preds_medqa,
        labels=["A", "B", "C", "D", "E"],
        average="macro",
        zero_division=0
    )

    recall_medqa = recall_score(
        gts_medqa,
        preds_medqa,
        labels=["A", "B", "C", "D", "E"],
        average="macro",
        zero_division=0
    )


    print("Accuracy (MedMCQA):", accuracy_medmcqa)
    print("Precision (MedMCQA):", precision_medmcqa)
    print("Recall (MedMCQA):", recall_medmcqa)

    print("Accuracy (MedQA):", accuracy_medqa)
    print("Precision (MedQA):", precision_medqa)
    print("Recall (MedQA):", recall_medqa)

    # save results
    with open(OUTPUT_PATH_MEDMCQA, "w") as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy_medmcqa,
                "precision": precision_medmcqa,
                "recall": recall_medmcqa
            },
            "predictions": results_medmcqa
        }, f, indent=2)

    # save results
    with open(OUTPUT_PATH_MEDQA, "w") as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy_medqa,
                "precision": precision_medqa,
                "recall": recall_medqa
            },
            "predictions": results_medqa
        }, f, indent=2)