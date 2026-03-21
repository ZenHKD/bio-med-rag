import os
import ast
import pandas as pd 
import re 
import json
import sys
from tqdm import tqdm

from pathlib import Path
# Run this on the root of the project 
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

pubmedqa = os.path.join(ROOT_DIR, "data", "external", "pubmedqa", "pubmedqa.csv")
medqa_txt = os.path.join(ROOT_DIR, "data", "external", "medqa", "textbooks")
medmcqa = os.path.join(ROOT_DIR, "data", "external", "medmcqa", "original.json")

def preprocess_pubmedqa(pubmedqa):
    df = pd.read_csv(pubmedqa)
    df["context"] = df["context"].apply(ast.literal_eval)
    docs = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="PubMedQA"):
        docs.append({
            "doc_id": r["pubid"],
            "text": " ".join(r["context"]["contexts"]),  # Concat list -> 1 string
            "source": "pubmedqa"
        })
    return docs


def clean_line(line: str) -> str:
    """Clean noise on each line."""
    line = re.sub(r'^\s*[IVXLCDM]+\.\s+[A-Z\s]+$', '', line)


    # Filename artifacts
    line = re.sub(r'\w+_Ch\d+_p\d+-p\d+\.indd\s*\d*', '', line)

    # Timestamps: 29/01/19 10:58 AM
    line = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(AM|PM)', '', line, flags=re.IGNORECASE)

    # Part I, Part II headers
    line = re.sub(r'^\s*[ivxlcdm]+\s*$', '', line, flags=re.IGNORECASE)

    line = re.sub(r'^\s*[IVXLCDMivxlcdm]+\.\s+[A-Z\s]+$', '', line)

    line = re.sub(r'\bPart\s+[IVXLCDM]+\b', '', line, flags=re.IGNORECASE)

    # Boilerplate lines
    boilerplates = [
        r'this page intentionally left blank',
        r'all rights reserved',
        r'printed in the united states',
        r'no part of this publication',
    ]
    for pattern in boilerplates:
        line = re.sub(pattern, '', line, flags=re.IGNORECASE)

    # Inline figure references
    line = re.sub(r'[\(\[](fig\.?|figure)\s*\d+[\.\-]?\d*[\)\]]', '', line, flags=re.IGNORECASE)

    # "see figure X" phrases
    line = re.sub(r'(see|shown in|as in|refer to)\s+(fig\.?|figure)\s*\d+[\.\-]?\d*', '', line, flags=re.IGNORECASE)

    # Standalone page numbers
    line = re.sub(r'^\s*\d+\s*$', '', line)

    # Citation markers
    line = re.sub(r'\[\d+\]|\(\d+\)', '', line)

    # URLs
    line = re.sub(r'https?://\S+|www\.\S+', '', line)

    # Non-ASCII
    line = re.sub(r'[^\x00-\x7F]+', ' ', line)

    # Collapse spaces
    line = re.sub(r'[ \t]{2,}', ' ', line)

    # Orphan short lines
    if len(line.strip()) <= 6:
        return ''

    return line.strip()

def clean_medical_book_txt(text: str) -> str:
    """Clean on the entire text after joining lines."""

    # Figure blocks (need DOTALL — must clean on full text)
    text = re.sub(r'(fig\.?|figure)\s*\d+[\.\-]?\d*.*?(\n\n|\Z)',
                  '', text, flags=re.IGNORECASE | re.DOTALL)

    # References section (need DOTALL)
    text = re.sub(r'\n(references|bibliography|further reading)\n.*',
                  '', text, flags=re.DOTALL | re.IGNORECASE)

    # Fix hyphenated line breaks (need to handle \n)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()

def preprocess_medqa(medqa_txt):
    docs = []
    txt_files = [f for f in os.listdir(medqa_txt) if f.endswith(".txt")]
    for filename in tqdm(txt_files, desc="MedQA textbooks"):
        with open(os.path.join(medqa_txt, filename), "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()

        # Pass 1: Clean each line
        cleaned_lines = []
        for line in raw_lines:
            cleaned = clean_line(line)
            if cleaned:
                cleaned_lines.append(cleaned)

        # Pass 2: Clean entire block (patterns need DOTALL)
        joined = " ".join(cleaned_lines)
        cleaned_text = clean_medical_book_txt(joined)

        doc_id = os.path.splitext(filename)[0]
        docs.append({
            "doc_id": doc_id,
            "text": cleaned_text,
            "source": "medqa"
        })
    return docs

def preprocess_medmcqa(file_path):
    processed = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc= "MedMCQA"):
            q = json.loads(line)

            options = {
                "A": q["opa"],
                "B": q["opb"],
                "C": q["opc"],
                "D": q["opd"]
            }

            answer_letter = ["A","B","C","D"][q["cop"] - 1]

            question_text = q["question"] + "\n\n"

            for k, v in options.items():
                question_text += f"{k}. {v}\n"

            processed.append({
                "question": question_text.strip(),
                "answer": answer_letter
            })

    return processed


if __name__ == "__main__":
    pubmedqa_docs = preprocess_pubmedqa(pubmedqa)
    medqa_docs = preprocess_medqa(medqa_txt)
    medmcqa_docs= preprocess_medmcqa(medmcqa)

    knowledge_dir = os.path.join(PROCESSED_DATA_DIR, "knowledge")
    medmcqa_dir = os.path.join(PROCESSED_DATA_DIR, "medmcqa")

    os.makedirs(knowledge_dir, exist_ok=True)
    os.makedirs(medmcqa_dir, exist_ok=True)

    print("Saving knowledge.json...")
    with open(os.path.join(knowledge_dir, "knowledge.json"), "w", encoding="utf-8") as f:
        json.dump(pubmedqa_docs + medqa_docs, f, ensure_ascii=False, indent=2)

    print("Saving medmcqa.json...")
    with open(os.path.join(medmcqa_dir, "medmcqa.json"), "w", encoding="utf-8") as f:
        json.dump(medmcqa_docs, f, ensure_ascii=False, indent=2)

    print("Done!")