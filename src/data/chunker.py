from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os 
import json 
import re
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# PubMedQA: word-based sliding window (size = words)
# MedQA:    RecursiveCharacterTextSplitter (size = characters)

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed", "knowledge")

JSON_FILE= os.path.join(PROCESSED_DATA_DIR, "knowledge.json")
OUTPUT_FILE= "chunks.jsonl"

def chunk_pubmedqa(text, word_size, word_overlap, min_words=10):
    """Word-based sliding window. Drops trailing chunks shorter than min_words."""
    words = text.split()
    chunks = []
    step = word_size - word_overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + word_size]
        if len(chunk_words) < min_words:
            break
        chunks.append(" ".join(chunk_words))

    return chunks

def chunk_medqa(text, char_size, char_overlap):
    """Character-based recursive splitting for long-form textbook text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def chunking(JSON_FILE):
    word_size = 512       # words  — used for pubmedqa
    word_overlap = 64
    char_size = 2048      # characters — used for medqa textbooks
    char_overlap = 256
    chunks = []

    with open(JSON_FILE, "r") as f:
        docs = json.load(f)

    for doc in tqdm(docs, desc="Chunking"):
        doc_id = doc["doc_id"]
        text = doc["text"]
        source = doc["source"]

        if source == "pubmedqa":
            chs = chunk_pubmedqa(text, word_size, word_overlap)
        else:
            chs = chunk_medqa(text, char_size, char_overlap)

        for i, c in enumerate(chs):
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{i}",
                "text": c,
                "source": source
            })

    return chunks

if __name__ == "__main__":
    chunks = chunking(JSON_FILE)

    out_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks -> {out_path}")
    print("Done!")
