from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os 
import json 
import re
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# use word-based chunking for pubmedqa & RecursiveCharacterTextSplitter for medqa textbooks

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed", "knowledge")

JSON_FILE= os.path.join(PROCESSED_DATA_DIR, "knowledge.json")
OUTPUT_FILE= "chunks.jsonl"

def chunk_pubmedqa(text, size, overlap):
    words= text.split()
    chunks= []

    step= size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)

    return chunks    

def chunk_medqa(text, size, overlap):

    splitter= RecursiveCharacterTextSplitter(
        chunk_size= size,
        chunk_overlap= overlap,
        separators= ["\n\n","\n","."," "])
    
    splits= splitter.split_text(text)
    return splits

def chunking(JSON_FILE):
    size= 512
    overlap= 64
    chunks= []


    with open(JSON_FILE, "r") as f:
        docs = json.load(f)


    for doc in tqdm(docs, desc= "Chunking"):
        doc_id= doc["doc_id"]
        text= doc["text"]
        source= doc["source"]


        if source== "pubmedqa":
            chs= chunk_pubmedqa(text, size, overlap)
            for i, c in enumerate(chs):
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{i}",                    
                    "text": c,
                    "source": source
                })

        else:
            chs= chunk_medqa(text, size, overlap)
            for i, c in enumerate(chs):
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{i}",
                    "text": c,
                    "source": source
                })

    return chunks

if __name__ == "__main__":
    chunks= chunking(JSON_FILE)

    with open(os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILE), "w") as f:
        json.dump(chunks, f, indent= 2)
