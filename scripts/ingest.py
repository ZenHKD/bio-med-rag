import torch
import os
import json
import numpy as np
import faiss
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # scripts/ -> bio-med-rag/
sys.path.insert(0, str(project_root))

from src.embeddings.encoder import Encoder

VECTORSTORE_DIR = os.path.join(project_root, "data", "vectorstore")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

INDEX_FILE    = "index.bin"
EMBED_FILE    = "embeddings.npy"
META_FILE     = "metadata.jsonl"

JSONL_FILE = os.path.join(project_root, "data", "processed", "knowledge", "chunks.jsonl")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # 32K context, instruction-aware, strong bio-med retrieval
BATCH_SIZE = 16

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load chunks (proper JSONL — one JSON object per line)
    print("Loading chunks...")
    docs = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    texts = [d["text"] for d in docs]
    print(f"Loaded {len(texts)} chunks")

    # Embed
    embed_model = Encoder(texts, BATCH_SIZE, MODEL_NAME, device)
    embeddings = embed_model.encode()  # model runs BF16 on GPU; numpy output is float32 (numpy has no BF16)

    # Build FAISS index (inner-product = cosine similarity since embeddings are normalised)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index + raw embeddings
    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, INDEX_FILE))
    np.save(os.path.join(VECTORSTORE_DIR, EMBED_FILE), embeddings)

    # Save metadata so FAISS results can be mapped back to source chunks
    with open(os.path.join(VECTORSTORE_DIR, META_FILE), "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({
                "chunk_id": doc["chunk_id"],
                "doc_id":   doc["doc_id"],
                "source":   doc["source"],
                "text":     doc["text"]
            }, ensure_ascii=False) + "\n")

    print(f"Saved index ({dim}-dim, {len(texts)} vectors) -> {VECTORSTORE_DIR}")
    print("Done!")