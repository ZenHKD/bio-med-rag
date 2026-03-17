import torch
import os
import json
import numpy as np
import faiss
import sys
import argparse
import requests
from pathlib import Path

FILE_DIR = Path(__file__).parent.parent  
sys.path.insert(0, str(FILE_DIR))

from src.embeddings.encoder import Encoder
from scripts.prompt import *

# Paths
VECTORSTORE_DIR = os.path.join(FILE_DIR, "data", "vectorstore")
INDEX_FILE      = "index.bin"
CHUNKS_FILE     = os.path.join(FILE_DIR, "data", "processed", "knowledge", "chunks.jsonl")

# vLLM server
VLLM_URL   = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"

EMBED_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
BATCH_SIZE       = 16
TOP_K            = 5

# Prompt
SYSTEM_PROMPT = prompt_mcq

device = "cuda" if torch.cuda.is_available() else "cpu"

class Bio_RAG:
    def __init__(self):
        pass

    def build_prompt(self, context: str, question: str) -> str:
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


    def embed_question(self, question: str, device: str) -> np.ndarray:
        encoder = Encoder([question], BATCH_SIZE, EMBED_MODEL_NAME, device)
        vec = encoder.encode()          # shape (1, dim)
        return vec.astype("float32")


    def retrieve(self, question: str, k: int, device: str) -> list[dict]:
        with open(CHUNKS_FILE, "r") as f:
            docs = json.load(f)

        index = faiss.read_index(os.path.join(VECTORSTORE_DIR, INDEX_FILE))
        q_vec = self.embed_question(question, device)

        distances, indices = index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(docs):
                continue
            chunk = docs[idx]
            results.append({"text": chunk["text"], "score": float(dist)})
        return results


    def call_llm(self, context: str, question: str, max_tokens: int) -> str:
        user_content = f"{SYSTEM_PROMPT}\n\n{self.build_prompt(context, question)}"
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        resp = requests.post(VLLM_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def chat(self, question, k = 5):

        chunks = self.retrieve(question, k, device)
        context = "\n\n".join(c["text"] for c in chunks)
        answer = self.call_llm(context, question, max_tokens=2)

        return answer

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Bio-Med RAG — query the pipeline")
#     parser.add_argument("--question", "-q", required=True, help="Medical question to answer")
#     parser.add_argument("--top-k",    "-k", type=int, default=TOP_K, help="Number of chunks to retrieve (default: 5)")
#     parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for LLM response (default: 512)")
#     parser.add_argument("--show-sources", action="store_true", help="Print retrieved source chunks")
#     args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     print(f"\nQuestion : {args.question}")
#     print(f"Retrieving top-{args.top_k} chunks...\n")

#     chunks = retrieve(args.question, k=args.top_k, device=device)
#     context = "\n\n".join(c["text"] for c in chunks)

#     print("Generating answer...\n")
#     answer = call_llm(context, args.question, max_tokens=args.max_tokens)

#     print("=" * 60)
#     print("ANSWER:")
#     print("=" * 60)
#     print(answer)

#     if args.show_sources:
#         print(f"\n{'=' * 60}")
#         print(f"SOURCES ({len(chunks)} chunks):")
#         print("=" * 60)
#         for i, chunk in enumerate(chunks, 1):
#             print(f"\n--- Chunk {i} (score: {chunk['score']:.4f}) ---")
#             print(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))
