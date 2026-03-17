import torch 
from sentence_transformers import SentenceTransformer
import os 
import json
import numpy as np 
import faiss
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # scripts/ -> bio-med-rag/
sys.path.insert(0, str(project_root))

from src.embeddings.encoder import Encoder 

VECTORSTORE_DIR= os.path.join(project_root, "data", "vectorstore")

INDEX_FILE= "index.bin"
EMBED_FILE= "embeddings.npy"

JSONL_FILE= os.path.join(project_root, "data", "processed", "knowledge", "chunks.jsonl")

if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"

    with open(JSONL_FILE, "r") as f:
        docs= json.load(f)

    texts= [d["text"] for d in docs]
    batch_size= 16
    model_name= "NeuML/pubmedbert-base-embeddings"

    embed_model= Encoder(texts, batch_size,  model_name, device)

    embeddings= embed_model.encode()

    dim= embeddings.shape[1]
    index= faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, INDEX_FILE))
    np.save(os.path.join(VECTORSTORE_DIR, EMBED_FILE), embeddings)


    