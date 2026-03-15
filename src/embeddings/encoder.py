
import torch 
from sentence_transformers import SentenceTransformer
import os 
import json
import numpy as np 
import faiss

FILE_DIR = os.getcwd()

if not (FILE_DIR.endswith("bio-med-rag")):
    raise ValueError("Please run this script from the bio-med-rag directory")
VECTORSTORE_DIR= os.path.join(FILE_DIR, "data", "vectorstore")

INDEX_FILE= "index.bin"
EMBED_FILE= "embeddings.npy"

JSONL_FILE= os.path.join(FILE_DIR, "data", "processed", "knowledge", "chunks.jsonl")

if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    embed_model= SentenceTransformer(
        "NeuML/pubmedbert-base-embeddings",
        device= device)
    
    with open(JSONL_FILE, "r") as f:
        docs= json.load(f)

    texts= [d["text"] for d in docs]

    embeddings= embed_model.encode(texts, show_progress_bar= True, batch_size= 128)

    embeddings= np.array(embeddings).astype("float16")
    dim= embeddings.shape[1]
    index= faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, INDEX_FILE))
    np.save(os.path.join(VECTORSTORE_DIR, EMBED_FILE), embeddings)


    