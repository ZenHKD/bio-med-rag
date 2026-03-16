import json
import os
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from src.embeddings.encoder import Encoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_INDEX_PATH  = str(PROJECT_ROOT / "data" / "vectorstore" / "index.bin")
DEFAULT_CHUNKS_PATH = str(PROJECT_ROOT / "data" / "processed" / "knowledge" / "chunks.jsonl")
DEFAULT_EMBED_MODEL = "NeuML/pubmedbert-base-embeddings"
DEFAULT_BATCH_SIZE  = 16


class VectorStore:

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
    ):
        self.embed_model_name = embed_model_name
        self.batch_size = batch_size
        self.device = device

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self.index = faiss.read_index(index_path)

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)

    def search(self, query: str, k: int = 5) -> List[Document]:
        encoder = Encoder([query], self.batch_size, self.embed_model_name, self.device)
        q_vec = encoder.encode().astype("float32")

        distances, indices = self.index.search(q_vec, k)

        docs: List[Document] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            docs.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "doc_id": chunk.get("doc_id", ""),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "source": chunk.get("source", ""),
                        "score": float(dist),
                    },
                )
            )
        return docs

    def as_retriever(self, k: int = 5) -> "FAISSRetriever":
        return FAISSRetriever(store=self, k=k)


class FAISSRetriever(BaseRetriever):

    store: VectorStore = Field(exclude=True)
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        return self.store.search(query, k=self.k)
