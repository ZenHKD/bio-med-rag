from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import CrossEncoderReranker

from src.vectorstore.store import VectorStore


def _chunks_to_documents(store: VectorStore) -> List[Document]:
    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "source": chunk.get("source", ""),
            },
        )
        for chunk in store.chunks
    ]


def get_baseline_retriever(store: VectorStore, k: int = 5):
    return store.as_retriever(k=k)


def get_hybrid_retriever(store: VectorStore, k: int = 5, bm25_weight: float = 0.5):
    dense_retriever = store.as_retriever(k=k)

    documents = _chunks_to_documents(store)
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, 1 - bm25_weight],
    )


def get_reranking_retriever(
    store: VectorStore,
    k: int = 10,
    top_n: int = 3,
    bm25_weight: float = 0.5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
):
    hybrid = get_hybrid_retriever(store, k=k, bm25_weight=bm25_weight)

    cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid,
    )
