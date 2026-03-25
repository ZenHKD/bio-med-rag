"""
RAG Pipeline (Phase 1 — HuggingFace, for benchmarking).

Flow:
    question
      -> dense_retrieve (Top-K FAISS candidates)
      -> rerank (Qwen3-Reranker-0.6B, Top-k)
      -> build context string
      -> Decoder.generate (Qwen3.5-4B 4-bit NF4)
      -> answer letter (A–E)

Phase 2: swap this file with src/serving/ (LangChain + vLLM).
"""
from typing import List, Optional

from langchain_core.documents import Document

from src.vectorstore.store import VectorStore
from src.retriever.retriever import retrieve_and_rerank, _get_reranker, DEFAULT_K_RETRIEVE, DEFAULT_K_RERANK
from src.llm.decoder import Decoder, DEFAULT_MODEL, DEFAULT_PROMPT

# Max characters per chunk included in context sent to the LLM
_MAX_CHUNK_CHARS = 1500

def _build_context(docs: List[Document], max_chars: int = _MAX_CHUNK_CHARS) -> str:
    """Concatenate reranked docs into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content.strip()[:max_chars]
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)


class RAGPipeline:
    """
    End-to-end RAG pipeline:  retrieve -> rerank -> generate.

    Args:
        store:          Loaded VectorStore (FAISS index + chunks).
        K:              Number of FAISS candidates retrieved (default 100).
        k:              Number of documents kept after reranking (default 5).
        reranker_model: HuggingFace model ID for the reranker.
        llm_model:      HuggingFace model ID for the LLM decoder.
        prompt_path:    Path to prompt.txt template.
        max_new_tokens: Max tokens the LLM generates.
    """

    def __init__(
        self,
        store: VectorStore,
        K: int = DEFAULT_K_RETRIEVE,
        k: int = DEFAULT_K_RERANK,
        reranker_model: str = "Qwen/Qwen3-Reranker-0.6B",
        llm_model: str = DEFAULT_MODEL,
        prompt_path: str = DEFAULT_PROMPT,
        max_new_tokens: int = None,
        thinking: bool = False,
        reranker_batch: int = 8,
    ):
        self.store          = store
        self.K              = K
        self.k              = k
        self.reranker_model = reranker_model
        self.reranker_batch = reranker_batch
        self.decoder = Decoder(
            model_name=llm_model,
            prompt_path=prompt_path,
            max_new_tokens=max_new_tokens,
            thinking=thinking,
        )
        # Pre-load the reranker now so both models are ready before the first query
        print(f"[RAGPipeline] Pre-loading reranker...")
        _get_reranker(reranker_model)

    def run(self, question: str, max_new_tokens: int = None) -> dict:
        """
        Run the full pipeline for a single question.

        Args:
            question: Formatted MCQ string (e.g. "Which drug...?\nA. ...\nB. ...").

        Returns:
            {
                "answer":    "A",          # predicted letter
                "sources":   [...],        # list of source metadata dicts
                "retrieved": 100,          # K dense candidates
                "reranked":  5,            # k after reranking
            }
        """
        # 1. Dense retrieve + 2. Rerank (single FAISS search)
        reranked = retrieve_and_rerank(
            store=self.store,
            query=question,
            K=self.K,
            k=self.k,
            reranker_model=self.reranker_model,
            reranker_batch=self.reranker_batch,
        )

        # 3. Build context
        context = _build_context(reranked)

        # 4. Generate
        gen = self.decoder.generate(context=context, question=question, max_new_tokens=max_new_tokens)

        # 5. Collect source metadata
        sources = [
            {
                "chunk_id":       doc.metadata.get("chunk_id", ""),
                "source":         doc.metadata.get("source", ""),
                "reranker_score": doc.metadata.get("reranker_score", None),
                "preview":        doc.page_content[:200].replace("\n", " "),
            }
            for doc in reranked
        ]

        return {
            "answer":    gen["answer"],
            "thinking":  gen["thinking"],
            "sources":   sources,
            "retrieved": self.K,
            "reranked":  len(reranked),
        }
