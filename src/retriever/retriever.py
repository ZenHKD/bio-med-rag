"""
Retriever module: Dense retrieval (FAISS) + Qwen3-Reranker-0.6B reranking.

Pipeline:
    query → dense_retrieve (Top-K=100) → rerank (Top-k=5) → List[Document]
"""

import torch
from typing import List, Optional

from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.vectorstore.store import VectorStore


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_K_RETRIEVE = 100   # number of FAISS candidates
DEFAULT_K_RERANK   = 5     # number of final results after reranking

# Qwen3-Reranker uses a specific yes/no token scoring prompt
_RERANKER_SYSTEM = (
    "Judge whether the Document is relevant to the Query. "
    "Output only 'yes' or 'no'."
)
_RERANKER_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "Query: {query}\n\n"
    "Document: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


# ---------------------------------------------------------------------------
# Reranker (lazy singleton to avoid reloading model on every call)
# ---------------------------------------------------------------------------
class _Reranker:
    """
    Wraps Qwen3-Reranker-0.6B.
    Scores each (query, document) pair by comparing logits of 'yes' vs 'no'
    tokens at the final position of the prompt.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Reranker] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()

        # Resolve yes/no token ids once
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no  = self.tokenizer.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(self, query: str, documents: List[str], batch_size: int = 8) -> List[float]:
        """Return a relevance score in [0, 1] for each document, scored in batches."""
        all_scores: List[float] = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            prompts = [
                _RERANKER_TEMPLATE.format(
                    system=_RERANKER_SYSTEM,
                    query=query,
                    document=doc,
                )
                for doc in batch_docs
            ]

            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=1024,       
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits[:, -1, :]  # (B, vocab)

            yes_logits = logits[:, self.token_yes]
            no_logits  = logits[:, self.token_no]

            scores = torch.softmax(
                torch.stack([yes_logits, no_logits], dim=-1), dim=-1
            )[:, 0]  # P(yes)

            all_scores.extend(scores.float().tolist())

        return all_scores


_reranker_instance: Optional[_Reranker] = None


def _get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> _Reranker:
    """Returns a cached singleton reranker (loads model only once)."""
    global _reranker_instance
    if _reranker_instance is None or _reranker_instance.model.name_or_path != model_name:
        _reranker_instance = _Reranker(model_name)
    return _reranker_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def dense_retrieve(store: VectorStore, query: str, K: int = DEFAULT_K_RETRIEVE) -> List[Document]:
    """
    FAISS top-K retrieval.

    Args:
        store: Loaded VectorStore.
        query: Raw question string.
        K:     Number of candidates to retrieve.

    Returns:
        List of up to K Documents sorted by FAISS distance (ascending = more similar).
    """
    return store.search(query, k=K)


def rerank(
    docs: List[Document],
    query: str,
    k: int = DEFAULT_K_RERANK,
    model_name: str = DEFAULT_RERANKER_MODEL,
    reranker_batch: int = 8,
) -> List[Document]:
    """
    Rerank documents with Qwen3-Reranker-0.6B and return the top-k.
    """
    if not docs:
        return []

    reranker = _get_reranker(model_name)
    texts  = [doc.page_content for doc in docs]
    scores = reranker.score(query, texts, batch_size=reranker_batch)

    # Attach reranker score to metadata and sort
    scored = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, doc in scored[:k]:
        doc.metadata["reranker_score"] = round(score, 4)
        results.append(doc)

    return results


def retrieve_and_rerank(
    store: VectorStore,
    query: str,
    K: int = DEFAULT_K_RETRIEVE,
    k: int = DEFAULT_K_RERANK,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    reranker_batch: int = 8,
) -> List[Document]:
    """
    Full pipeline convenience function: dense retrieve → rerank.
    """
    candidates = dense_retrieve(store, query, K=K)
    return rerank(candidates, query, k=k, model_name=reranker_model, reranker_batch=reranker_batch)
