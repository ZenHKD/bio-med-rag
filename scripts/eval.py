"""
Evaluation script for the bio-med RAG pipeline.

Usage:
    # 100 MedQA questions, fast eval (~3-4s/q)
    python scripts/eval.py

    # Full MedMCQA eval
    python scripts/eval.py --dataset medmcqa --n 0 --output results/medmcqa_all.json

    # Custom settings
    python scripts/eval.py --dataset medqa --n 200 --K 20 --k 5 --output results/out.json

Results are always printed to stdout after the run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vectorstore.store import VectorStore
from src.pipeline.rag_chain import RAGPipeline
from src.evaluation.evaluator import (
    load_medqa,
    load_medmcqa,
    evaluate,
    EvalResult,
    MCQItem,
    ItemResult,
)


# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Bio-Med RAG evaluation")
    parser.add_argument("--dataset", choices=["medqa", "medmcqa"], default="medqa",
                        help="Benchmark dataset (default: medqa)")
    parser.add_argument("--n", type=int, default=100,
                        help="Questions to sample; 0 = all (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--K", type=int, default=20,
                        help="FAISS candidate pool size (default: 20 for speed)")
    parser.add_argument("--k", type=int, default=5,
                        help="Docs kept after reranking (default: 5)")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Max tokens for LLM (default: 10 — only need 1 answer letter)")
    parser.add_argument("--reranker_batch", type=int, default=8,
                        help="Reranker scoring batch size (default: 8)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B",
                        help="LLM model name")
    parser.add_argument("--reranker", type=str, default="Qwen/Qwen3-Reranker-0.6B",
                        help="Reranker model name")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
def print_results(result: EvalResult):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Dataset  : {result.dataset}")
    print(f"  Total    : {result.n_total}")
    print(f"  Correct  : {result.n_correct}")
    print(f"  Accuracy : {result.accuracy * 100:.2f}%")
    print(f"{'─' * 60}")
    print(f"  {'Label':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for label in sorted(result.precision.keys()):
        p  = result.precision.get(label, 0)
        r  = result.recall.get(label, 0)
        f1 = result.f1.get(label, 0)
        print(f"  {label:<8} {p * 100:>9.2f}% {r * 100:>9.2f}% {f1 * 100:>9.2f}%")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
def save_results(result: EvalResult, path: str):
    out = {
        "dataset":   result.dataset,
        "n_total":   result.n_total,
        "n_correct": result.n_correct,
        "accuracy":  result.accuracy,
        "precision": result.precision,
        "recall":    result.recall,
        "f1":        result.f1,
        "per_item": [
            {
                "question":  r.question[:200],   # truncate for readability
                "gold":      r.gold,
                "predicted": r.predicted,
                "correct":   r.correct,
            }
            for r in result.per_item
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {path}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    n    = args.n if args.n > 0 else None

    # 1. Load dataset
    print(f"Loading dataset: {args.dataset} (n={'all' if n is None else n})")
    if args.dataset == "medqa":
        items = load_medqa(n=n, seed=args.seed)
    else:
        items = load_medmcqa(n=n, seed=args.seed)
    print(f"  → {len(items)} questions")

    # 2. Build pipeline (thinking=False for eval speed)
    print("\nLoading vectorstore...")
    store = VectorStore(device="cuda")

    print(f"Loading pipeline (LLM={args.model}, Reranker={args.reranker}, K={args.K}, max_new_tokens={args.max_new_tokens})...")
    pipeline = RAGPipeline(
        store=store,
        K=args.K,
        k=args.k,
        reranker_model=args.reranker,
        llm_model=args.model,
        thinking=False,
        max_new_tokens=args.max_new_tokens,
        reranker_batch=args.reranker_batch,
    )

    # 3. Evaluate with live tqdm progress bar
    print(f"\nEvaluating {len(items)} questions...\n")
    n_correct = 0

    pbar = tqdm(
        total=len(items),
        desc="Evaluating",
        unit="q",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] acc={postfix}",
    )

    def on_item(i: int, item: MCQItem, result: ItemResult):
        nonlocal n_correct
        if result.correct:
            n_correct += 1
        acc = n_correct / (i + 1)
        pbar.set_postfix_str(f"{acc * 100:.1f}%")
        pbar.update(1)

    t0     = time.time()
    result = evaluate(pipeline, items, dataset_name=args.dataset, progress_callback=on_item)
    pbar.close()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed / 60:.1f} min ({elapsed / len(items):.1f}s / question)")

    # 4. Print and optionally save
    print_results(result)

    if args.output:
        save_results(result, args.output)


if __name__ == "__main__":
    main()
