"""
Interactive CLI demo for the bio-med RAG pipeline.

Usage:
    python scripts/query.py
    python scripts/query.py --K 100 --k 5 --model Qwen/Qwen3.5-4B
    python scripts/query.py --thinking --max_new_tokens 1024

Type your question at the prompt, then press Enter.
Type 'exit' or 'quit' to stop.
"""

import argparse
import sys
import textwrap
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vectorstore.store import VectorStore
from src.pipeline.rag_chain import RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Bio-Med RAG CLI demo")
    parser.add_argument("--K", type=int, default=100,
                        help="Number of FAISS candidates (default: 100)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of docs after reranking (default: 5)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B",
                        help="LLM model name (default: Qwen/Qwen3.5-4B)")
    parser.add_argument("--reranker", type=str, default="Qwen/Qwen3-Reranker-0.6B",
                        help="Reranker model name")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode for stronger reasoning (uses more VRAM & tokens)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max new tokens (default: 64 no-think, 1024 thinking)")
    parser.add_argument("--show_thinking", action="store_true",
                        help="Print the model's thinking chain (requires --thinking)")
    return parser.parse_args()


def print_result(result: dict, show_thinking: bool = False):
    if show_thinking and result.get("thinking"):
        print(f"\n{'─'*60}")
        print("  [Thinking]")
        print(textwrap.fill(
            result["thinking"], width=100,
            initial_indent="  ", subsequent_indent="  "
        ))
    print(f"\n{'='*60}")
    print(f"  Answer: {result['answer']}")
    print(f"  Retrieved: {result['retrieved']} candidates → {result['reranked']} reranked")
    print(f"{'─'*60}")
    print("  Sources:")
    for i, src in enumerate(result["sources"], 1):
        score = f"{src['reranker_score']:.3f}" if src["reranker_score"] is not None else "n/a"
        preview = textwrap.shorten(src["preview"], width=200, placeholder="...")
        print(f"  [{i}] score={score} | {src['source']}")
        print(f"       {preview}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    print("Loading vectorstore...")
    store = VectorStore(device="cuda")

    print(f"Loading pipeline (LLM={args.model}, Reranker={args.reranker}, thinking={args.thinking})...")
    pipeline = RAGPipeline(
        store=store,
        K=args.K,
        k=args.k,
        reranker_model=args.reranker,
        llm_model=args.model,
        thinking=args.thinking,
        max_new_tokens=args.max_new_tokens,
    )

    mode_label = "[THINKING ON]" if args.thinking else "[THINKING OFF]"
    print(f"\nBio-Med RAG ready {mode_label}. Type your question (or 'exit' to quit).\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        result = pipeline.run(question)
        print_result(result, show_thinking=args.show_thinking)


if __name__ == "__main__":
    main()
