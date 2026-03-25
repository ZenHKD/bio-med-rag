import sys
import time
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Biomedical RAG Demo",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading vector store...")
def load_store(device: str):
    from src.vectorstore.store import VectorStore

    return VectorStore(device=device)


@st.cache_resource(show_spinner="Loading pipeline (LLM + reranker)...")
def load_pipeline(
    _store,
    K: int,
    k: int,
    llm_model: str,
    reranker_model: str,
    thinking: bool,
):
    from src.pipeline.rag_chain import RAGPipeline

    return RAGPipeline(
        store=_store,
        K=K,
        k=k,
        llm_model=llm_model,
        reranker_model=reranker_model,
        thinking=thinking,
    )


with st.sidebar:
    st.header("Configuration")

    st.subheader("Hardware")
    device = st.selectbox("Device", ["cuda", "cpu"], index=0)

    st.subheader("Models")
    llm_model = st.text_input("LLM model", value="Qwen/Qwen3.5-4B")
    reranker_model = st.text_input("Reranker model", value="Qwen/Qwen3-Reranker-0.6B")

    st.subheader("Retrieval")
    K = st.slider("Dense retrieval top-K (FAISS candidates)", 10, 200, 100, step=10)
    k = st.slider("Reranked top-k passed to LLM", 1, 20, 5)

    st.subheader("Generation")
    thinking = st.toggle(
        "Enable thinking mode",
        value=False,
        help="Reasoning chain before final answer. Uses more VRAM & tokens.",
    )
    show_thinking = st.toggle(
        "Show thinking chain",
        value=False,
        disabled=not thinking,
        help="Display the model's reasoning. Requires thinking mode.",
    )
    max_new_tokens = st.number_input(
        "Max new tokens (0 = auto)",
        min_value=0,
        max_value=4096,
        value=0,
        help="0 = 64 (no-think) or 2048 (thinking), matching decoder.py defaults.",
    )

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Data paths** (set in store.py)")
    st.code(
        "data/vectorstore/index.bin\ndata/vectorstore/metadata.jsonl",
        language="text",
    )
    st.caption("Run `python scripts/ingest.py` to build the index.")


try:
    store = load_store(device)
    pipeline = load_pipeline(
        store,
        K,
        k,
        llm_model,
        reranker_model,
        thinking,
    )
except FileNotFoundError as e:
    st.error(
        f"**File not found:** {e}\n\n"
        "Build the index first:\n```\npython scripts/ingest.py\n```"
    )
    st.stop()
except Exception as e:
    st.error(f"**Failed to load pipeline:** {e}")
    st.stop()


st.title("Biomedical Information Retrieval System")
st.caption("Retrieval-Augmented Generation")

mode_badge = "Thinking ON" if thinking else "Thinking OFF"
st.caption(f"Model: `{llm_model}` · Reranker: `{reranker_model}` · {mode_badge}")


submitted_question: str | None = None
option_map: dict | None = None

tab_mcq, tab_free = st.tabs(["MCQ (A / B / C / D)", "Free-text question"])

with tab_mcq:
    st.caption("Matches MedQA / MedMCQA evaluation format.")
    with st.form("mcq_form"):
        stem = st.text_area(
            "Question stem / clinical scenario",
            placeholder="e.g. A 45-year-old man presents with crushing chest pain...",
            height=110,
        )
        col1, col2 = st.columns(2)
        with col1:
            opt_a = st.text_input("A.")
            opt_b = st.text_input("B.")
        with col2:
            opt_c = st.text_input("C.")
            opt_d = st.text_input("D.")
        mcq_submit = st.form_submit_button("Retrieve & Answer", type="primary")

    if mcq_submit and stem.strip():
        option_map = {
            key: val
            for key, val in {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}.items()
            if val.strip()
        }
        opts_str = "\n".join(f"{key}. {val}" for key, val in option_map.items())
        submitted_question = f"{stem.strip()}\n{opts_str}" if opts_str else stem.strip()

with tab_free:
    st.caption(
        "Open-ended question - context is still retrieved; the LLM answers freely."
    )
    with st.form("free_form"):
        free_q = st.text_area(
            "Question",
            placeholder="e.g. What is the mechanism of action of beta-blockers?",
            height=110,
        )
        free_submit = st.form_submit_button("Retrieve & Answer", type="primary")

    if free_submit and free_q.strip():
        submitted_question = free_q.strip()
        option_map = None


def render_sources(sources: list, retrieved: int, reranked: int, latency: float):
    label = f"{retrieved} candidates → {reranked} reranked · {latency:.1f}s"
    with st.expander(label):
        if not sources:
            st.info("No source metadata returned.")
            return
        for i, src in enumerate(sources, 1):
            score = src.get("reranker_score")
            score_str = f"score: **{score:.3f}**" if score is not None else "score: n/a"
            st.markdown(
                f"**[{i}]** {score_str} · "
                f"`{src.get('chunk_id', '-')}` · "
                f"*{src.get('source', '-')}*"
            )
            st.caption(src.get("preview", ""))
            if i < len(sources):
                st.divider()


def render_assistant_msg(msg: dict):
    answer = msg["answer"]
    opts = msg.get("option_map")
    chain = msg.get("thinking", "")
    sources = msg.get("sources", [])
    retrieved = msg.get("retrieved", 0)
    reranked = msg.get("reranked", 0)
    latency = msg.get("latency", 0.0)

    if opts and answer in opts:
        st.success(f"**{answer}** - {opts[answer]}")
    elif opts and answer == "E":
        st.warning("**E** - Retrieved context was insufficient to select an option.")
    elif answer == "?":
        st.error("The model did not return a parseable answer.")
    else:
        st.markdown(answer)

    if chain and show_thinking:
        with st.expander("Thinking chain"):
            st.markdown(chain)

    render_sources(sources, retrieved, reranked, latency)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant_msg(msg)


if submitted_question:
    with st.chat_message("user"):
        st.markdown(submitted_question)
    st.session_state.messages.append({"role": "user", "content": submitted_question})

    with st.chat_message("assistant"):
        with st.spinner("Searching corpus and generating answer…"):
            t0 = time.perf_counter()
            _tokens = (
                int(max_new_tokens) if max_new_tokens and max_new_tokens > 0 else None
            )
            result = pipeline.run(submitted_question, max_new_tokens=_tokens)
            elapsed = time.perf_counter() - t0

        assistant_msg = {
            "role": "assistant",
            "answer": result["answer"],
            "thinking": result.get("thinking", ""),
            "sources": result.get("sources", []),
            "retrieved": result.get("retrieved", K),
            "reranked": result.get("reranked", k),
            "option_map": option_map,
            "latency": elapsed,
        }
        render_assistant_msg(assistant_msg)

    st.session_state.messages.append(assistant_msg)
