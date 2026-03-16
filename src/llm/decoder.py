"""
Decoder module — LLM client for bio-med-rag.

Connects to a local vLLM server via its OpenAI-compatible API.

Usage:
    from src.llm.decoder import get_decoder

    llm = get_decoder()
    response = llm.invoke("What is hypertension?")
"""

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL    = "http://localhost:8080/v1"
DEFAULT_MODEL_NAME  = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS  = 512


def get_decoder(
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ChatOpenAI:
    """Build a ChatOpenAI instance pointing to the local vLLM server."""
    return ChatOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
