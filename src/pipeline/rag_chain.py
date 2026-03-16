from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.llm.decoder import get_decoder
from src.vectorstore.store import VectorStore
from src.retriever.retriever import (
    get_baseline_retriever,
    get_reranking_retriever,
)

MEDICAL_PROMPT = ChatPromptTemplate.from_template(
    "You are a biomedical expert assistant.\n"
    "Answer the question using ONLY the information in the provided context.\n"
    'If the context does not contain enough information, say "I don\'t know."\n'
    "Be concise and precise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    "Answer the following question based on the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain(retriever, prompt, decoder):
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | decoder
        | StrOutputParser()
    )
    return chain, retriever


def get_baseline_chain(store: VectorStore, k: int = 5, **decoder_kwargs):
    decoder = get_decoder(**decoder_kwargs)
    retriever = get_baseline_retriever(store, k=k)
    return _build_chain(retriever, DEFAULT_PROMPT, decoder)


def get_custom_chain(
    store: VectorStore,
    k: int = 10,
    top_n: int = 3,
    bm25_weight: float = 0.5,
    **decoder_kwargs,
):
    decoder = get_decoder(**decoder_kwargs)
    retriever = get_reranking_retriever(
        store, k=k, top_n=top_n, bm25_weight=bm25_weight
    )
    return _build_chain(retriever, MEDICAL_PROMPT, decoder)


def query(chain, retriever, question: str) -> dict:
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return {"result": answer, "source_documents": source_docs}
