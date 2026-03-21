## This repo is created for the Final Project Presentation of the NLP course in USTH

Members:
- Pham Quang Vinh
- Hoang Khanh Dong
- Nguyen Lam Tung
- Pham Duy Anh
- Nguyen Vu Hong Ngoc
- Le Chi Thanh Lam

# Repository Structure

```
bio-med-rag/
├── data/
│   ├── external/                 # raw datasets (gitignored)
│   └── vectorstore/              # persisted embeddings (gitignored)
├── notebooks/                    # exploration notebooks
├── scripts/
│   ├── ingest.py                 # ingest data
│   └── query.py                  # run RAG demo in CLI
├── src/
│   ├── data/                     # data processing
│   ├── embeddings/               # embedding models
│   ├── vectorstore/              # vector store
│   ├── retriever/                # retriever
│   ├── llm/                      # LLM
│   ├── pipeline/                 # RAG pipeline (fully deploy with LangChain and vLLM)
│   ├── evaluation/               # evaluation
│   └── serving/                  # serving
├── .gitignore
├── requirements.txt
├── set_up_dataset.py
└── README.md
```

# Pipeline

Query → [Encoder] → Retrieve Chunks → [Reranker] → [Decoder LLM] → Generated Answer
