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
├── configs/
│   ├── config.yaml               # base config (git-tracked)
│   └── config.local.yaml.example # contributor override template
├── data/
│   ├── external/                 # raw datasets (gitignored)
│   └── vectorstore/              # persisted embeddings (gitignored)
├── notebooks/                    # exploration notebooks
├── scripts/
│   ├── ingest.py                 # ingest data
│   └── query.py                  # run RAG query
├── src/
│   ├── config/
│   ├── data/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retriever/
│   ├── llm/
│   ├── pipeline/
│   └── evaluation/
├── .env.example
├── .gitignore
├── requirements.txt
├── set_up_dataset.py
└── README.md
```

# Pipeline

Query → [Encoder] → Retrieve Chunks → [Reranker?] → [Decoder LLM] → Generated Answer


# Setup

## 1. Environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
HF_TOKEN=hf_your_token_here
# OPENAI_API_KEY=...   (optional, only if using OpenAI encoder/decoder)
# COHERE_API_KEY=...   (optional, only if using Cohere reranker)
```

## 2. Download datasets

```bash
python set_up_dataset.py
```

## 3. Preprocess (pubmedqa, medqa textbooks and bioasq) and concat datasets (pubmedqa & medqa)

```bash
python src/data/preprocess.py
```

## 4. Chunking 

```bash
python src/data/chunker.py
```

## 5. Indexing & Embedding

```bash
python scripts/ingest.py
```

## 6. Build vectorstore & run a query

```bash
python scripts/ingest.py
python scripts/query.py --question "What is BCR-ABL?"
```