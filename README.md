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

## 2. Personal config override

> **Do not edit `configs/config.yaml` directly** — it holds shared defaults and is git-tracked.

```bash
cp configs/config.local.yaml.example configs/config.local.yaml
```

`configs/config.local.yaml` is gitignored. Open it and uncomment only the fields you want to change, for example:

```yaml
encoder:
  model_name: "BAAI/bge-base-en-v1.5"
  device: cuda

decoder:
  model_name: "BioMistral/BioMistral-7B"
  device: cuda

reranker:
  enabled: true
  provider: cross_encoder
  top_n: 3
```

Anything not specified here falls back to `configs/config.yaml`.

## 3. Download datasets

```bash
python set_up_dataset.py
```

## 4. Preprocess (pubmedqa, medqa textbooks and bioasq) and concat datasets (pubmedqa & medqa)

```bash
python src/data/preprocess.py
```


## 5. Build vectorstore & run a query

```bash
python scripts/ingest.py
python scripts/query.py --question "What is BCR-ABL?"
```