<div align="center">

# RAG — End-to-End Projects

**Hands-on Retrieval-Augmented Generation projects, built to run locally with open-source models.**

<a href="#projects"><img src="https://skillicons.dev/icons?i=py,fastapi,pytorch,js,html,css,git&perline=7" alt="Skills" /></a>

<br />

![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Playwright](https://img.shields.io/badge/Playwright-2EAD33?style=flat-square)
![Whisper](https://img.shields.io/badge/Whisper-000000?style=flat-square&logo=openai&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-000000?style=flat-square)
![Llama](https://img.shields.io/badge/Llama_3.2-0467DF?style=flat-square&logo=meta&logoColor=white)

![Stars](https://img.shields.io/github/stars/Ramakm/RAG?style=flat-square)
![Forks](https://img.shields.io/github/forks/Ramakm/RAG?style=flat-square)
![Issues](https://img.shields.io/github/issues/Ramakm/RAG?style=flat-square)
![License](https://img.shields.io/github/license/Ramakm/RAG?style=flat-square)

</div>

---

## Overview

Most RAG tutorials stop at "upload a PDF". These projects go further. Each one is a complete,
working application covering **ingestion → cleaning → chunking → embeddings → vector search →
grounded generation**, with a usable UI on top.

- **Local first:** open-source embeddings and LLMs via Ollama. No API keys, and your data stays on your machine (Agentic RAG is the exception and uses an Anthropic API key).
- **Production-minded:** they tackle the messy real-world parts, not just the happy path.
- **Self-contained:** every project lives in its own folder with its own README and a one-command launcher.

## Projects

| # | Project | What it does | Stack |
|:-:|---|---|---|
| 1 | [**BrowseAI**](./1.%20BrowseAI) | Turns any website URL into a searchable knowledge base you can chat with. Answers cite their source pages. | FastAPI · trafilatura · sentence-transformers · FAISS · Ollama |
| 2 | [**OmniRAG**](./2.%20OmniRAG) | Multimodal RAG over documents, images, audio and video. Ask in text or with an image; answers cite pages, pictures and timestamps. | FastAPI · PyMuPDF · Whisper · CLIP · Tesseract · FAISS · Ollama |
| 3 | [**Agentic RAG**](./3.%20Agentic%20RAG) | A LangGraph agent that plans, retrieves from vector search and/or guarded SQL, grades evidence, retries weak queries and fact-checks its cited answers. | LangGraph · Chroma · FastAPI · Streamlit · sqlglot · Claude |
| 📚 | [**RAG Fundamentals**](./RAG%20Fundamentals) | 39 notebooks that walk through RAG techniques, from a simple baseline to GraphRAG, Self-RAG and evaluation. | Jupyter · sentence-transformers · rank_bm25 · networkx · Ollama |

> More projects coming soon.

### 1. BrowseAI — Chat With Any Website

Paste a URL. BrowseAI crawls the site, strips navigation, footers and repeated boilerplate,
renders JavaScript pages when needed, then indexes the real content. You get streaming answers
with clickable citations back to the exact passages they came from.

<img width="1983" height="793" alt="image" src="https://github.com/user-attachments/assets/85a8fced-7b39-46f3-b8b7-339d3fb088d2" />


```bash
ollama serve && ollama pull llama3.2
cd "1. BrowseAI" && ./run.sh        # → http://localhost:8000
```

[Read the full BrowseAI docs →](./1.%20BrowseAI/README.md)

### 2. OmniRAG — Multimodal RAG

Drop in PDFs, slides, photos, recordings and videos. OmniRAG parses documents and their figures,
OCRs scans and screenshots, transcribes speech with Whisper and embeds images and video frames with
CLIP. One search spans everything, so you can ask with text or attach an image. Answers cite the
exact page, picture or timestamp, and media sources play from the cited moment.

<img width="1983" height="793" alt="Multimodal RAG Pipeline Architecture" src="https://github.com/user-attachments/assets/a040877e-8908-4981-a3a1-6895de6d78dd" />


```bash
ollama serve && ollama pull llama3.2      # optional: ollama pull moondream
cd "2. OmniRAG" && ./run.sh               # → http://localhost:8001
```

[Read the full OmniRAG docs →](./2.%20OmniRAG/README.md)

### 3. Agentic RAG — Plan, Retrieve, Verify

Ask questions in plain English across reports, specs, manuals, tickets **and** a sales/returns
database. A LangGraph agent plans the query, picks vector search and/or guarded SQL, grades the
evidence, rewrites weak queries (up to 3 retries), writes a cited answer and fact-checks it against
the sources. SQL runs through an allowlist guard with a forced `LIMIT`, a read-only connection and a timeout.

```bash
cd "3. Agentic RAG"
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env              # add your ANTHROPIC_API_KEY
make demo && make ui              # → http://localhost:8501
```

[Read the full Agentic RAG docs →](./3.%20Agentic%20RAG/README.md)

### 📚 RAG Fundamentals — 39 Techniques, One Notebook Each

A guided learning path through the RAG techniques that matter, ordered from the simple baseline to
advanced architectures. Every technique lives in its own folder with a runnable notebook, all sharing
one small local toolkit (`common.py`) and sample data.

- **Basics:** simple, CSV and JSON RAG, reliable RAG, chunk size, proposition chunking
- **Evaluation:** DeepEval, GroUSE-style judges, end-to-end and reference-free evaluation
- **Query and context side:** query transformations, HyDE, HyPE, semantic chunking, compression, RSE
- **Retrieval:** fusion (BM25 + vectors), reranking, MMR / Dartboard, hierarchical indices, multimodal (captioning, ColPali)
- **Iterative and advanced:** feedback loops, adaptive retrieval, CRAG, Self-RAG, GraphRAG, LightRAG, RAPTOR, MemoRAG, agentic RAG

```bash
ollama serve && ollama pull llama3.2
cd "RAG Fundamentals" && pip install -r requirements.txt && jupyter lab
```

[Browse the full learning path →](./RAG%20Fundamentals/README.md)

## Getting started

```bash
git clone https://github.com/Ramakm/RAG.git
cd RAG
```

Each project folder has its own setup instructions. You'll need **Python 3.10+** and
**[Ollama](https://ollama.com/download)** with at least one model pulled.

## Contributing

Ideas, issues and pull requests are welcome. If a project helped you, a ⭐ goes a long way.

## License

Released under the [MIT License](./LICENSE).

## Connect with me

[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/techwith_ram)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ramakm)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ramakrushnamohapatra/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwith.ram)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@techwithram)
