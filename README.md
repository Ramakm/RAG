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

- **Local first:** open-source embeddings and LLMs via Ollama. No API keys, and your data stays on your machine.
- **Production-minded:** they tackle the messy real-world parts, not just the happy path.
- **Self-contained:** every project lives in its own folder with its own README and a one-command launcher.

## Projects

| # | Project | What it does | Stack |
|:-:|---|---|---|
| 1 | [**BrowseAI**](./1.%20BrowseAI) | Turns any website URL into a searchable knowledge base you can chat with. Answers cite their source pages. | FastAPI · trafilatura · sentence-transformers · FAISS · Ollama |
| 2 | [**OmniRAG**](./2.%20OmniRAG) | Multimodal RAG over documents, images, audio and video. Ask in text or with an image; answers cite pages, pictures and timestamps. | FastAPI · PyMuPDF · Whisper · CLIP · Tesseract · FAISS · Ollama |

> More projects coming soon.

### 1. BrowseAI — Chat With Any Website

Paste a URL. BrowseAI crawls the site, strips navigation, footers and repeated boilerplate,
renders JavaScript pages when needed, then indexes the real content. You get streaming answers
with clickable citations back to the exact passages they came from.

```
URL ─► Crawl ─► Extract ─► Clean ─► Chunk ─► Embed ─► FAISS ─► Retrieve ─► Ollama ─► Cited answer
```

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

```
Files ─► Parse ─► Transcribe ─► Vision (CLIP · OCR · captions) ─► Embed ─► 2× FAISS ─► Fused retrieval ─► Ollama ─► Cited answer
```

```bash
ollama serve && ollama pull llama3.2      # optional: ollama pull moondream
cd "2. OmniRAG" && ./run.sh               # → http://localhost:8001
```

[Read the full OmniRAG docs →](./2.%20OmniRAG/README.md)

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
