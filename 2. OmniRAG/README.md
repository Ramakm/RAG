# OmniRAG — Multimodal RAG Over Documents, Images, Audio and Video

Drop in PDFs, slides, photos, recordings or videos. OmniRAG reads the text, looks at the images,
transcribes the speech and indexes all of it in one collection. Ask a question in text, or attach an
image, and get answers that cite the exact **page**, **picture** or **timestamp**. You can open
or play each source right in the chat.

Everything runs locally: Whisper for speech, CLIP for images, MiniLM for text, FAISS for search
and an open-source LLM served by [Ollama](https://ollama.com).

<img width="1162" height="670" alt="Screenshot 2026-10-15 at 5 33 08 PM" src="https://github.com/user-attachments/assets/c10883ae-b02a-442a-a9e5-32308ce96ffc" />


```
                 ┌────────────────────────── INGESTION PIPELINE ──────────────────────────┐
  PDF/DOCX/PPTX ─┤ Parse ── text by page/slide · embedded figures · scanned pages ──┐     │
  PNG/JPG       ─┤ Vision ─ CLIP embedding · OCR · vision-model caption (optional) ──┼─► Embed ─► FAISS
  MP3/WAV/M4A   ─┤ Transcribe ─ Whisper + VAD → timestamped transcript windows ──────┤  MiniLM   text index
  MP4/MOV/WEBM  ─┤ Transcribe + sample frames → CLIP, near-duplicate frames dropped ─┘  CLIP     visual index
                 └────────────────────────────────────────────────────────────────────────┘
                                                                                     │
  "What did they decide?"  ─► text search (MiniLM) ─┐                                 │
  "the revenue chart"      ─► text→image (CLIP)    ─┼─► calibrated fusion ─► top-k mixed sources
  [attached image]         ─► image→image (CLIP)   ─┘                                 │
                                                                                     ▼
                    cited answer ◄── Ollama LLM (streamed; vision models also see the images)
```

## Quick start

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com/download). Optional:
[Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR.

```bash
# 1. LLM server and a chat model (one time)
ollama serve                 # leave running in its own terminal
ollama pull llama3.2         # any chat model works

# 2. Optional extras
ollama pull moondream        # small vision model: image captions + answers that "see" images
brew install tesseract       # or: sudo apt install tesseract-ocr  (reads text in images and scans)

# 3. Start OmniRAG (creates .venv and installs everything on first run)
./run.sh
```

Open **http://localhost:8001**, drop some files, press **Build collection**, then **Start chatting**.

> OmniRAG runs on port 8001 so it can sit next to BrowseAI (8000). The first build downloads
> the CLIP (~600 MB), Whisper base (~150 MB) and MiniLM (~90 MB) models once.

## What each modality goes through

| Input | Formats | What OmniRAG extracts | Cited as |
|---|---|---|---|
| **Documents** | PDF, DOCX, PPTX, TXT, MD, CSV, TSV, JSON | Text by page or slide (incl. tables and speaker notes), embedded figures, OCR of scanned pages | `report.pdf · Page 3 › Results` + page thumbnail |
| **Images** | PNG, JPG, WEBP, GIF, BMP, TIFF | CLIP visual embedding, OCR text, vision-model description | Image thumbnail, full-size viewer |
| **Audio** | MP3, WAV, M4A, OGG, FLAC, AAC, OPUS | Whisper transcript in ~45 s windows with timestamps | `talk.m4a · 02:15–03:00` + inline player |
| **Video** | MP4, MOV, MKV, WEBM, AVI | Transcript windows **and** sampled frames (CLIP + OCR + caption) | `demo.mp4 · 01:20` + player that starts there |

## How ingestion handles messy real-world files

1. **Parse** ([backend/parsers.py](backend/parsers.py))
   - PDFs are read block by block, and lines wrapped mid-paragraph are rejoined.
   - **Running headers/footers** (any line at the top or bottom of at least half the pages) and bare page numbers are removed.
   - A page with almost no text layer is treated as a **scan**. It's rendered at 2× and sent to OCR.
   - **Embedded figures** are extracted from PDF/DOCX/PPTX. Icons, bullets, thin rules and repeated logos are skipped by size, aspect ratio and hash.
   - CSV rows become `column: value` lines, so every chunk describes itself.
2. **Transcribe** ([backend/media.py](backend/media.py))
   - [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (int8 on CPU) with **voice-activity detection**, so silence and music beds are skipped.
   - Segments are grouped into ~45-second windows that overlap by one segment.
3. **Vision** ([backend/vision.py](backend/vision.py))
   - Video frames are sampled every N seconds (spread wider for long videos, max 120 per video).
   - Every visual gets a **CLIP** embedding. **Near-duplicates are dropped** using that embedding: static video shots (cosine > 0.95 to the previous kept frame) and logos repeated across slides (> 0.97).
   - **OCR** keeps only words Tesseract is confident about, so photos don't turn into gibberish.
   - With a vision model in Ollama (moondream, llava, llama3.2-vision, qwen2.5vl, gemma3), up to 60 visuals per build also get a factual **description**. Standalone images come first, then figures, then frames.
4. **Chunk + Embed + Index** ([backend/chunker.py](backend/chunker.py), [backend/store.py](backend/store.py))
   - Heading-aware chunks (~900 chars, 150 overlap) never cross a page/slide boundary. Each one keeps its location breadcrumb, which is embedded with the text.
   - Two FAISS indexes per collection: **text** (MiniLM, 384-d), holding passages, transcripts and image descriptions/OCR; and **visual** (CLIP, 512-d), holding images, figures and frames.

**Duplicate files** (same SHA-256) are skipped, including when you add files to an existing collection later.

## How retrieval works across modalities

Cosine scores from different embedding spaces aren't comparable. CLIP gives any image containing text
about 0.25 against almost any query, while MiniLM scores for related passages land around 0.4–0.8. So each list
is **calibrated** onto a shared 0–1 relevance scale before fusing:

- **Text → text** (MiniLM): passages, transcripts, image OCR and captions
- **Text → image** (CLIP): "the slide with the bar chart" finds the figure even when no text mentions it
- **Image → image** (CLIP): attach or paste a picture to find where it (or something like it) appears

An item matched by several routes gets a bonus. Keyword overlap, modality hints ("in the video",
"the chart") and file names in the question ("in report.pdf") nudge the ranking. Weak matches are
dropped instead of padding the context, and results are capped per file for variety. Broad questions
("summarize this collection") start from the opening of each file.

**Generation.** Each source is described to the LLM by modality: passage, transcript with timestamp,
or image with its description and OCR. If the selected model can see, the retrieved images and your
attached image are sent as pixels too. If you attach an image while using a text-only model,
OmniRAG asks any installed vision model to describe it first and adds that description to the search.

## Using the app

| Area | What it does |
|---|---|
| **Drop zone** | Drag files anywhere on the page. Queue, remove and name the collection, then build. |
| **Processing settings** | Whisper size (tiny/base/small), video frame interval, OCR on/off, vision-model descriptions on/off. |
| **Live pipeline** | Upload → Parse → Transcribe → Vision → Embed → Index, with per-file activity and counters. |
| **Chat** | Streaming answers with `[n]` citations. Source cards show page thumbnails, images and frames. Click to expand, view full size, or **play audio/video from the cited timestamp**. |
| **Ask with an image** | Click the image button, paste a screenshot, or drop an image on the composer. |
| **Add files** | Grow an existing collection at any time. Only the new files are processed. |
| **Files** | Every file with pages, duration, word and frame counts. Click to open the original. |
| **Model picker** | Any chat model in Ollama. Models that can see images are labeled. |

## Project structure

```
OmniRAG/
├── run.sh                 # one-command setup + launch (port 8001)
├── requirements.txt
├── backend/
│   ├── app.py             # FastAPI: uploads, SSE progress, chat, file serving, UI
│   ├── config.py          # settings (env-overridable) + supported formats
│   ├── parsers.py         # PDF / DOCX / PPTX / text parsing, header/footer cleanup, figures
│   ├── media.py           # faster-whisper transcription, PyAV frame sampling
│   ├── vision.py          # CLIP embeddings, OCR, Ollama vision captions
│   ├── chunker.py         # heading-aware + transcript-window chunking
│   ├── store.py           # two FAISS indexes, calibrated multimodal fusion
│   ├── llm.py             # vision-aware prompting + streaming, offline fallback
│   └── ingest.py          # pipeline orchestration + progress events
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js             # vanilla JS, no build step
└── data/                  # created at runtime: collections (files, thumbnails, indexes)
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8001` | Web server port |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `OMNIRAG_LLM_MODEL` | auto | Force a chat model |
| `OMNIRAG_VISION_MODEL` | auto | Force the vision model used for captions |
| `OMNIRAG_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Text embeddings (rebuild collections after changing) |
| `OMNIRAG_CLIP_MODEL` | `clip-ViT-B-32` | Image/text embeddings (rebuild collections after changing) |
| `OMNIRAG_WHISPER_MODEL` | `base` | Default speech-to-text size |
| `OMNIRAG_CHUNK_SIZE` / `OMNIRAG_CHUNK_OVERLAP` | `900` / `150` | Chunking, in characters |
| `OMNIRAG_MAX_CAPTIONS` | `60` | Vision-model descriptions per build |
| `OMNIRAG_MAX_FRAMES` | `120` | Sampled frames per video |
| `OMNIRAG_MAX_UPLOAD_MB` | `2048` | Per-file upload limit |
| `OMNIRAG_DATA_DIR` | `./data` | Where collections are stored |

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Ollama status, chat and vision models, embedding models, OCR availability |
| `POST` | `/api/ingest` | multipart `files[]`, optional `kb_id` (add to existing), `name`, `options` JSON → `{job_id, kb_id}` |
| `GET` | `/api/ingest/{job_id}/events` | SSE: `stage`, `file`, `log`, `stats`, `complete`, `error` |
| `GET` | `/api/kbs` · `/api/kbs/{id}` | List collections / details incl. files |
| `DELETE` | `/api/kbs/{id}` | Delete a collection and its files |
| `GET` | `/api/kbs/{id}/raw/{files\|assets}/…` | Original files and thumbnails/frames (range requests for media seeking) |
| `POST` | `/api/kbs/{id}/chat` | `{question, history, model, image?}` → SSE: `sources`, `meta`, `token`…, `done` |

Interactive docs: http://localhost:8001/docs

## Troubleshooting

- **"Vision · none".** That's fine: images are still searchable by what they look like (CLIP) and by text (OCR). Run `ollama pull moondream` to add descriptions and let answers see images.
- **"OCR · not installed".** Install Tesseract (`brew install tesseract`) and restart.
- **Long transcription.** Pick **Tiny** in Processing settings, or set `OMNIRAG_WHISPER_MODEL=tiny`. **Small** is more accurate but slower.
- **First build is slow.** The models download once. Later builds start immediately.
- **Video won't seek.** Some MKV/AVI files don't play natively in browsers. MP4 (H.264) and WEBM play everywhere.

## Connect with me

[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/techwith_ram)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ramakm)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ramakrushnamohapatra/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwith.ram)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@techwithram)
