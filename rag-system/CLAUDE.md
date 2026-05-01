# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end RAG (Retrieval-Augmented Generation) pipeline for processing real PDF documents. The system ingests PDFs, chunks text, generates vector embeddings, indexes them with FAISS, retrieves relevant context, and (when complete) generates LLM responses.

## Setup & Running

No formal build system exists. Install dependencies manually:

```bash
pip install PyPDF2 sentence-transformers faiss-cpu numpy
```

Run the main application (currently a stub):

```bash
python app.py
```

There is no test suite yet.

## Architecture

The pipeline flows linearly through five modules:

1. **`ingestion/`** — PDF text extraction and chunking
   - `pdf_loader.py`: `load_pdf(file_path)` uses PyPDF2 to extract raw text
   - `chunking.py`: `chunk_text(text, chunk_size=500, overlap=100)` splits text into overlapping windows

2. **`embeddings/`** — Text vectorization
   - `embedder.py`: Loads `SentenceTransformer('all-MiniLM-L6-v2')` at module import; `get_embeddings(chunks)` converts text chunks to numpy vectors

3. **`retrieval/`** — Semantic search over indexed embeddings
   - `faiss_index.py`: `create_faiss_index(embeddings)` builds a FAISS `IndexFlatL2` index
   - `retrieval.py`: `retrieve(query, model, index, chunks, k=3)` returns the top-k most similar chunks

4. **`generation/llm_pipeline.py`** — LLM response generation (**not yet implemented**)

5. **`evaluation/metrics.py`** — RAG quality metrics (**not yet implemented**)

6. **`app.py`** — Orchestrates the full pipeline (**not yet implemented**)

## Key Implementation Notes

- The SentenceTransformer model is a module-level global in `embedder.py` — it loads on import, which is fine for a single-process pipeline but needs care if multiple modules import it.
- FAISS index uses L2 distance (`IndexFlatL2`). Embeddings from `all-MiniLM-L6-v2` are unit-normalized, so L2 and cosine similarity are equivalent here.
- PDF documents should be placed in `data/`.
- `chunk_size` and `overlap` in `chunking.py` are the primary tuning knobs for retrieval quality.
