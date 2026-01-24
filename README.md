# YV Search (Demo)

YV Search is a hallucination-resistant Retrieval-Augmented Generation (RAG) assistant built to answer questions strictly from provided documentation.

The project emphasizes **precision, transparency, and refusal over guesswork**, making it suitable for internal knowledge bases and technical documentation workflows.

---

## 🔍 What It Does

- Ingests PDF documentation
- Chunks and embeds content using sentence transformers
- Retrieves relevant context via FAISS similarity search
- Applies multiple guardrails before answering
- Optionally generates a concise answer using an LLM
- Refuses to answer when confidence thresholds are not met

---

## 🧠 Key Features

### Retrieval Guardrails
- **Distance threshold filtering**
- **Keyword overlap validation**
- **Scope coverage checks** (e.g., tenant-specific questions)
- **Multi-intent validation** (e.g., sync + permissions)

### Optional Answer Generation
- Concise answer generation can be toggled on/off
- Answers are generated **only** when retrieval confidence passes
- Automatic fallback to source documents if generation is unavailable

### Transparency First
- Retrieved context can always be inspected
- No hidden reasoning or fabricated answers
- Refusal messaging is explicit and intentional

---

## 🧱 Architecture Overview

1. PDF text extraction (with OCR fallback)
2. Recursive text chunking
3. Vector embeddings using `all-MiniLM-L6-v2`
4. FAISS similarity search
5. Guardrail evaluation
6. Optional LLM answer synthesis

---

## 🖥️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- OpenAI (optional, for answer generation)

---

## ▶️ Running the App

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py


