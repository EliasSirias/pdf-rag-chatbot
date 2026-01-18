##Powered By YVSearch
Hard Work Defines it.

# PDF RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of PDF documents.

## Features
- PDF text extraction with OCR fallback
- Semantic search using FAISS vector store
- Optional LLM-based answer generation (API key not included)
- Streamlit-based user interface

## Knowledge Base
The chatbot demonstrates retrieval-augmented question answering over a small set of example PDF documents.
The documents represent general technical documentation topics such as system synchronization and access control.

For best results, questions should align with the content of the provided documentation.

## Setup Instructions
```bash
pip install -r requirements.txt
streamlit run app.py
## Repository Notes
This repository intentionally excludes:
- PDF documents (customer-facing documentation)
- API keys or environment files
- Virtual environment files (`.venv`)

PDF files are ignored via `.gitignore` and must be supplied locally to run the application.
This ensures sensitive documents and credentials are not committed to version control.

## Usage Notes
Questions should be aligned with the topics covered in the loaded documentation for best results.

