# YV Search Demo  
**Powered by YV Search — Hard work defines it.**

## Overview
`yv-search-demo` is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of PDF documents.  
It demonstrates document ingestion, semantic search, and answer generation in a simple, local setup.

## Features
- PDF text extraction with OCR fallback  
- Semantic search using FAISS vector store  
- Optional LLM-based answer generation (API key not included)  
- Streamlit-based user interface  

## Knowledge Base
The chatbot operates over a small set of example PDF documents representing general technical documentation topics such as:
- system synchronization  
- access control  
- configuration and troubleshooting guidance  

For best results, questions should align with the content of the loaded documents.

## Quick Start

### Prerequisites
- Python 3.x  
- `pip`  

### Install dependencies
-```bash
pip install -r requirements.txt

### Run the APP
streamlit run app.py

### Example Questions

-"How do I manually sync data on a mobile device?"

-"What permissions are required to manage user groups?"

-"Why might a sync operation fail?"
(Results depend on the content of the provided PDFs.)
### Repository Notes
This repository intentionally excludes:

-Example PDF documents

-API keys or environment files

-Virtual environment files (.venv)

PDF files are ignored via .gitignore and must be supplied locally to run the application.
This ensures sensitive or proprietary documents and credentials are not committed to version control.

### Usage Notes

This project is intended as a demo of retrieval-augmented question answering.
It focuses on correctness, clarity, and safe handling of documents rather than production deployment.

