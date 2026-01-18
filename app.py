import os
import tempfile
import streamlit as st

import pdfplumber
import pytesseract
from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from openai import OpenAI


# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Fix-It RAG Bot", page_icon="🛠️")
st.title("🛠️ Troubleshooting Assistant (PDF RAG)")



SYSTEM_PROMPT = """You are a technical support assistant.
Answer ONLY using the provided context from the PDF.
If the context is insufficient, say you do not have enough information.

Format your answer as:
1) Summary
2) Most likely cause
3) Step-by-step fix
4) Verification
5) Escalate if...
6) Sources (use [chunk#])
"""


# -------------------------------
# PDF Extraction (Text + OCR)
# -------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            text += page_text + "\n"

            # OCR fallback if page has little text
            if len(page_text.strip()) < 20:
                try:
                    image = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text + "\n"
                except Exception:
                    pass

    return text.strip()


# -------------------------------
# Build Vector Store
# -------------------------------
def build_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vs = FAISS.from_texts(
        chunks,
        embeddings,
        metadatas=[{"chunk": i} for i in range(len(chunks))]
    )

    return vs, chunks


# -------------------------------
# UI — Upload PDF
# -------------------------------
PDF_PATHS = [
    "docs/mobile_sync_troubleshooting.pdf",
    "docs/Manage Group Permissions.pdf",
]

st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("chunks", None)

if st.session_state.vectorstore is None:
    with st.spinner("Loading PDF knowledge base..."):
        all_text = []
        for path in PDF_PATHS:
            all_text.append(extract_text_from_pdf(path))
            all_text.append("\n\n=== DOCUMENT BREAK ===\n\n")

        extracted_text = "".join(all_text)

        if len(extracted_text) < 200:
            st.error("Very little text detected in your PDFs.")
        else:
            vs, chunks = build_vectorstore(extracted_text)
            st.session_state.vectorstore = vs
            st.session_state.chunks = chunks
            st.success(f"PDFs loaded ✅ ({len(chunks)} chunks)")


# -------------------------------
# Ticket Details
# -------------------------------
#st.subheader("Ticket Details (optional but helpful)")
#product = st.text_input("Product / Area")
#error_msg = st.text_area("Error message")
#symptoms = st.text_area("Symptoms")
#environment = st.text_input("Environment")


# -------------------------------
# Ask Question
# -------------------------------
question = st.text_input("Question", placeholder="How do I fix X?")

if st.button("Get Answer") and question:
    if st.session_state.vectorstore is None:
        st.error("Upload a PDF first.")
    else:
        query = question

        docs = st.session_state.vectorstore.similarity_search(query, k=1 )

        context = "\n\n".join(
            d.page_content.replace("=== DOCUMENT BREAK ===", "").strip()
            for d in docs
        )


        if not os.getenv("OPENAI_API_KEY"):
            st.warning("No API key set. Showing retrieved context only.")
            st.text(context)




