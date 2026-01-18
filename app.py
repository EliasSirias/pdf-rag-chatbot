import os
import streamlit as st
import pdfplumber
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#PROJECT_CODENAME = "YV"
# -------------------------------
# Config + UI Style
# -------------------------------
st.set_page_config(page_title="Fix-It RAG Bot", page_icon="🛠️")

st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 950px; }
.stButton>button { border-radius: 14px; padding: 0.6rem 1rem; font-weight: 600; }
.chat-bubble { padding: 0.9rem 1rem; border-radius: 16px; margin: 0.4rem 0;
  border: 1px solid rgba(49,51,63,0.2); }
.user { background: rgba(0, 122, 255, 0.08); }
.bot  { background: rgba(46, 204, 113, 0.08); }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Fix-It RAG Bot")
st.caption("Ask questions about **Mobile Sync** and **Group Permissions** documentation.")

st.sidebar.header("Settings")
k = st.sidebar.slider("Top-K retrieved chunks", 1, 4, 1)
show_context = st.sidebar.checkbox("Show retrieved context", value=False)

# -------------------------------
# PDF Extraction (Text + OCR)
# -------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
            if len(page_text.strip()) < 20:
                try:
                    image = page.to_image(resolution=300).original
                    text += pytesseract.image_to_string(image) + "\n"
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

    vs = FAISS.from_texts(chunks, embeddings)
    return vs, chunks

# -------------------------------
# Load PDFs at startup
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
            st.success(f"Knowledge base loaded ✅ ({len(chunks)} chunks)")

# -------------------------------
# Ask Question
# -------------------------------
question = st.text_input("Question", placeholder="e.g., How do I sync data on mobile?")

if st.button("Get Answer") and question:
    if st.session_state.vectorstore is None:
        st.error("Knowledge base not loaded.")
    else:
        docs = st.session_state.vectorstore.similarity_search(question, k=k)

        context = "\n\n".join(
            d.page_content.replace("=== DOCUMENT BREAK ===", "").strip()
            for d in docs
        )

        st.markdown(f'<div class="chat-bubble user"><b>You:</b><br>{question}</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-bubble bot"><b>Bot:</b><br>Answer retrieved from documentation:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble bot">{context}</div>', unsafe_allow_html=True)

        if show_context:
            with st.expander("Retrieved context"):
                st.text(context)
