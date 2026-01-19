import streamlit as st
import pdfplumber
import pytesseract
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# Config (must be first Streamlit call)
# -------------------------------
st.set_page_config(page_title="Fix-It RAG Bot", page_icon="🛠️")

#PROJECT_CODENAME = "YV"

# -------------------------------
# UI Style
# -------------------------------
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
st.caption("RAG chatbot over example technical documentation PDFs (synchronization and permissions examples)")
#st.caption(f"{PROJECT_CODENAME} • hard work defines it")

# -------------------------------
# Retrieval guardrails (define BEFORE use)
# -------------------------------
NOT_FOUND_MESSAGE = (
    "This information isn’t found in the available documentation.<br>"
    "Please check your core tenant configuration and source material.<br>"
    "If needed, escalate to the appropriate team members."
)

st.sidebar.header("Settings")
k = st.sidebar.slider("Top-K retrieved chunks", 1, 4, 1)
min_hit_count = st.sidebar.slider("Min matching chunks required", 1, 4, 1)
# For LangChain FAISS, score is often L2 distance (lower = better). Tune as needed.

show_context = st.sidebar.checkbox("Show retrieved context", value=False)
st.sidebar.caption("Default strictness tuned for documentation accuracy (1.25)")
DEFAULT_MAX_DISTANCE = 1.25
max_distance = st.sidebar.slider(
    "Max distance allowed (lower = stricter)",
    0.2,
    2.0,
    DEFAULT_MAX_DISTANCE,
    0.05
)

# -------------------------------
# Keyword overlap guardrail
# -------------------------------
def keyword_overlap_ok(question: str, kept_texts: list[str]) -> bool:
    # Extract simple keywords (>=3 chars). Keeps acronyms like SSO.
    q_terms = set(re.findall(r"[A-Za-z]{3,}", question.lower()))
    if not q_terms:
        return True

    ctx_terms = set(
        re.findall(r"[A-Za-z]{3,}", " ".join(kept_texts).lower())
    )

    # Require at least one meaningful shared term
    return len(q_terms.intersection(ctx_terms)) >= 1

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
BASE_DIR = Path(__file__).resolve().parent

PDF_PATHS = [
    BASE_DIR / "docs" / "mobile_sync_troubleshooting_generic_v1.pdf",
    BASE_DIR / "docs" / "manage_group_permissions_generic_v1.pdf",
]


st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("chunks", None)

if st.session_state.vectorstore is None:
    with st.spinner("Loading PDF knowledge base..."):
        all_text = []

        for path in PDF_PATHS:
            if not path.exists():
                st.error(f"Missing file: {path}")
                st.stop()

            all_text.append(extract_text_from_pdf(str(path)))
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
        docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(question, k=k)
        SAFE_MESSAGE = (
            "This information isn’t found in the available documentation.\n"
            "Please refer to the source material or your tenant configuration."
        )

        # Use the vectorstore you already have in session_state
        results = st.session_state.vectorstore.similarity_search_with_score(question, k=4)

        if not results:
            st.markdown(
                f'<div class="chat-bubble bot"><b>Bot:</b><br>{SAFE_MESSAGE}</div>',
                unsafe_allow_html=True
            )
            st.stop()

        best_doc, best_score = results[0]
        print("DEBUG best_score:", best_score)

        # TEMP: "weak" threshold — we'll tune after you see the scores
        if best_score > 0.6:
            st.markdown(
                f'<div class="chat-bubble bot"><b>Bot:</b><br>{SAFE_MESSAGE}</div>',
                unsafe_allow_html=True
            )
            st.stop()


        kept = []
        for d, score in docs_with_scores:
            text = d.page_content.replace("=== DOCUMENT BREAK ===", "").strip()
            if text and score <= max_distance:
                kept.append((text, score))

        st.markdown(
            f'<div class="chat-bubble user"><b>You:</b><br>{question}</div>',
            unsafe_allow_html=True
        )

        kept_texts = [t for t, _ in kept]
        passes_keywords = keyword_overlap_ok(question, kept_texts)

        if len(kept) < min_hit_count or not passes_keywords:
            st.markdown(
                f'<div class="chat-bubble bot"><b>Bot:</b><br>{NOT_FOUND_MESSAGE}</div>',
                unsafe_allow_html=True
            )
        else:
            context = "\n\n".join(t for t, _ in kept)

            st.markdown(
                '<div class="chat-bubble bot"><b>Bot:</b><br>Answer retrieved from documentation:</div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="chat-bubble bot">{context}</div>', unsafe_allow_html=True)

            if show_context:
                with st.expander("Retrieved context (with scores)"):
                    for t, s in kept:
                        st.write(f"Score: {s:.4f}")
                        st.text(t)
                        st.divider()
