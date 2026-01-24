import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import streamlit as st

st.set_page_config(page_title="YV Search (Demo)", page_icon="🔎")
import pdfplumber
import re
from pathlib import Path
import pytesseract

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)  # solid + cheap-ish


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a technical support assistant.
Answer the question using ONLY the documentation context.
If the answer cannot be found in the context, respond with exactly:
"I can't find this information in the provided documentation."

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:
""".strip()

    return llm.invoke(prompt).content.strip()


# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------------
# Config (must be first Streamlit call)
# -------------------------------

# PROJECT_CODENAME = "YV"

# -------------------------------
# UI Style
# -------------------------------
st.markdown(
    """
<style>
/*Margin For Answer Retrieved*/
.bot-header {
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
/* Preserve newlines in answer text */
.chat-bubble pre,
.chat-bubble .answer-text {
    white-space: pre-wrap;   /* keeps line breaks and wraps nicely */
    word-wrap: break-word;
}
/* Emphasized helper text */
.confidence-note {
    color: rgba(0, 0, 139, 0.95);   /* YV deep blue */
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.35rem;
}

/* Page width & spacing */
.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Primary buttons – premium touch */
.stButton > button {
    border-radius: 14px;
    padding: 0.6rem 1.1rem;
    font-weight: 600;
    border: 1px solid rgba(49, 51, 63, 0.20);
}

.stButton > button:hover {
    border: 1px solid rgba(0, 0, 139, 0.45); /* subtle YV blue */
}

/* Chat bubbles (shared) */
.chat-bubble {
    padding: 0.9rem 1.1rem;
    border-radius: 18px;
    margin: 0.6rem 0;
    border: 1px solid rgba(49, 51, 63, 0.18);
    line-height: 1.45;
}

/* User messages */
.chat-bubble.user {
    background: rgba(0, 122, 255, 0.10);
}

/* Bot messages */
.chat-bubble.bot {
    background: rgba(46, 204, 113, 0.10);
}
</style>

""",
    unsafe_allow_html=True,
)

st.title("🔎 YV Search (Demo)")
st.caption("Hallucination-resistant RAG assistant over example technical documentation")
# st.caption(f"{PROJECT_CODENAME} • YV-hard work defines it")

# -------------------------------
# Retrieval guardrails (define BEFORE use)
# -------------------------------
NOT_FOUND_MESSAGE = (
    "This information isn’t found in the available documentation.<br>"
    "Please check your core tenant configuration and source material.<br>"
    "If needed, escalate to the appropriate team members."
)

st.sidebar.header("Settings")
k = st.sidebar.slider("Top-K retrieved chunks", 1, 4, 2)
min_hit_count = st.sidebar.slider("Min matching chunks required", 1, 4, 1)
# For LangChain FAISS, score is often L2 distance (lower = better). Tune as needed.

show_context = st.sidebar.checkbox("Show retrieved context", value=False)
use_llm = st.sidebar.checkbox(
    "Answer generation is optional; sources remain available.", value=False
)
st.sidebar.caption("Default strictness tuned for documentation accuracy (1.25)")
DEFAULT_MAX_DISTANCE = 1.25
max_distance = st.sidebar.slider(
    "Max distance allowed (lower = stricter)", 0.2, 2.0, DEFAULT_MAX_DISTANCE, 0.05
)
st.sidebar.markdown(
    """
    <div class="confidence-note">
        This assistant only responds when documentation relevance meets confidence thresholds.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# Keyword overlap guardrail
# -------------------------------
def keyword_overlap_ok(question: str, kept_texts: list[str]) -> bool:
    # Extract simple keywords (>=3 chars). Keeps acronyms like SSO.
    q_terms = set(re.findall(r"[A-Za-z]{3,}", question.lower()))
    if not q_terms:
        return True

    ctx_terms = set(re.findall(r"[A-Za-z]{3,}", " ".join(kept_texts).lower()))

    # Require at least one meaningful shared term
    return len(q_terms.intersection(ctx_terms)) >= 1


# -------------------------------
# Scope coverage guardrail (NEW)
# -------------------------------
def scope_coverage_ok(question: str, kept_texts: list[str]) -> bool:
    q = question.lower()
    ctx = " ".join(kept_texts).lower()

    # If the user asks tenant-scoped questions, require tenant language in the retrieved text.
    tenant_scoped = bool(
        re.search(r"\btenant\b|\bper[-\s]?tenant\b|\bspecific tenant\b", q)
    )
    if tenant_scoped:
        return bool(
            re.search(r"\btenant\b|\bper[-\s]?tenant\b|\btenant[-\s]?specific\b", ctx)
        )

    # You can add more scopes later (version/env/role) the same way.
    return True


def multi_intent_coverage_ok(question: str, kept_texts: list[str]) -> bool:
    q = question.lower()
    ctx = " ".join(kept_texts).lower()

    asks_sync = bool(re.search(r"\bsync\b|\bsynchroniz", q))
    asks_perms = bool(re.search(r"\bpermission\b|\bpermissions\b|\bgroup\b", q))

    if asks_sync and asks_perms:
        has_sync = bool(re.search(r"\bsync\b|\bsynchroniz", ctx))
        has_perms = bool(re.search(r"\bpermission\b|\bpermissions\b|\bgroup\b", ctx))
        return has_sync and has_perms

    return True


# -------------------------------
# PDF Extraction (Text + OCR)
# -------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

            # OCR only if page is basically empty AND tesseract is available
            if len(page_text.strip()) < 20:
                try:
                    if Path(pytesseract.pytesseract.tesseract_cmd).exists():
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


def generate_answer(question: str, context: str) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # good cost/quality for demos
        temperature=0.1,
    )

    system = (
        "You are a documentation assistant. "
        "Answer ONLY using the provided documentation context. "
        "If the answer is not clearly supported by the context, say you don't have enough information."
    )

    user = f"""Question:
{question}

Documentation context:
{context}

Return a concise answer (3-6 sentences). If steps exist, use bullets."""
    resp = llm.invoke([("system", system), ("user", user)])
    return resp.content.strip()


if st.button("Ask YV Search") and question:
    if st.session_state.vectorstore is None:
        st.error("Knowledge base not loaded.")
        st.stop()

    docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(
        question, k=k
    )

    if not docs_with_scores:
        st.markdown(
            f'<div class="chat-bubble bot"><b>Bot:</b><br>{NOT_FOUND_MESSAGE}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    kept = []
    for d, score in docs_with_scores:
        text = d.page_content.replace("=== DOCUMENT BREAK ===", "").strip()
        if text and score <= max_distance:
            kept.append((text, score))

    st.markdown(
        f'<div class="chat-bubble user"><b>You Asked:</b><br>{question}</div>',
        unsafe_allow_html=True,
    )

    kept_texts = [t for t, _ in kept]
    passes_keywords = keyword_overlap_ok(question, kept_texts)
    passes_scope = scope_coverage_ok(question, kept_texts)
    passes_multi = multi_intent_coverage_ok(question, kept_texts)

    if (
        len(kept) < min_hit_count
        or not passes_keywords
        or not passes_scope
        or not passes_multi
    ):
        if not passes_scope:
            msg = (
                "I found documentation related to your topic, but it doesn’t cover the "
                "**tenant-specific** part of your question.<br>"
                "Please check tenant configuration or provide tenant-scoped documentation."
            )
        elif not passes_multi:
            msg = (
                "I found documentation related to part of your question, but it does not cover "
                "the full combination (e.g., **sync + permissions**).<br>"
                "Please provide documentation that links these topics, or rephrase to one topic."
            )
        else:
            msg = NOT_FOUND_MESSAGE

        st.markdown(
            f'<div class="chat-bubble bot"><b>Bot:</b><br>{msg}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Build context once
    context = "\n\n".join(t for t, _ in kept)

    # Optional LLM generation with safe fallback
    answer = None
    if use_llm:
        try:
            answer = generate_answer(question, context)
        except Exception:
            st.sidebar.warning(
                "Answer generation temporarily unavailable — showing sources only."
            )
            answer = None

    # Render answer OR retrieved context
    if answer and answer.strip():
        st.markdown(
            f'<div class="chat-bubble bot"><div class="answer-text">{answer}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="chat-bubble bot bot-header">Retrieved documentation context:</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-bubble bot"><div class="answer-text">{context}</div></div>',
            unsafe_allow_html=True,
        )

    # Optional sources
    if show_context:
        with st.expander("Sources (retrieved context)"):
            for t, s in kept:
                st.write(f"Score: {s:.4f}")
                st.code(t)
                st.divider()
