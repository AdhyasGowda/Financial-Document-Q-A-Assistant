import streamlit as st
import pdfplumber
import pandas as pd
import io
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
from typing import List, Dict, Tuple

# Configuration

OLLAMA_URL = "http://127.0.0.1:11434" 
MODEL_NAME = "llama3:latest"            
MAX_CONTEXT_CHARS = 1500 
TOP_K = 2  

# Helpers: extraction

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, List[pd.DataFrame]]:
    """Return extracted text and list of tables (as DataFrames) from a PDF file bytes.
    Uses pdfplumber to extract text and tables where possible.
    """
    text_parts = []
    tables = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                text_parts.append(page_text)

                try:
                    page_tables = page.extract_tables()
                    for t in page_tables:
                        try:
                            df = pd.DataFrame(t[1:], columns=t[0]) if len(t) > 1 else pd.DataFrame(t)
                            tables.append(df)
                        except Exception:
                            pass
                except Exception:
                    pass
    except Exception as e:
        st.error(f"PDF parsing error: {e}")

    full_text = "\n\n".join([p for p in text_parts if p])
    return full_text, tables


def extract_text_from_excel(file_bytes: bytes) -> Tuple[str, List[pd.DataFrame]]:
    """Read all sheets and tables from an Excel file and return a combined text and list of DataFrames.
    """
    tables = []
    text_parts = []
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
                if not df.empty:
                    tables.append(df)
                    text_parts.append(f"Sheet: {sheet_name}\n" + df.to_string(index=False))
            except Exception as e:
                text_parts.append(f"(couldn't read sheet {sheet_name}: {e})")
    except Exception as e:
        st.error(f"Excel parsing error: {e}")

    full_text = "\n\n".join(text_parts)
    return full_text, tables

# Helpers: chunking + retriever

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk long text into overlapping chunks of roughly max_chars characters.
    """
    if not text:
        return []
    text = text.replace('\r', '')
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunks.append(text[start:end])
        if end == L:
            break
        start = end - overlap
    return chunks


class SimpleRetriever:
    """TF-IDF based retriever over document chunks.
    """
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.matrix = None

    def build(self, texts: List[str]):
        self.chunks = texts
        if not texts:
            self.vectorizer = None
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        if self.matrix is None or self.vectorizer is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idxs if sims[i] > 0]

# Helpers: basic numeric extraction

def find_monetary_values(text: str) -> List[Tuple[str, float]]:
    """Find currency-like numbers and return them as (match_text, numeric_value).
    Very heuristic: looks for numbers with commas, decimals, optional currency symbols.
    """
    pattern = r"(?P<sym>[$₹€£]?)\s*(?P<num>[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(?P<suf>mn|m|million|bn|b|crore)?"
    matches = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        num = m.group('num').replace(',', '')
        try:
            v = float(num)
            suf = (m.group('suf') or '').lower()
            if suf in ('m', 'mn', 'million'):
                v = v * 1e6
            elif suf in ('bn', 'b'):
                v = v * 1e9
            elif suf == 'crore':
                v = v * 1e7
            matches.append((m.group(0), v))
        except Exception:
            continue
    return matches

# Helpers: Ollama client

def generate_with_ollama(prompt: str, model: str = MODEL_NAME, timeout: int = 120) -> str:
    url = OLLAMA_URL.rstrip('/') + '/v1/chat/completions'
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful financial document assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.0
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)  # <-- increased
        resp.raise_for_status()
        data = resp.json()
        if 'choices' in data and isinstance(data['choices'], list):
            return ''.join([c.get('message', {}).get('content','') for c in data['choices']])
        return resp.text
    except Exception as e:
        return f"[Model call failed: {e}]"

# App UI and orchestration

st.set_page_config(page_title="Financial Document QA", layout="wide")
st.title("📊 Financial Document Q&A — Local (Streamlit + Ollama)")

st.markdown(
    """
    Upload PDF or Excel financial statements (income statement, balance sheet, cash flow). The app extracts text and tables, builds a lightweight retriever, and uses a local SLM (via Ollama) to answer your questions about the uploaded documents.
    """
)

st.sidebar.header("Upload documents")
uploaded_file = st.file_uploader("Upload a financial document", type=["pdf", "xlsx", "xls"])

st.sidebar.header("Model / Settings")
ollama_url_input = st.sidebar.text_input("Ollama URL", OLLAMA_URL)
model_name_input = st.sidebar.text_input("Model name", MODEL_NAME)
if ollama_url_input:
    OLLAMA_URL = ollama_url_input
if model_name_input:
    MODEL_NAME = model_name_input

if 'docs' not in st.session_state:
    st.session_state['docs'] = []  
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = SimpleRetriever()
if 'chat' not in st.session_state:
    st.session_state['chat'] = []  

# Process uploads

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    text, tables = "", []

    if uploaded_file.type == "application/pdf":
        text, tables = extract_text_from_pdf(file_bytes)
        st.write("📑 Extracted text preview:", text[:500])

    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel"
    ]:
        text, tables = extract_text_from_excel(file_bytes)
        st.write("📊 Extracted Excel preview:", text[:500])

    # 🔹 Chunk text and add to session state
    chunks = chunk_text(text, max_chars=1000, overlap=200)
    st.session_state['docs'].append({
        'name': uploaded_file.name,
        'text': text,
        'tables': tables,
        'chunks': chunks
    })

    # 🔹 Rebuild retriever with all chunks
    all_chunks = []
    for d in st.session_state['docs']:
        all_chunks.extend(d['chunks'])
    st.session_state['retriever'].build(all_chunks)

    st.success(f"✅ Added {uploaded_file.name} with {len(chunks)} chunks and {len(tables)} tables.")

user_query = st.text_input("Ask a question about your documents")
submit = st.button("Ask")

if submit and user_query:
    # retrieve top-K context
    retriever: SimpleRetriever = st.session_state['retriever']
    retrieved = retriever.retrieve(user_query, top_k=TOP_K)
    ctx_texts = []
    if retrieved:
        for idx, score in retrieved:
            chunk = retriever.chunks[idx]
            ctx_texts.append(f"(score={score:.3f})\n{chunk}")
    else:
        st.warning("No relevant context found in documents. The model will answer only from its general knowledge.")

    context_for_prompt = '\n\n'.join(ctx_texts)[:MAX_CONTEXT_CHARS]

    # Build prompt for the model. We provide clear instructions to behave as a document QA assistant.
    prompt = f"You are a helpful financial document assistant. Use only the information from the CONTEXT when answering. If the answer is not present, say 'I could not find that in the uploaded documents.'\n\nCONTEXT:\n{context_for_prompt}\n\nQUESTION: {user_query}\n\nAnswer concisely and show any numbers you used and the source (document name) if possible."

    with st.spinner("Generating answer from local model..."):
        response = generate_with_ollama(prompt, model=MODEL_NAME)

    # Save to chat
    st.session_state['chat'].append((user_query, response))

# Render chat history
if st.session_state['chat']:
    st.write("### Conversation")
    for i, (u, a) in enumerate(reversed(st.session_state['chat'])):
        st.markdown(f"**You:** {u}")
        st.markdown(f"**Assistant:** {a}")
        st.write('---')

# Footer: tips
st.markdown(
    """
    **Tips:**
    - For best results, upload well-formatted financial statements (tables in Excel or clearly structured PDFs).
    - If the model fails to answer correctly, try pressing 'Show extracted monetary values' to verify the parsed numbers.
    - Configure `OLLAMA_URL` and `MODEL_NAME` in the sidebar to point to your local SLM.

    """
)

# Error handling / cleanup button
if st.button("Clear documents & chat"):
    st.session_state['docs'] = []
    st.session_state['retriever'] = SimpleRetriever()
    st.session_state['chat'] = []
    st.success("Cleared.")
