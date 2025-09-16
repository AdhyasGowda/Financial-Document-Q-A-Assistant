# Financial-Document-Q-A-Assistant
A local web application for analyzing financial documents (PDFs &amp; Excel) and answering user questions using natural language. Built with Streamlit and a local Ollama-hosted Small Language Model (SLM).

---

## Features
1. Upload PDF or Excel financial statements (Income Statement, Balance Sheet, Cash Flow)
2. Extract text and tables from uploaded documents
3. Context-aware question-answering using local Ollama models
4. Conversational chat interface with follow-up questions
5. Shows numbers used in answers and references the source document
6. Clear feedback and error handling

---

## Requirements
1. Python 3.10+ recommended.
2. Install dependencies:
bash
pip install streamlit pdfplumber pandas openpyxl scikit-learn numpy requests

---

## Setup
1. Run a local Ollama model:
Make sure your model is hosted and listening at http://127.0.0.1:11434. Update OLLAMA_URL and MODEL_NAME in the app if needed.
2. Start the Streamlit app:
bash
streanlit run financialDocument.py
3. Open your browser at http://localhost:8501

---

## Usage
1. Upload PDF or Excel financial documents via the sidebar.
2. The app will extract text and tables, chunk them, and build a retriever.
3. Enter your question in the input box, e.g., “What is the total revenue for 2023?”
4. Click Ask — the app will generate an answer using the local Ollama model.
5. Chat history is maintained in the session. Use Clear documents & chat to reset.

---

## Configuration
1. OLLAMA_URL: URL of the local Ollama model API
2. MODEL_NAME: Name of the local model, e.g., llama3:latest
3. MAX_CONTEXT_CHARS: Maximum number of characters from retrieved chunks for context
4. TOP_K: Number of top chunks to retrieve for QA

---

## Notes
1.Supported file types: PDF, XLSX, XLS
2.For best results, upload well-formatted financial statements.
3.If answers are incomplete, check the extracted text preview to ensure the data is readable.
