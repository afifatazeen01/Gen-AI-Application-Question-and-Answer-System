# 🤖 GenAI Question Answering System

**Developed by: Afifa Tazeen**
**Tech Stack: Python | NLP | RAG | Streamlit**

---

## 📌 Project Overview

A Generative AI–based Question Answering system built using the
**RAG (Retrieval-Augmented Generation)** pipeline architecture.
The system retrieves relevant context from a knowledge base and
generates accurate, grounded answers to user questions.

---

## 🏗️ Architecture — RAG Pipeline

```
Documents → Chunking → TF-IDF Embedding → Vector Store
                                               ↓
User Query → Embed → Cosine Similarity Search → Top-K Chunks
                                               ↓
              Build Prompt (Context + Question) → Generate Answer
```

---

## 🔧 Key Components

| Component | Description |
|-----------|-------------|
| `TextChunker` | Splits documents into overlapping chunks |
| `TFIDFEmbedder` | Converts text to TF-IDF vectors |
| `Cosine Similarity` | Measures semantic similarity between vectors |
| `AnswerGenerator` | Builds prompt and generates extractive answers |
| `GenAIQAEngine` | Orchestrates the full RAG pipeline |

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 💡 Features

- Upload custom .txt documents to knowledge base
- Paste text directly
- Adjustable Top-K retrieval
- Adjustable similarity threshold
- View retrieved context chunks with similarity scores
- View the exact prompt sent to LLM
- Chat history
- Pre-loaded AI/ML knowledge base for demo

---

## 📊 Technical Details

- **Chunking:** Recursive word-based with configurable overlap
- **Embedding:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Similarity:** Cosine similarity between sparse TF-IDF vectors
- **Retrieval:** Top-K filtering with similarity threshold
- **Generation:** Extractive QA with question-type detection
- **UI:** Streamlit with custom CSS styling

---

## 🔄 Production Extension

In production, replace the `AnswerGenerator.generate()` method with:

```python
# OpenAI
import openai
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)

# Or use FAISS/Chroma for vector store
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
```

---

## 📁 Project Structure

```
genai_qa_project/
├── app.py          # Streamlit UI
├── qa_engine.py    # Core RAG pipeline
├── requirements.txt
└── README.md
```
