import streamlit as st
import os
from qa_engine import GenAIQAEngine

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Question Answering System",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .context-box {
        background: #fff9f0;
        border-left: 4px solid #f6a623;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 GenAI Question Answering System</h1>
    <p>Built with RAG Pipeline | LLM + Retrieval-Based Architecture</p>
    <small>Developed by Afifa Tazeen | Python • NLP • Generative AI</small>
</div>
""", unsafe_allow_html=True)

# ── Initialize engine ─────────────────────────────────────────
@st.cache_resource
def load_engine():
    return GenAIQAEngine()

engine = load_engine()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    top_k = st.slider(
        "Top-K Chunks to Retrieve",
        min_value=1, max_value=5, value=3,
        help="Number of relevant context chunks to retrieve"
    )
    
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Minimum similarity score for retrieval"
    )
    
    st.markdown("---")
    st.markdown("## 📊 System Info")
    st.info(f"📄 Documents Loaded: {engine.get_doc_count()}")
    st.info(f"🔢 Total Chunks: {engine.get_chunk_count()}")
    
    st.markdown("---")
    st.markdown("## 🧠 RAG Pipeline")
    st.markdown("""
    **How it works:**
    1. 📄 Load Documents
    2. ✂️ Chunk Text
    3. 🔢 Create Embeddings
    4. 🔍 Retrieve Relevant Chunks
    5. 📝 Build Prompt
    6. 🤖 Generate Answer
    """)

# ── Main Layout ───────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

# ── Left: Document Upload ─────────────────────────────────────
with col1:
    st.markdown("### 📄 Knowledge Base")
    
    tab1, tab2 = st.tabs(["📁 Upload Document", "✏️ Paste Text"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=["txt"],
            help="Upload any .txt document to build knowledge base"
        )
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            if st.button("📥 Add to Knowledge Base"):
                with st.spinner("Processing document..."):
                    chunks_added = engine.add_document(
                        content, uploaded_file.name
                    )
                st.success(f"✅ Added {chunks_added} chunks from '{uploaded_file.name}'")
                st.rerun()
    
    with tab2:
        custom_text = st.text_area(
            "Paste your document text here:",
            height=150,
            placeholder="Paste any text content here — articles, policies, notes..."
        )
        doc_name = st.text_input("Document name:", value="custom_document")
        if st.button("📥 Add Text to Knowledge Base"):
            if custom_text.strip():
                with st.spinner("Processing text..."):
                    chunks_added = engine.add_document(custom_text, doc_name)
                st.success(f"✅ Added {chunks_added} chunks!")
                st.rerun()
            else:
                st.warning("Please enter some text first.")
    
    # Show loaded documents
    docs = engine.get_documents()
    if docs:
        st.markdown("**📚 Loaded Documents:**")
        for doc in docs:
            st.markdown(f"- 📄 `{doc}`")
    
    # Load sample data button
    if st.button("🎯 Load Sample AI Knowledge Base"):
        with st.spinner("Loading sample documents..."):
            engine.load_sample_data()
        st.success("✅ Sample knowledge base loaded!")
        st.rerun()

# ── Right: Q&A Interface ──────────────────────────────────────
with col2:
    st.markdown("### 💬 Ask a Question")
    
    question = st.text_input(
        "Your Question:",
        placeholder="What is machine learning? How does RAG work?",
        key="question_input"
    )
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        ask_btn = st.button("🔍 Get Answer", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        if "chat_history" in st.session_state:
            st.session_state.chat_history = []
        st.rerun()
    
    if ask_btn and question.strip():
        if engine.get_chunk_count() == 0:
            st.warning("⚠️ Please add documents to the knowledge base first, or click 'Load Sample AI Knowledge Base'")
        else:
            with st.spinner("🔍 Retrieving context and generating answer..."):
                result = engine.answer_question(
                    question,
                    top_k=top_k,
                    threshold=similarity_threshold
                )
            
            # Show answer
            st.markdown("#### 🤖 Answer:")
            st.markdown(f"""
            <div class="answer-box">
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Chunks Retrieved", result['chunks_retrieved'])
            with m2:
                st.metric("Avg Similarity", f"{result['avg_similarity']:.2f}")
            with m3:
                st.metric("Confidence", result['confidence'])
            
            # Show retrieved context
            with st.expander("📄 View Retrieved Context Chunks"):
                for i, chunk in enumerate(result['contexts']):
                    st.markdown(f"""
                    <div class="context-box">
                        <strong>Chunk {i+1} | Source: {chunk['source']} | 
                        Similarity: {chunk['similarity']:.3f}</strong><br><br>
                        {chunk['text']}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show prompt used
            with st.expander("🔧 View Prompt Sent to LLM"):
                st.code(result['prompt_used'], language="text")
            
            # Save to history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append({
                "question": question,
                "answer": result['answer']
            })

# ── Chat History ──────────────────────────────────────────────
if "chat_history" in st.session_state and st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📜 Chat History")
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {item['question'][:60]}..."):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<center>
    <small>
        🤖 GenAI QA System | Built with Python, NLP & RAG Architecture<br>
        Developed by <strong>Afifa Tazeen</strong> | 
        Tools: Python • TF-IDF • Cosine Similarity • Streamlit
    </small>
</center>
""", unsafe_allow_html=True)
