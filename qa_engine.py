"""
GenAI Question Answering Engine
================================
Core RAG (Retrieval-Augmented Generation) Pipeline

Pipeline:
1. Document Loading
2. Text Chunking
3. TF-IDF Embedding
4. Cosine Similarity Retrieval
5. Prompt Construction
6. Answer Generation

Author: Afifa Tazeen
"""

import re
import math
import string
from collections import defaultdict


# ── Text Preprocessing ────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def tokenize(text: str) -> list:
    """Tokenize text into words, removing punctuation and stopwords."""
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'we',
        'you', 'he', 'she', 'they', 'what', 'which', 'who', 'how'
    }
    text = preprocess_text(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens


# ── TF-IDF Embedder ───────────────────────────────────────────
class TFIDFEmbedder:
    """
    TF-IDF based text embedder for semantic similarity.
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    - TF: how often a word appears in a document
    - IDF: how rare the word is across all documents
    - Rare + frequent in doc = high score = important word
    """

    def __init__(self):
        self.vocabulary = {}      # word → index
        self.idf_scores = {}      # word → IDF score
        self.fitted = False

    def fit(self, documents: list):
        """Build vocabulary and compute IDF scores from documents."""
        # Build vocabulary
        all_words = set()
        for doc in documents:
            tokens = tokenize(doc)
            all_words.update(tokens)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}

        # Compute IDF: log(N / df) where df = document frequency
        N = len(documents)
        doc_freq = defaultdict(int)
        for doc in documents:
            tokens = set(tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1

        self.idf_scores = {}
        for word in self.vocabulary:
            df = doc_freq.get(word, 0)
            if df > 0:
                self.idf_scores[word] = math.log((N + 1) / (df + 1)) + 1
            else:
                self.idf_scores[word] = 1.0

        self.fitted = True

    def transform(self, text: str) -> dict:
        """
        Convert text to TF-IDF vector (sparse dictionary format).
        Returns: {word: tfidf_score}
        """
        tokens = tokenize(text)
        if not tokens:
            return {}

        # Compute TF
        tf = defaultdict(float)
        for token in tokens:
            tf[token] += 1
        for token in tf:
            tf[token] /= len(tokens)

        # Compute TF-IDF
        tfidf = {}
        for token, tf_score in tf.items():
            if token in self.vocabulary:
                idf = self.idf_scores.get(token, 1.0)
                tfidf[token] = tf_score * idf

        return tfidf

    def cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        """
        Compute cosine similarity between two TF-IDF vectors.
        
        cosine_sim = (A · B) / (||A|| × ||B||)
        Range: 0 (no similarity) to 1 (identical)
        """
        if not vec1 or not vec2:
            return 0.0

        # Dot product
        common_words = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[w] * vec2[w] for w in common_words)

        # Magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)


# ── Text Chunker ──────────────────────────────────────────────
class TextChunker:
    """
    Splits documents into smaller overlapping chunks.
    
    Why chunk?
    - LLMs have context window limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size  # words per chunk
        self.overlap = overlap        # overlapping words between chunks

    def chunk(self, text: str, source: str = "unknown") -> list:
        """
        Split text into overlapping chunks.
        Returns list of {'text': ..., 'source': ..., 'chunk_id': ...}
        """
        words = text.split()
        chunks = []
        chunk_id = 0
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])

            if len(chunk_text.strip()) > 30:  # skip tiny chunks
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': chunk_id
                })
                chunk_id += 1

            if end >= len(words):
                break

            # Move start forward with overlap
            start = end - self.overlap

        return chunks


# ── Answer Generator ──────────────────────────────────────────
class AnswerGenerator:
    """
    Generates answers from retrieved context using rule-based + extractive methods.
    
    In production: Replace generate() with OpenAI/Gemini/HuggingFace API call.
    This version demonstrates the prompt construction and extractive QA pipeline.
    """

    def build_prompt(self, question: str, contexts: list) -> str:
        """
        Build a structured RAG prompt combining context + question.
        
        This is the exact format used when calling LLM APIs like OpenAI.
        """
        context_text = "\n\n".join([
            f"[Context {i+1} | Source: {c['source']}]\n{c['text']}"
            for i, c in enumerate(contexts)
        ])

        prompt = f"""You are an intelligent Question Answering assistant. 
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this question."
Be concise, accurate, and helpful.

================== CONTEXT ==================
{context_text}
=============================================

Question: {question}

Answer:"""
        return prompt

    def generate(self, question: str, contexts: list) -> str:
        """
        Generate answer from contexts using extractive + rule-based approach.
        
        NOTE: In production, this would call:
        - OpenAI: openai.chat.completions.create(...)
        - Gemini: genai.GenerativeModel(...).generate_content(...)
        - HuggingFace: pipeline('question-answering', ...)
        """
        if not contexts:
            return "I don't have enough information to answer this question. Please add relevant documents to the knowledge base."

        question_tokens = set(tokenize(question))
        combined_context = ' '.join([c['text'] for c in contexts])

        # Split into sentences for extractive QA
        sentences = re.split(r'[.!?]+', combined_context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Score each sentence by relevance to question
        scored_sentences = []
        for sent in sentences:
            sent_tokens = set(tokenize(sent))
            overlap = len(question_tokens & sent_tokens)
            score = overlap / max(len(question_tokens), 1)
            if score > 0:
                scored_sentences.append((score, sent))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Build answer from top relevant sentences
        if not scored_sentences:
            # Fallback: return first part of most relevant chunk
            return f"Based on the retrieved context: {contexts[0]['text'][:400]}..."

        # Take top 3 most relevant sentences
        top_sentences = [s for _, s in scored_sentences[:3]]

        # Detect question type and format answer
        q_lower = question.lower()

        if any(w in q_lower for w in ['what is', 'what are', 'define', 'explain']):
            answer = "Based on the knowledge base: " + ". ".join(top_sentences) + "."
        elif any(w in q_lower for w in ['how', 'process', 'steps', 'work']):
            answer = "Here's how it works: " + ". ".join(top_sentences) + "."
        elif any(w in q_lower for w in ['why', 'reason', 'purpose']):
            answer = "The reason is: " + ". ".join(top_sentences) + "."
        elif any(w in q_lower for w in ['when', 'time', 'date']):
            answer = "Regarding timing: " + ". ".join(top_sentences) + "."
        else:
            answer = ". ".join(top_sentences) + "."

        # Clean up answer
        answer = re.sub(r'\s+', ' ', answer).strip()
        if answer and not answer.endswith('.'):
            answer += '.'

        return answer if answer else "Based on the retrieved context: " + contexts[0]['text'][:300]


# ── Main QA Engine ────────────────────────────────────────────
class GenAIQAEngine:
    """
    Complete RAG-based Question Answering Engine.
    
    Architecture:
    ┌─────────────────────────────────────────────┐
    │              GenAI QA Engine                │
    │                                             │
    │  Documents → Chunks → Embeddings            │
    │       ↓                                     │
    │  Query → Embed → Retrieve → Prompt → Answer │
    └─────────────────────────────────────────────┘
    """

    def __init__(self):
        self.chunker = TextChunker(chunk_size=80, overlap=15)
        self.embedder = TFIDFEmbedder()
        self.generator = AnswerGenerator()

        self.chunks = []            # all text chunks
        self.chunk_vectors = []     # TF-IDF vectors for each chunk
        self.documents = []         # document names
        self.fitted = False

        # Load sample data by default
        self.load_sample_data()

    # ── Document Management ───────────────────────────────────
    def add_document(self, text: str, name: str = "document") -> int:
        """Add a document to the knowledge base."""
        new_chunks = self.chunker.chunk(text, source=name)
        self.chunks.extend(new_chunks)
        if name not in self.documents:
            self.documents.append(name)
        self._refit_embedder()
        return len(new_chunks)

    def _refit_embedder(self):
        """Refit TF-IDF embedder on all current chunks."""
        if not self.chunks:
            return
        all_texts = [c['text'] for c in self.chunks]
        self.embedder.fit(all_texts)
        self.chunk_vectors = [
            self.embedder.transform(c['text']) for c in self.chunks
        ]
        self.fitted = True

    def get_doc_count(self) -> int:
        return len(self.documents)

    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_documents(self) -> list:
        return self.documents

    # ── Retrieval ─────────────────────────────────────────────
    def retrieve(self, question: str, top_k: int = 3,
                 threshold: float = 0.3) -> list:
        """
        Retrieve top-K most relevant chunks for the question.
        
        Steps:
        1. Embed the question using TF-IDF
        2. Compute cosine similarity with all chunk vectors
        3. Filter by threshold
        4. Return top-K ranked results
        """
        if not self.fitted or not self.chunks:
            return []

        query_vector = self.embedder.transform(question)

        # Score all chunks
        scored = []
        for i, chunk_vec in enumerate(self.chunk_vectors):
            sim = self.embedder.cosine_similarity(query_vector, chunk_vec)
            if sim >= threshold:
                scored.append({
                    'text': self.chunks[i]['text'],
                    'source': self.chunks[i]['source'],
                    'chunk_id': self.chunks[i]['chunk_id'],
                    'similarity': round(sim, 4)
                })

        # Sort by similarity descending
        scored.sort(key=lambda x: x['similarity'], reverse=True)

        return scored[:top_k]

    # ── Answer ────────────────────────────────────────────────
    def answer_question(self, question: str,
                        top_k: int = 3,
                        threshold: float = 0.3) -> dict:
        """
        Full RAG pipeline: retrieve context → build prompt → generate answer.
        
        Returns dict with answer, contexts, metrics, and prompt used.
        """
        # Step 1: Retrieve relevant contexts
        contexts = self.retrieve(question, top_k=top_k, threshold=threshold)

        # Step 2: Build prompt (what would be sent to LLM API)
        prompt = self.generator.build_prompt(question, contexts)

        # Step 3: Generate answer
        if contexts:
            answer = self.generator.generate(question, contexts)
            avg_sim = sum(c['similarity'] for c in contexts) / len(contexts)
            confidence = "High" if avg_sim > 0.5 else "Medium" if avg_sim > 0.3 else "Low"
        else:
            answer = "I couldn't find relevant information in the knowledge base. Please try rephrasing or add more documents."
            avg_sim = 0.0
            confidence = "None"

        return {
            'answer': answer,
            'contexts': contexts,
            'chunks_retrieved': len(contexts),
            'avg_similarity': avg_sim,
            'confidence': confidence,
            'prompt_used': prompt,
            'question': question
        }

    # ── Sample Data ───────────────────────────────────────────
    def load_sample_data(self):
        """Load a sample AI/ML knowledge base for demonstration."""
        sample_docs = {
            "AI_Fundamentals": """
Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think and learn.
Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed.
Deep learning is a subset of machine learning using neural networks with many layers to learn complex patterns.
Natural Language Processing (NLP) allows computers to understand and generate human language.
Computer Vision enables machines to interpret and understand visual information from images and videos.

Supervised learning uses labeled training data where the algorithm learns input-output mappings.
Unsupervised learning finds hidden patterns in data without labeled examples.
Reinforcement learning trains agents by rewarding desired behaviors and punishing undesired ones.

Neural networks are inspired by the human brain and consist of interconnected nodes or neurons.
Each neuron receives input, applies an activation function, and passes output to the next layer.
The most common activation functions are ReLU, Sigmoid, and Tanh.
Backpropagation is the algorithm used to train neural networks by computing gradients.
Gradient descent optimizes model weights by iteratively moving in the direction of steepest descent.
""",
            "RAG_Systems": """
RAG stands for Retrieval-Augmented Generation, a technique that enhances LLMs with external knowledge.
RAG solves the problem of knowledge cutoff, hallucination, and lack of private data access in LLMs.
The RAG pipeline consists of document loading, chunking, embedding, retrieval, and generation steps.

Document chunking splits large texts into smaller pieces for more precise retrieval.
Chunk overlap ensures that context is not lost at the boundaries between chunks.
Embeddings convert text into numerical vectors where similar meanings produce similar vectors.
Cosine similarity measures how similar two vectors are, ranging from 0 to 1.
Vector databases store embeddings and enable fast similarity search across millions of chunks.

Popular vector databases include FAISS, Chroma, Pinecone, Weaviate, and Qdrant.
The retrieval step finds the most relevant chunks for a given user query.
Top-K retrieval selects the K most similar chunks above a similarity threshold.
The prompt combines retrieved context with the user question for the LLM to answer.
LLMs grounded in retrieved context produce more accurate and verifiable answers.
RAG reduces hallucination by forcing the model to base answers on retrieved documents.
""",
            "LLM_Transformers": """
Large Language Models (LLMs) are neural networks trained on massive text datasets to understand and generate language.
The Transformer architecture is the foundation of all modern LLMs including GPT, BERT, Claude, and Gemini.
Self-attention is the key mechanism allowing every token to look at every other token in a sequence.
Multi-head attention runs multiple attention operations in parallel to capture different relationships.
Positional encoding adds position information to token embeddings since Transformers process all tokens simultaneously.

BERT is an encoder-only model that reads text bidirectionally, ideal for understanding tasks.
GPT is a decoder-only model that generates text autoregressively from left to right.
Pre-training trains LLMs on billions of tokens to learn language, facts, and reasoning.
Fine-tuning adapts a pre-trained model to specific tasks using smaller labeled datasets.
RLHF (Reinforcement Learning from Human Feedback) aligns LLMs with human preferences.

LoRA is a parameter-efficient fine-tuning technique that trains only small low-rank matrices.
LoRA reduces trainable parameters by 99% while maintaining model quality close to full fine-tuning.
Hallucination occurs when an LLM generates confident but factually incorrect information.
Temperature controls randomness: low temperature gives focused answers, high temperature gives creative ones.
Context window is the maximum number of tokens an LLM can process in a single interaction.
""",
            "Computer_Vision": """
Computer Vision enables machines to interpret and understand visual information from images and videos.
Convolutional Neural Networks (CNNs) are the primary architecture for image processing tasks.
A convolution operation applies filters that slide across an image to detect local patterns and features.
Early CNN layers detect simple features like edges and corners while deeper layers detect complex objects.
Pooling layers reduce spatial dimensions while preserving the most important features detected.

Object detection identifies what objects are present and where they are located in an image.
A bounding box defines the location of a detected object using coordinates and dimensions.
YOLO (You Only Look Once) is a real-time object detection algorithm that processes images in one pass.
Non-Maximum Suppression removes duplicate bounding boxes for the same detected object.
Intersection over Union measures overlap between predicted and ground truth bounding boxes.

Image segmentation assigns a class label to every pixel in an image.
Semantic segmentation labels all pixels of the same class with the same color.
Instance segmentation separately identifies each individual object even of the same class.
Object tracking maintains consistent identity of detected objects across video frames.
DeepSORT tracks multiple objects using both position and appearance features.
Optical flow tracks pixel movement between consecutive video frames.
Pose estimation detects human body keypoints to understand body position and activity.
Anomaly detection identifies unusual patterns that deviate from learned normal behavior.
""",
            "Python_ML": """
Python is the most popular programming language for machine learning and data science.
NumPy provides efficient numerical computing with n-dimensional arrays and mathematical functions.
Pandas offers data manipulation and analysis tools built on top of NumPy arrays.
Scikit-learn provides a comprehensive set of machine learning algorithms and preprocessing tools.
TensorFlow is an open-source deep learning framework developed by Google for building neural networks.
PyTorch is an open-source deep learning framework developed by Meta with dynamic computation graphs.
Keras is a high-level neural network API that runs on top of TensorFlow for rapid development.
Streamlit is a Python framework for building interactive web applications for machine learning projects.

LSTM stands for Long Short-Term Memory, a type of recurrent neural network for sequential data.
LSTMs have three gates: forget gate, input gate, and output gate to control information flow.
The forget gate decides what information to discard from the cell state.
The input gate determines what new information to add to the cell state.
Word embeddings convert words into dense numerical vectors capturing semantic relationships.
Tokenization is the process of breaking text into tokens such as words or subwords.
"""
        }

        self.chunks = []
        self.documents = []

        for doc_name, content in sample_docs.items():
            new_chunks = self.chunker.chunk(content, source=doc_name)
            self.chunks.extend(new_chunks)
            self.documents.append(doc_name)

        self._refit_embedder()
