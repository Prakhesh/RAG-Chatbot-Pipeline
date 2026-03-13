import streamlit as st
import pickle
import faiss
import numpy as np
import re
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder

VECTOR_DIR = "vectorstore/"

# ----------------------------
# Load models & data
# ----------------------------
@st.cache_resource
def load_all():
    st.info("🔹 Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    st.info("🔹 Loading re-ranker model...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    st.info("🔹 Loading FAISS index...")
    index = faiss.read_index(VECTOR_DIR + "index.faiss")

    with open(VECTOR_DIR + "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return embed_model, reranker, index, chunks


embed_model, reranker, index, chunks = load_all()

# ----------------------------
# Highlight query words
# ----------------------------
def highlight_query(text, query):
    for word in query.split():
        text = re.sub(
            f"({re.escape(word)})",
            r"<mark>\1</mark>",
            text,
            flags=re.IGNORECASE,
        )
    return text


# ----------------------------
# Core QA logic (FAISS + RERANK)
# ----------------------------
def get_answer(query, faiss_k, rerank_threshold):
    # ---- FAISS retrieval ----
    q_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, faiss_k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    # ---- Re-ranking ----
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    rerank_scores = reranker.predict(pairs)

    # ---- Filter by threshold ----
    final_chunks = []
    debug_items = []

    for chunk, score in zip(retrieved_chunks, rerank_scores):
        debug_items.append((score, chunk))
        if score >= rerank_threshold:
            final_chunks.append(chunk)

    if not final_chunks:
        return (
            "❌ This information is not available in the PDF.",
            distances[0],
            rerank_scores,
            [],
        )

    context = "\n\n".join(final_chunks)

    prompt = f"""
You must answer ONLY using the context below.
If the answer is not present, say:
"This information is not available in the PDF."

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )

    return response["message"]["content"], distances[0], rerank_scores, final_chunks


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 Local PDF Chatbot (FAISS + Re-Ranker + TinyLLaMA)")
st.caption("Answers strictly from PDF content only")

query = st.text_input("Ask a question from the PDF:")

faiss_k = st.slider("FAISS Top-K Results", 3, 15, 8)

rerank_threshold = st.slider(
    "Re-ranker Threshold (higher = stricter)",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.1,
)

if st.button("Get Answer") and query:
    with st.spinner("🔍 Retrieving, re-ranking & generating answer..."):
        answer, faiss_dist, rerank_scores, matched_chunks = get_answer(
            query, faiss_k, rerank_threshold
        )

    # ---- Debug Info ----
    st.subheader("🔍 Debug Info")

    st.write("FAISS Distances:")
    st.write(faiss_dist)

    st.write("Re-ranker scores (before threshold):")
    st.write(rerank_scores)

    st.subheader("Matched PDF Context:")
    if matched_chunks:
        for chunk in matched_chunks:
            st.markdown(
                "- " + highlight_query(chunk, query),
                unsafe_allow_html=True,
            )
    else:
        st.warning("No chunks passed the re-ranker threshold.")

    # ---- Final Answer ----
    st.subheader("✅ Answer")
    st.success(answer)
