import streamlit as st
import pickle
import faiss
import ollama
import re
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "vectorstore"

# ---------------------------
# Load Models & Vector DB
# ---------------------------
@st.cache_resource
def load_data():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")

    with open(f"{VECTOR_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(f"{VECTOR_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return embed_model, index, chunks, metadata


embed_model, index, chunks, metadata = load_data()

# ---------------------------
# Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Highlight Query
# ---------------------------
def highlight_query(text, query):
    for word in query.split():
        text = re.sub(
            f"({re.escape(word)})",
            r"<mark>\1</mark>",
            text,
            flags=re.IGNORECASE,
        )
    return text


# ---------------------------
# Answer Function
# ---------------------------
def get_answer(query, top_k=5):

    q_emb = embed_model.encode([query]).astype("float32")

    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)

    retrieved_chunks = []
    sources = []

    for score, idx in zip(scores[0], indices[0]):

        if idx >= len(chunks):
            continue

        if score < 0.30:
            continue

        retrieved_chunks.append(chunks[idx])

        if idx < len(metadata):
            sources.append(metadata[idx]["source"])

    if not retrieved_chunks:
        return (
            "This information is not available in the PDF.",
            [],
            [],
            scores[0],
        )

    context = "\n\n".join(retrieved_chunks[:3])

    prompt = f"""
Answer ONLY from the context below.

Rules:
1. Use only the context.
2. Do not use outside knowledge.
3. Do not guess.
4. If answer is not found, reply:
This information is not available in the PDF.

Context:
{context}

Question:
{query}

Answer:
"""

    try:

        response = ollama.chat(
            model="tinyllama",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        answer = response["message"]["content"]

    except Exception as e:

        answer = f"Ollama Error: {str(e)}"

    return (
        answer,
        retrieved_chunks,
        list(set(sources)),
        scores[0],
    )


# ---------------------------
# UI
# ---------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide",
)

st.title("📄 PDF Question Answering Assistant")

st.caption("FAISS + TinyLlama + PDF Only Answers")

# Sidebar
st.sidebar.title("💬 Chat History")

for item in st.session_state.history:
    st.sidebar.write(f"Q: {item['question']}")
    st.sidebar.write(f"A: {item['answer']}")
    st.sidebar.divider()

query = st.text_input(
    "Ask a question from your PDF:"
)

top_k = st.slider(
    "Top K Results",
    min_value=1,
    max_value=10,
    value=5,
)

if st.button("Get Answer"):

    if query.strip():

        with st.spinner("Searching PDF..."):

            answer, matched_chunks, sources, scores = get_answer(
                query,
                top_k,
            )

        st.session_state.history.append(
            {
                "question": query,
                "answer": answer,
            }
        )

        st.subheader("✅ Answer")
        st.success(answer)

        st.subheader("📄 Sources")

        if sources:
            for source in sources:
                st.write("•", source)
        else:
            st.warning("No source found")

        st.subheader("📚 Retrieved Context")

        for chunk in matched_chunks:
            st.markdown(
                highlight_query(chunk, query),
                unsafe_allow_html=True,
            )
            st.divider()

        with st.expander("🔍 Debug Information"):
            st.write("Similarity Scores")
            st.write(scores)