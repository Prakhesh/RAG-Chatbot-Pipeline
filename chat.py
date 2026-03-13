import faiss
import pickle
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

VECTOR_DIR = "vectorstore/"
SIMILARITY_THRESHOLD = 0.70

print("🔹 Loading models...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")
with open(f"{VECTOR_DIR}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def get_answer(query, debug=True):
    q_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, idxs = index.search(q_emb, k=5)

    print("\n🟢 Debug Info:")
    print("Query:", query)
    print("Distances (cosine):", scores[0])

    # Threshold check
    if scores[0][0] < SIMILARITY_THRESHOLD:
        return "❌ This information is not available in the PDF."

    candidates = [chunks[i] for i in idxs[0]]
    pairs = [[query, c] for c in candidates]

    rerank_scores = rerank_model.predict(pairs)
    best_chunk = candidates[np.argmax(rerank_scores)]

    print("Top Matching Chunks:")
    for c in candidates:
        print("-", c[:80], "...")

    prompt = f"""
Answer ONLY using the context below.
If answer not found, say:
"This information is not available in the PDF."

Context:
{best_chunk}

Question:
{query}
"""

    response = ollama.chat(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )

    return response["message"]["content"]

# CLI loop
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (type 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", get_answer(q))
