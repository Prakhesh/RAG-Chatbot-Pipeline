import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- LOAD MODEL ----------------
print("🔹 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- LOAD VECTOR DB ----------------
index = faiss.read_index("vectorstore/index.faiss")

with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("✅ Chatbot ready!\n")

# ---------------- ASK FUNCTION ----------------
def ask(query, top_k=3):
    q_emb = model.encode([query]).astype("float32")

    # normalize (important for cosine similarity)
    faiss.normalize_L2(q_emb)

    scores, idx = index.search(q_emb, top_k)

    results = []
    for i in idx[0]:
        if i < len(chunks):
            results.append(chunks[i])

    return "\n\n".join(results)


# ---------------- CHAT LOOP ----------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk (type 'exit'): ")

        if query.lower() == "exit":
            print("Bye 👋")
            break

        answer = ask(query)

        print("\n📌 Answer:\n")
        print(answer)