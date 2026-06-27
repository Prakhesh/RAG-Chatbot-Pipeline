import faiss
import pickle
import ollama

from sentence_transformers import SentenceTransformer

# ---------------- LOAD MODEL ----------------

print("🔹 Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ---------------- LOAD VECTOR DB ----------------

print("🔹 Loading FAISS index...")

index = faiss.read_index(
    "vectorstore/index.faiss"
)

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

with open(
    "vectorstore/metadata.pkl",
    "rb"
) as f:
    metadata = pickle.load(f)

print("✅ Chatbot Ready!\n")


# ---------------- ASK FUNCTION ----------------

def ask(query,
        top_k=10,
        threshold=0.30):

    # QUERY EMBEDDING

    q_emb = model.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(
        q_emb
    )

    # SEARCH

    scores, indices = index.search(
        q_emb,
        top_k
    )

    retrieved_chunks = []
    sources = []

    # FILTER RESULTS

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx >= len(chunks):
            continue

        if score >= threshold:

            retrieved_chunks.append(
                chunks[idx]
            )

            if idx < len(metadata):

                sources.append(
                    metadata[idx]["source"]
                )

    # NO MATCH FOUND

    if not retrieved_chunks:

        return {
            "answer":
            "This information is not available in the PDF.",
            "sources": []
        }

    # BEST CONTEXT

    context = "\n\n".join(
        retrieved_chunks[:3]
    )

    # PROMPT

    prompt = f"""
You are a strict PDF Question Answering Assistant.

Rules:

1. Answer ONLY from the provided context.
2. Do NOT guess.
3. Do NOT use outside knowledge.
4. Keep answers concise.
5. If answer is missing, reply exactly:
This information is not available in the PDF.

Context:
{context}

Question:
{query}

Answer:
"""

    # OLLAMA

    response = ollama.chat(
        model="tinyllama",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    answer = response[
        "message"
    ]["content"]

    return {
        "answer": answer,
        "sources": list(
            set(sources)
        )
    }


# ---------------- CHAT LOOP ----------------

if __name__ == "__main__":

    while True:

        query = input(
            "\nAsk Question (type 'exit'): "
        )

        if query.lower() == "exit":

            print("\n👋 Goodbye")
            break

        result = ask(query)

        print("\n📌 Answer:\n")
        print(result["answer"])

        if result["sources"]:

            print("\n📄 Sources:")

            for source in result["sources"]:

                print(
                    f"- {source}"
                )