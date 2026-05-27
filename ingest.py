import os
import fitz
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/"
VECTOR_DIR = "vectorstore/"

os.makedirs(VECTOR_DIR, exist_ok=True)

print("🔹 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- PDF READ ----------------
def read_pdf(path):
    try:
        doc = fitz.open(path)
        text = ""

        for page in doc:
            text += page.get_text()

        return text

    except Exception:
        print("⚠️ Skipping corrupted PDF:", path)
        return ""


# ---------------- CHUNK ----------------
def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap

    return chunks


# ---------------- LOAD FILES ----------------
all_chunks = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        path = os.path.join(DATA_DIR, file)
        print("Processing:", file)

        text = read_pdf(path)

        if not text.strip():
            continue

        all_chunks.extend(chunk_text(text))


# ---------------- SAFETY CHECK ----------------
if len(all_chunks) == 0:
    raise ValueError("❌ No valid text found in PDFs")


# ---------------- EMBEDDINGS ----------------
print("🔹 Creating embeddings...")
embeddings = model.encode(all_chunks).astype("float32")

# normalize
faiss.normalize_L2(embeddings)

# ---------------- FAISS ----------------
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ---------------- SAVE ----------------
faiss.write_index(index, VECTOR_DIR + "/index.faiss")

with open(VECTOR_DIR + "/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ Vector DB created successfully!")