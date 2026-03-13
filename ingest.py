import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

PDF_DIR = "data/"
VECTOR_DIR = "vectorstore/"

os.makedirs(VECTOR_DIR, exist_ok=True)

print("🔹 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

all_text = []
for pdf_file in os.listdir(PDF_DIR):
    if pdf_file.endswith(".pdf"):
        reader = PdfReader(os.path.join(PDF_DIR, pdf_file))
        for page in reader.pages:
            all_text.append(page.extract_text())

full_text = "\n".join(all_text)

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

chunks = split_text(full_text)

print("🔹 Creating embeddings...")
embeddings = embed_model.encode(chunks).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings)

faiss.write_index(index, f"{VECTOR_DIR}/index.faiss")
with open(f"{VECTOR_DIR}/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Vectorstore created successfully!")
