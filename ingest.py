import os
import fitz
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"

os.makedirs(VECTOR_DIR, exist_ok=True)

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def read_pdf(path):
    try:
        doc = fitz.open(path)

        text = ""

        for page in doc:
            text += page.get_text()

        return text

    except Exception as e:
        print("Error:", e)
        return ""

def chunk_text(text,
               size=800,
               overlap=150):

    chunks = []

    start = 0

    while start < len(text):
        chunks.append(
            text[start:start+size]
        )

        start += size - overlap

    return chunks

all_chunks = []
all_metadata = []

for file in os.listdir(DATA_DIR):

    if file.endswith(".pdf"):

        path = os.path.join(DATA_DIR, file)

        print("Processing:", file)

        text = read_pdf(path)

        if not text.strip():
            continue

        chunks = chunk_text(text)

        for chunk in chunks:

            all_chunks.append(chunk)

            all_metadata.append(
                {
                    "source": file
                }
            )

if len(all_chunks) == 0:
    raise ValueError(
        "No valid PDF text found"
    )

print("Creating embeddings...")

embeddings = model.encode(
    all_chunks,
    batch_size=32,
    show_progress_bar=True
).astype("float32")

faiss.normalize_L2(
    embeddings
)

index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

index.add(embeddings)

faiss.write_index(
    index,
    os.path.join(
        VECTOR_DIR,
        "index.faiss"
    )
)

with open(
    os.path.join(
        VECTOR_DIR,
        "chunks.pkl"
    ),
    "wb"
) as f:
    pickle.dump(all_chunks, f)

with open(
    os.path.join(
        VECTOR_DIR,
        "metadata.pkl"
    ),
    "wb"
) as f:
    pickle.dump(all_metadata, f)

print("Vector DB Created")