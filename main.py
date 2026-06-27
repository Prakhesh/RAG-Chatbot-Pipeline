from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import os
import fitz
import faiss
import pickle
import ollama
import numpy as np

from sentence_transformers import SentenceTransformer

app = FastAPI(
    title="PDF RAG Assistant"
)

# ----------------------------------
# CONFIG
# ----------------------------------

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ----------------------------------
# LOAD MODEL
# ----------------------------------

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chat_history = []

# ----------------------------------
# REQUEST MODEL
# ----------------------------------

class QuestionRequest(BaseModel):
    question: str


# ----------------------------------
# PDF READER
# ----------------------------------

def read_pdf(path):

    doc = fitz.open(path)

    text = ""

    for page in doc:
        text += page.get_text()

    return text


# ----------------------------------
# CHUNKING
# ----------------------------------

def chunk_text(
    text,
    size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        chunks.append(
            text[start:start+size]
        )

        start += size - overlap

    return chunks


# ----------------------------------
# CREATE VECTOR DB
# ----------------------------------

def create_vector_db():

    all_chunks = []
    metadata = []

    for file in os.listdir(DATA_DIR):

        if not file.endswith(".pdf"):
            continue

        path = os.path.join(
            DATA_DIR,
            file
        )

        text = read_pdf(path)

        chunks = chunk_text(text)

        for chunk in chunks:

            all_chunks.append(chunk)

            metadata.append(
                {
                    "source": file
                }
            )

    if len(all_chunks) == 0:
        return

    embeddings = embed_model.encode(
        all_chunks
    ).astype("float32")

    faiss.normalize_L2(
        embeddings
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        VECTOR_DIR + "/index.faiss"
    )

    with open(
        VECTOR_DIR + "/chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            all_chunks,
            f
        )

    with open(
        VECTOR_DIR + "/metadata.pkl",
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f
        )


# ----------------------------------
# LOAD VECTOR DB
# ----------------------------------

def load_vector_db():

    index = faiss.read_index(
        VECTOR_DIR + "/index.faiss"
    )

    with open(
        VECTOR_DIR + "/chunks.pkl",
        "rb"
    ) as f:

        chunks = pickle.load(f)

    with open(
        VECTOR_DIR + "/metadata.pkl",
        "rb"
    ) as f:

        metadata = pickle.load(f)

    return (
        index,
        chunks,
        metadata
    )


# ----------------------------------
# HOME
# ----------------------------------

@app.get("/")
def home():

    return {
        "message":
        "PDF RAG Assistant API Running"
    }


# ----------------------------------
# PDF UPLOAD API
# ----------------------------------

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):

    save_path = os.path.join(
        DATA_DIR,
        file.filename
    )

    with open(
        save_path,
        "wb"
    ) as buffer:

        buffer.write(
            await file.read()
        )

    create_vector_db()

    return {
        "message":
        "PDF uploaded successfully",
        "file":
        file.filename
    }


# ----------------------------------
# ASK API
# ----------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    (
        index,
        chunks,
        metadata
    ) = load_vector_db()

    q_emb = embed_model.encode(
        [request.question]
    ).astype("float32")

    faiss.normalize_L2(
        q_emb
    )

    scores, indices = index.search(
        q_emb,
        5
    )

    retrieved_chunks = []
    sources = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if score < 0.30:
            continue

        retrieved_chunks.append(
            chunks[idx]
        )

        sources.append(
            metadata[idx]["source"]
        )

    if len(retrieved_chunks) == 0:

        answer = (
            "This information is not "
            "available in the PDF."
        )

    else:

        context = "\n\n".join(
            retrieved_chunks[:3]
        )

        prompt = f"""
Answer ONLY using context.

If answer is missing say:

This information is not available in the PDF.

Context:
{context}

Question:
{request.question}

Answer:
"""

        response = ollama.chat(
            model="tinyllama",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response[
            "message"
        ]["content"]

    chat_history.append(
        {
            "question":
            request.question,
            "answer":
            answer
        }
    )

    return {
        "question":
        request.question,
        "answer":
        answer,
        "sources":
        list(set(sources))
    }


# ----------------------------------
# CHAT HISTORY API
# ----------------------------------

@app.get("/history")
def get_history():

    return chat_history