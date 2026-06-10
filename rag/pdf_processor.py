import os
import fitz
from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdfs(pdf_dir):
    documents = []

    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, filename)
        doc = fitz.open(path)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                documents.append({
                    "source": filename,
                    "page": page_num,
                    "text": text
                })

    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def create_chunks(documents):
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": i
            })

    return all_chunks