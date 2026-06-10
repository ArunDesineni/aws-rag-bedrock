import os
from config import PDF_DIR, FAISS_INDEX_PATH
from rag.pdf_processor import extract_text_from_pdfs, create_chunks
from rag.vector_store import build_faiss_index


if os.path.exists(FAISS_INDEX_PATH):
    print("Vectorstore already exists. Skipping ingestion.")
    exit()

print("Extracting PDF text...")
documents = extract_text_from_pdfs(PDF_DIR)

print(f"Extracted {len(documents)} page(s).")

print("Creating chunks...")
chunks = create_chunks(documents)

print(f"Created {len(chunks)} chunk(s).")

print("Building FAISS index...")
build_faiss_index(chunks)

print("Ingestion complete.")