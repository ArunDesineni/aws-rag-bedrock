import os
import pickle
import faiss
import numpy as np
from config import FAISS_INDEX_PATH, METADATA_PATH, VECTOR_DIR
from rag.bedrock_client import get_embedding


def build_faiss_index(chunks):
    os.makedirs(VECTOR_DIR, exist_ok=True)

    embeddings = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{len(chunks)}")
        embedding = get_embedding(chunk["text"])
        embeddings.append(embedding)

    vectors = np.array(embeddings).astype("float32")

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index saved.")


def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def search(query, top_k):
    index, metadata = load_faiss_index()

    query_embedding = np.array([get_embedding(query)]).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results