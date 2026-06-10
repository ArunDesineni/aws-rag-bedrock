import os
from dotenv import load_dotenv

load_dotenv()


def get_config(key, default=None):
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key, default)


AWS_REGION = get_config("AWS_DEFAULT_REGION", "eu-north-1")
S3_BUCKET_NAME = get_config("S3_BUCKET_NAME")
S3_PREFIX = get_config("S3_PREFIX", "")

BEDROCK_MODEL_ID = get_config("BEDROCK_MODEL_ID")
EMBEDDING_MODEL_ID = get_config("EMBEDDING_MODEL_ID")

CHUNK_SIZE = int(get_config("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(get_config("CHUNK_OVERLAP", 150))
TOP_K = int(get_config("TOP_K", 5))

PDF_DIR = "data/pdfs"
VECTOR_DIR = "vectorstore"
FAISS_INDEX_PATH = f"{VECTOR_DIR}/faiss.index"
METADATA_PATH = f"{VECTOR_DIR}/metadata.pkl"