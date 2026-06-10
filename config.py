import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = os.getenv("S3_PREFIX", "")

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 5))

PDF_DIR = "data/pdfs"
VECTOR_DIR = "vectorstore"
FAISS_INDEX_PATH = f"{VECTOR_DIR}/faiss.index"
METADATA_PATH = f"{VECTOR_DIR}/metadata.pkl"