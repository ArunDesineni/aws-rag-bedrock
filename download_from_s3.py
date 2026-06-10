import os
import boto3
from config import S3_BUCKET_NAME, S3_PREFIX, PDF_DIR

os.makedirs(PDF_DIR, exist_ok=True)

s3 = boto3.client("s3")

response = s3.list_objects_v2(
    Bucket=S3_BUCKET_NAME,
    Prefix=S3_PREFIX
)

objects = response.get("Contents", [])

if not objects:
    print("No files found in S3 bucket.")
    exit()

downloaded = 0

for obj in objects:
    key = obj["Key"]

    if not key.lower().endswith(".pdf"):
        continue

    filename = os.path.basename(key)
    local_path = os.path.join(PDF_DIR, filename)

    if os.path.exists(local_path):
        print(f"Skipping existing file: {filename}")
        continue

    print(f"Downloading {key} -> {local_path}")
    s3.download_file(S3_BUCKET_NAME, key, local_path)
    downloaded += 1

print(f"Done. Downloaded {downloaded} new PDF(s).")