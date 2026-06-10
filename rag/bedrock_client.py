import json
import time
import random
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from config import AWS_REGION, BEDROCK_MODEL_ID, EMBEDDING_MODEL_ID


client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(
        retries={
            "max_attempts": 10,
            "mode": "adaptive"
        }
    )
)


def invoke_with_retry(model_id, body, max_retries=8):
    for attempt in range(max_retries):
        try:
            return client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code in ["ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"]:
                wait = min(30, (2 ** attempt) + random.uniform(0, 1))
                time.sleep(wait)
            else:
                raise

    raise Exception("Bedrock is throttling. Wait 1–2 minutes and try again.")


def get_embedding(text):
    response = invoke_with_retry(
        EMBEDDING_MODEL_ID,
        {
            "inputText": text[:8000]
        }
    )

    result = json.loads(response["body"].read())
    return result["embedding"]


def generate_answer(prompt):
    response = invoke_with_retry(
        BEDROCK_MODEL_ID,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 700,
                "temperature": 0
            }
        }
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]