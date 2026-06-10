import json
import time
import random
import boto3
from config import AWS_REGION, BEDROCK_MODEL_ID, EMBEDDING_MODEL_ID


client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def invoke_with_retry(model_id, body, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
        except client.exceptions.ThrottlingException:
            wait = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)

    raise Exception("Bedrock is still throttling after retries. Try again later.")


def get_embedding(text):
    response = invoke_with_retry(
        EMBEDDING_MODEL_ID,
        {
            "inputText": text
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
                "maxTokens": 900,
                "temperature": 0
            }
        }
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]