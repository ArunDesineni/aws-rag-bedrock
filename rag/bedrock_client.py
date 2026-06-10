import json
import boto3
from config import AWS_REGION, BEDROCK_MODEL_ID, EMBEDDING_MODEL_ID


client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def get_embedding(text):
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({
            "inputText": text
        })
    )

    result = json.loads(response["body"].read())
    return result["embedding"]


def generate_answer(prompt):
    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 900,
                "temperature": 0.2
            }
        })
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]