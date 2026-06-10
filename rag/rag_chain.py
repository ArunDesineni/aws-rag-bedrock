from config import TOP_K
from rag.vector_store import search
from rag.bedrock_client import generate_answer


def build_prompt(question, contexts):
    context_text = ""

    for i, item in enumerate(contexts, start=1):
        context_text += f"""
Source {i}:
PDF: {item["source"]}
Page: {item["page"]}
Text:
{item["text"]}
"""

    return f"""
You are a helpful RAG assistant for GenAI and Agentic AI PDFs.

Answer the user's question using only the provided context.
If the answer is not in the context, say:
"I could not find this information in the uploaded PDFs."

Be clear, practical, and beginner-friendly.
Include citations using PDF name and page number.

Context:
{context_text}

Question:
{question}

Answer:
"""


def answer_question(question, top_k=TOP_K):
    contexts = search(question, top_k)
    prompt = build_prompt(question, contexts)
    answer = generate_answer(prompt)

    sources = []
    seen = set()

    for item in contexts:
        key = (item["source"], item["page"])
        if key not in seen:
            sources.append({
                "source": item["source"],
                "page": item["page"],
                "score": round(item["score"], 3)
            })
            seen.add(key)

    return answer, sources