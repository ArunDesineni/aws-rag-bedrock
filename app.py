import os
import streamlit as st
from config import FAISS_INDEX_PATH
from rag.rag_chain import answer_question

st.set_page_config(
    page_title="GenAI RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 GenAI & Agentic AI RAG Assistant")
st.caption("Powered by Amazon Bedrock Nova 2 Lite + FAISS")

if not os.path.exists(FAISS_INDEX_PATH):
    st.error("Vectorstore not found. Run: python ingest.py")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Answer Mode")

    mode = st.radio(
        "Choose response depth",
        ["Quick Answer", "Standard", "Research Mode"],
        index=1
    )

    mode_settings = {
        "Quick Answer": {
            "top_k": 3,
            "description": "Fast, concise answers."
        },
        "Standard": {
            "top_k": 5,
            "description": "Balanced answer with good context."
        },
        "Research Mode": {
            "top_k": 7,
            "description": "Detailed answer using more PDF context."
        }
    }

    top_k = mode_settings[mode]["top_k"]

    st.caption(mode_settings[mode]["description"])
    st.caption(f"Retrieving {top_k} chunks from PDFs.")

    st.markdown("---")
    st.subheader("Suggested questions")

    suggestions = [
        "What is Generative AI?",
        "What is Agentic AI?",
        "Explain RAG in simple terms.",
        "Compare GenAI and Agentic AI.",
        "What are common use cases of AI agents?"
    ]

    for q in suggestions:
        if st.button(q):
            st.session_state.pending_question = q

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

        
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question from your PDFs...")

if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs and generating answer..."):
            answer, sources = answer_question(question, top_k=top_k)

        st.markdown(answer)

        if sources:
            st.markdown("### Sources")
            for src in sources:
                st.markdown(
                    f"- `{src['source']}` — page **{src['page']}** "
                    f"(similarity: {src['score']})"
                )

    final_response = answer

    if sources:
        final_response += "\n\n### Sources\n"
        for src in sources:
            final_response += (
                f"- {src['source']} — page {src['page']} "
                f"(similarity: {src['score']})\n"
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_response
    })