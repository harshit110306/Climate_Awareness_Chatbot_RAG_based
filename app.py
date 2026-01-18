import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Climate Awareness Chatbot")
st.title("🌍 Climate Awareness RAG Chatbot")
st.caption("Powered by Ollama (Local LLM) | SDG 13")

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS DB
vectorstore = FAISS.load_local(
    "faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Local LLM via Ollama
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.3
)


# Prompt
prompt = ChatPromptTemplate.from_template(
    """You are a climate awareness assistant.
Answer clearly using the context below.

Context:
{context}

Question:
{question}
"""
)

query = st.text_input("Ask a climate-related question:")

if query:
    with st.spinner("Thinking..."):
        # Retrieve documents
        docs = retriever.invoke(query)

        # Build context
        context = "\n\n".join(doc.page_content for doc in docs)

        # Format prompt
        messages = prompt.format_messages(
            context=context,
            question=query
        )

        # Call local LLM
        response = llm.invoke(messages)

        st.success(response.content)
