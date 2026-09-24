import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# Set Streamlit Page Configuration
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚")

# Initialize Session State Variables
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, chunks


def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)
    return [chunks[idx] for idx in indices[0] if idx < len(chunks)]


def generate_llm_answer(groq_api_key, context_chunks, question):
    os.environ["GROQ_API_KEY"] = groq_api_key.strip()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    context_str = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the provided context to answer the question. "
        "If the answer cannot be derived from the context, state that you cannot find it."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

    # Standard call targeting openai/gpt-oss-120b directly
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model="openai/gpt-oss-120b",
    )

    return chat_completion.choices[0].message.content


# --- Streamlit UI ---
st.title("📚 PDF RAG with GPT-OSS-120B")

with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Indexing PDF..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            if raw_text.strip():
                chunks = chunk_text(raw_text)
                index, stored_chunks = build_faiss_index(chunks)
                st.session_state.vector_db = index
                st.session_state.chunks = stored_chunks
                st.session_state.chat_history = []
                st.success("Document indexed!")
            else:
                st.error("Could not extract text from PDF.")

# Main Chat Interface
if st.session_state.vector_db is not None:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about your PDF...")

    if user_query:
        if not groq_api_key.strip():
            st.warning("Please enter your Groq API Key in the sidebar.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    try:
                        context = retrieve_relevant_chunks(
                            user_query, st.session_state.vector_db, st.session_state.chunks
                        )
                        answer = generate_llm_answer(groq_api_key, context, user_query)
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
else:
    st.info("Upload a PDF and click 'Process Document' to begin.")
