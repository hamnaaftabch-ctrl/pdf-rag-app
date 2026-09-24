import os
import tempfile
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize Session State Variables
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def load_embedding_model():
    """Load and cache the open-source sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

# Helper Functions
def extract_text_from_pdf(pdf_file):
    """Step 1: Extract raw text from the uploaded PDF file."""
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


def chunk_text(text, chunk_size=500, overlap=50):
    """Step 2a: Break raw text into overlapping word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_faiss_index(chunks):
    """Step 2b & 3: Tokenize, embed chunks, and store them in a FAISS vector index."""
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize vectors for Cosine Similarity search
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product with normalized vectors = Cosine Similarity
    index.add(embeddings)

    return index, chunks


def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    """Step 4: Fetch the most relevant text chunks from FAISS based on the query."""
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    return retrieved_chunks


def generate_llm_answer(groq_api_key, context_chunks, question):
    """Step 5: Send context and user question to Groq API using standard SDK pattern."""
    clean_key = groq_api_key.strip()
    
    # Store key in environment variable as requested
    os.environ["GROQ_API_KEY"] = clean_key

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    context_str = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the following retrieved pieces of context to answer the user's question. "
        "If you do not know the answer or if it cannot be derived from the provided context, "
        "state that you cannot find the answer in the provided document. "
        "Do not use external knowledge or fabricate information."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

    # llama-3.1-8b-instant is active and available on all free Groq accounts
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model="llama-3.1-8b-instant",
    )

    return chat_completion.choices[0].message.content


# --- Streamlit UI Components ---
st.title("📚 PDF Question-Answering (RAG Engine)")
st.markdown("Upload a PDF document, enter your Groq API key, and ask questions directly grounded in your document's context.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")

    st.markdown("---")
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None and st.button("Process Document"):
        with st.spinner("Extracting text and generating vector index..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            if raw_text.strip():
                chunks = chunk_text(raw_text)
                index, stored_chunks = build_faiss_index(chunks)

                st.session_state.vector_db = index
                st.session_state.chunks = stored_chunks
                st.session_state.chat_history = []
                st.success(f"Processed {len(stored_chunks)} document chunks successfully!")
            else:
                st.error("No readable text found in the uploaded PDF.")

# Main Chat Interface
if st.session_state.vector_db is not None:
    st.subheader("💬 Ask a question about your PDF")

    # Render previous conversation history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask something about the uploaded document...")

    if user_query:
        if not groq_api_key or not groq_api_key.strip():
            st.warning("⚠️ Please enter a valid Groq API Key in the sidebar before asking questions.")
        else:
            # Display user prompt
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching document & generating answer..."):
                    try:
                        retrieved_context = retrieve_relevant_chunks(
                            user_query, st.session_state.vector_db, st.session_state.chunks
                        )
                        answer = generate_llm_answer(groq_api_key, retrieved_context, user_query)

                        st.markdown(answer)

                        # Show retrieved context in an expander for transparency
                        with st.expander("🔍 View Retrieved Context Chunks"):
                            for i, chunk in enumerate(retrieved_context, 1):
                                st.write(f"**Chunk {i}:**")
                                st.caption(chunk)

                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Please upload a PDF file and click **Process Document** in the sidebar to begin.")
