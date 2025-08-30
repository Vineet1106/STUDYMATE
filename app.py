import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# 1. Load Embedding Model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# 2. Extract Text from PDF
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf_doc:
        text += page.get_text()
    return text

# -------------------------------
# 3. Create Knowledge Base
# -------------------------------
def create_index(text, model):
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]
    embeddings = model.encode(sentences, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, sentences

# -------------------------------
# 4. Search Function
# -------------------------------
def search(query, index, sentences, model, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    results = [sentences[i] for i in indices[0]]
    return results

# -------------------------------
# 5. Streamlit UI
# -------------------------------
st.title("📄 PDF Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and indexing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        index, sentences = create_index(pdf_text, model)
    st.success("✅ PDF processed! Now ask me anything.")

    query = st.text_input("Ask a question:")
    if query:
        results = search(query, index, sentences, model)
        st.write("*Answer:*")
        for r in results:
            st.write("- " + r)