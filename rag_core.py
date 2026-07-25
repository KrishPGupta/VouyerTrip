import os
import numpy as np
import faiss
from pypdf import PdfReader
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.5-flash-lite"

def extract_text_from_pdfs(uploaded_files):
    """Extract raw text from one or more uploaded PDF files."""
    full_text = ""
    for f in uploaded_files:
        reader = PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping chunks (character-based, simple + reliable)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=100):
    """Get Gemini embeddings for a list of strings, batched to minimize API requests."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        for emb in result.embeddings:
            all_embeddings.append(emb.values)
    return np.array(all_embeddings, dtype="float32")

def build_faiss_index(chunks):
    """Embed chunks and build a FAISS index in memory."""
    vectors = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, chunks

def retrieve_top_chunks(query, index, chunks, k=3):
    """Embed the query and pull the k most similar chunks."""
    query_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_answer(query, retrieved_chunks):
    """Feed retrieved context + query to Gemini and get an answer."""
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )
    return response.text

def get_pdf_thumbnail(uploaded_file, zoom=1.5):
    """Render the first page of a PDF as a PNG thumbnail (bytes). Resets the file pointer after reading."""
    import fitz  # PyMuPDF

    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes
