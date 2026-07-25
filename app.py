import streamlit as st
from dotenv import load_dotenv
import os
import base64

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = st.secrets.get("GEMINI_API_KEY", "")

from rag_core import extract_text_from_pdfs, chunk_text, build_faiss_index, retrieve_top_chunks, generate_answer, get_pdf_thumbnail

st.set_page_config(page_title="VouyerTrip", page_icon="✈️", layout="wide")

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_b64 = img_to_base64("assets/pietro-de-grandi-T7K4aEPoGGk-unsplash.jpg")
wing_b64 = img_to_base64("assets/tom-barrett-M0AWNxnLaMw-unsplash.jpg")

st.markdown(f"""
<style>
.stApp {{
    background-image: linear-gradient(rgba(10,15,20,0.72), rgba(10,15,20,0.82)), url("data:image/jpeg;base64,{bg_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Make the sticky Streamlit header opaque so scrolled content doesn't bleed through it */
[data-testid="stHeader"] {{
    background-color: rgba(8,12,16,0.95);
}}

.block-container {{
    padding-top: 6rem;
    padding-bottom: 2rem;
    max-width: 100%;
    overflow: visible;
}}

.st-key-upload_panel {{
    background-image: linear-gradient(rgba(10,15,25,0.55), rgba(10,15,25,0.75)), url("data:image/jpeg;base64,{wing_b64}");
    background-size: cover;
    background-position: center;
    border-radius: 16px;
    padding: 2rem 1.5rem;
    min-height: 620px;
}}

.st-key-qa_panel {{
    background-color: rgba(10,15,20,0.55);
    border-radius: 16px;
    padding: 2rem 2rem;
    min-height: 620px;
}}

.hero-wrap {{
    overflow: visible;
    margin-top: 30px;
    position: relative;
    z-index: 1;
}}
.hero-title {{
    color: white;
    font-size: 2.1rem;
    font-weight: 700;
    text-shadow: 0 2px 10px rgba(0,0,0,0.6);
    line-height: 1.4;
    overflow: visible;
}}
.hero-subtitle {{
    color: #dcdcdc;
    font-size: 0.95rem;
    margin-top: 0.3rem;
    margin-bottom: 1.5rem;
}}
h3 {{
    color: white !important;
}}
label, .stMarkdown p {{
    color: #eaeaea !important;
}}

.pdf-thumb-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 0.8rem;
}}
.pdf-thumb-card {{
    width: 110px;
    text-align: center;
}}
.pdf-thumb-card img {{
    width: 110px;
    height: 155px;
    object-fit: cover;
    border-radius: 6px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.15);
    display: block;
}}
.pdf-thumb-card .cap {{
    color: #dcdcdc;
    font-size: 0.68rem;
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
</style>

<div class="hero-wrap">
<div class="hero-title">✈️ VouyerTrip</div>
<div class="hero-subtitle">Upload PDFs, ask questions, get grounded answers — powered by Gemini + FAISS</div>
</div>
""", unsafe_allow_html=True)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

left, right = st.columns([3, 7])

with left:
    with st.container(key="upload_panel"):
        st.markdown("### 1. Upload documents")
        uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            cards = []
            for f in uploaded_files:
                try:
                    thumb_bytes = get_pdf_thumbnail(f, zoom=0.6)
                    thumb_b64 = base64.b64encode(thumb_bytes).decode()
                    cards.append(f'<div class="pdf-thumb-card"><img src="data:image/png;base64,{thumb_b64}" /><div class="cap">{f.name}</div></div>')
                except Exception:
                    cards.append(f'<div class="pdf-thumb-card"><div class="cap">Preview unavailable<br>{f.name}</div></div>')
            row_html = '<div class="pdf-thumb-row">' + ''.join(cards) + '</div>'
            st.markdown(row_html, unsafe_allow_html=True)

        if st.button("Process documents", disabled=not uploaded_files):
            with st.spinner("Extracting text and building index..."):
                raw_text = extract_text_from_pdfs(uploaded_files)
                chunks = chunk_text(raw_text)
                index, chunks = build_faiss_index(chunks)
                st.session_state.index = index
                st.session_state.chunks = chunks
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")

with right:
    with st.container(key="qa_panel"):
        st.markdown("### 2. Ask a question")
        query = st.text_input("Your question")

        if st.button("Get answer", disabled=not query or st.session_state.index is None):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                top_chunks = retrieve_top_chunks(query, st.session_state.index, st.session_state.chunks, k=3)
                answer = generate_answer(query, top_chunks)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved chunks used")
            for i, c in enumerate(top_chunks, 1):
                with st.expander(f"Chunk {i}"):
                    st.write(c)

        if st.session_state.index is None:
            st.info("Upload and process a PDF on the left to get started.")
