# ✈️ VouyerTrip

Upload PDFs, ask questions, get grounded answers — powered by Gemini + FAISS.

VouyerTrip is a Retrieval-Augmented Generation (RAG) app built with Streamlit. Upload one or more PDF documents, and the app extracts the text, chunks it, embeds it using Google's Gemini embedding model, and builds an in-memory FAISS vector index. Ask a question, and it retrieves the most relevant chunks and generates an answer grounded strictly in your uploaded documents.

## Features

- Upload multiple PDFs at once with live thumbnail previews
- Automatic text extraction and chunking (with configurable overlap)
- Batched embedding calls to Gemini (minimizes API request usage)
- In-memory FAISS similarity search for fast retrieval
- Answers generated only from retrieved context (reduces hallucination)
- Expandable view of the exact chunks used to generate each answer

## Tech Stack

- **Frontend / App**: Streamlit
- **Embeddings & Generation**: Google Gemini API (`gemini-embedding-001`, `gemini-3.5-flash-lite`)
- **Vector Search**: FAISS (`IndexFlatL2`)
- **PDF Parsing**: pypdf, PyMuPDF (for thumbnails)

## Project Structure
rag-doc-qa/
├── app.py # Streamlit UI and app logic
├── rag_core.py # PDF parsing, chunking, embedding, retrieval, generation
├── requirements.txt # Python dependencies
├── assets/ # Background images used in the UI
└── .env # Your Gemini API key (not committed — see setup below)

## Setup

1. **Clone the repo**
```bash
   git clone https://github.com/KrishPGupta/VouyerTrip.git
   cd VouyerTrip
```

2. **Create a virtual environment and install dependencies**
```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
```

3. **Add your Gemini API key**

   Create a `.env` file in the project root:
   GEMINI_API_KEY=your_api_key_here

You can get a free API key from [Google AI Studio](https://ai.google.dev/).

4. **Run the app**
```bash
   streamlit run app.py
```

   The app will open at `http://localhost:8501`.

## Usage

1. Upload one or more PDF files on the left panel.
2. Click **Process documents** to extract, chunk, and index the text.
3. Type a question in the right panel and click **Get answer**.
4. Expand **Retrieved chunks used** to see exactly which parts of the document informed the answer.

## Notes on API Quota

This app uses the Gemini API free tier, which has a daily request limit (currently 1,000 requests/day per project for embeddings). Embedding calls are batched (up to 100 chunks per request) to minimize usage, and larger documents will naturally use more of the quota. If you hit a `429 RESOURCE_EXHAUSTED` error, the daily limit has been reached and will reset at midnight Pacific Time.

This project is intended for local/personal use or evaluation rather than public multi-user deployment, since the free-tier quota is shared across all requests made with a single API key.

## License

No license specified.

## Deployed Live

https://vouyertrip.streamlit.app/

## Author

Krish Gupta
