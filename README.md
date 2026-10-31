# Local PDF RAG Application

A fully offline, privacy-focused Retrieval-Augmented Generation (RAG) application that allows users to query large PDF documents. Built with a modern LangChain LCEL architecture, this project processes documents locally using ChromaDB for vector storage and Ollama for embeddings and text generation.

## 🚀 Features
* **100% Offline & Private:** No API keys required. All document processing, embedding, and inference happens locally on your machine.
* **Modern LangChain Architecture:** Utilizes LangChain Expression Language (LCEL) for a clean, modular, and highly readable retrieval and generation pipeline.
* **Efficient Vector Retrieval:** Chunks and embeds massive PDFs (700+ pages) into a persistent local ChromaDB instance for fast semantic search.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** LangChain (LCEL)
* **Vector Database:** ChromaDB
* **Local LLM Engine:** Ollama
* **Embedding Model:** `nomic-embed-text`
* **Generative Model:** `llama3.2`

## 📥 Installation & Setup

1. **Clone the repository**
```bash
git clone [https://github.com/yourusername/rag-model.git](https://github.com/yourusername/rag-model.git)
cd rag-model
```

2. **Set up a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install langchain-chroma langchain-ollama langchain-core chromadb pypdf
```

4. **Download local AI models via Ollama**
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## 🚀 Usage

**1. Ingesting the PDF**
Place your target PDF in the `data/` folder (e.g., `data/sample.pdf`). Run the ingestion script to parse, chunk, embed, and store the document in a local `chroma_db` folder.
```bash
python app.py
```

**2. Querying the Document**
Run the query script to retrieve answers based on the embedded PDF chunks.
```bash
python query.py
```
