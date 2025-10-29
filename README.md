# Simple-RAG
This is a simpel python implementation of Retrieval-Augmented Generation to generate contextual responses to a user querie The script supports document ingestion from PDF, PPTX, and DOCX files, converts them into text, splits them into small chunks, embeds them, and stores them in memory for fast search and retrieval.


The current script Uses local models form huggingface for both embeddings and language generation, running with Ollama.

## Requirements

- Python 3.8+
- [ollama](https://ollama.com/) installed and running locally
- The following Python packages:
  - `tkinter`
  - `os`, `re`
  - `ollama` (Python package)
  - `pypdf`
  - `python-pptx`
  - `python-docx`
  - `langchain-text-splitters`
  - `tiktoken`
 
  Install dependencies with:
```bash
pip install ollama pypdf python-pptx python-docx langchain-text-splitters tiktoken
```

> **Note:** You must download the required Hugging Face models locally.
