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
> EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf',
> LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'



## Limitations

- If the question covers multiple topics at the same time, the system may not be able to provide a good answer. This is because the system only retrieves chunks based on the similarity of the query to the chunks, without considering the context of the query.
The solution could be to have the chatbot to write its own query based on the user's input, then retrieve the knowledge based on the generated query.

- The top N results are returned based on the cosine similarity which may not always give the best results, especially when each chunks contains a lot of information.
To address this issue, we can use a reranking model to re-rank the retrieved chunks based on their relevance to the query.

- The database is stored in memory, which may not be scalable for large datasets. We can use a more efficient vector database such as Qdrant, Pinecone.

 
