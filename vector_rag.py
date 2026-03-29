"""
Vector-based RAG Engine
-----------------------
Traditional approach: Chunk → Embed → Store in Vector DB → Similarity Search → LLM
"""

import chromadb
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer


@st.cache_resource
def _load_embedder():
    import os
    local_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
    if os.path.exists(local_path):
        return SentenceTransformer(local_path)
    return SentenceTransformer("all-MiniLM-L6-v2")



def _chunk_document(text: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """Split text into fixed-size overlapping chunks."""
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        line_len = len(line.split())
        if current_len + line_len > chunk_size and current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({"text": chunk_text, "index": len(chunks)})
            # Keep overlap
            overlap_lines = []
            overlap_len = 0
            for prev_line in reversed(current_chunk):
                overlap_len += len(prev_line.split())
                if overlap_len > overlap:
                    break
                overlap_lines.insert(0, prev_line)
            current_chunk = overlap_lines
            current_len = overlap_len
        current_chunk.append(line)
        current_len += line_len

    if current_chunk:
        chunks.append({"text": "\n".join(current_chunk), "index": len(chunks)})

    return chunks


class VectorRAG:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini", chunk_size: int = 400, chunk_overlap: int = 80):
        self.openai_client = openai_client
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = _load_embedder()
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.chunks = []

    def index_document(self, text: str) -> dict:
        """Chunk and embed the document into ChromaDB."""
        self.chunks = _chunk_document(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)

        # Reset collection
        try:
            self.chroma_client.delete_collection("rag_demo")
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_demo",
            metadata={"hnsw:space": "cosine"},
        )

        texts = [c["text"] for c in self.chunks]
        embeddings = self.embedder.encode(texts).tolist()
        ids = [f"chunk_{i}" for i in range(len(texts))]

        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)

        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": len(embeddings[0]),
            "avg_chunk_words": sum(len(t.split()) for t in texts) // len(texts),
        }

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k similar chunks for a query."""
        if self.collection is None or self.collection.count() == 0:
            raise RuntimeError("Collection not indexed. Please re-index the document.")
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )
        retrieved = []
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            retrieved.append({"text": doc, "score": round(1 - dist, 4)})
        return retrieved

    def query(self, question: str, top_k: int = 3) -> dict:
        """Full Vector RAG pipeline: retrieve then generate."""
        retrieved = self.retrieve(question, top_k)
        context = "\n\n---\n\n".join([r["text"] for r in retrieved])

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question using ONLY "
                        "the provided context. If the answer is not in the context, "
                        "say 'Not found in the document.' Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.1,
        )

        usage = response.usage
        tokens = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": (usage.prompt_tokens + usage.completion_tokens) if usage else 0,
            "llm_calls": 1,
        }

        return {
            "answer": response.choices[0].message.content,
            "retrieved_chunks": retrieved,
            "method": "Vector RAG",
            "tokens": tokens,
            "details": {
                "chunks_searched": len(self.chunks),
                "chunks_retrieved": top_k,
                "embedding_model": "all-MiniLM-L6-v2",
            },
        }
