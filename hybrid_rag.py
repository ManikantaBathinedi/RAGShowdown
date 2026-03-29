"""
Hybrid RAG Engine
-----------------
Combines both Vector and Vectorless approaches:
1. Vector search for broad candidate retrieval
2. Tree reasoning to refine and validate relevance
3. LLM generates the final answer from filtered context

Best of both worlds: speed of vector search + precision of reasoning.
"""

import json
import re
from openai import OpenAI
from vector_rag import VectorRAG
from vectorless_rag import VectorlessRAG, _get_all_sections, _get_section_content


class HybridRAG:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini",
                 vector_rag: VectorRAG = None, vectorless_rag: VectorlessRAG = None):
        self.openai_client = openai_client
        self.model = model
        self.vector_rag = vector_rag
        self.vectorless_rag = vectorless_rag

    def index_document(self, text: str) -> dict:
        """Return stats from the shared engines (already indexed)."""
        return {
            "vector_index": {"total_chunks": len(self.vector_rag.chunks)},
            "tree_index": {"total_sections": len(_get_all_sections(self.vectorless_rag.tree))},
            "strategy": "Vector search + Tree reasoning reranker",
        }

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Hybrid pipeline:
        1. Get broad candidates from vector search
        2. Get tree-reasoning candidates from vectorless approach
        3. Merge and deduplicate
        4. Use LLM to rerank/filter based on reasoning
        5. Generate final answer
        """
        # Stage 1: Vector retrieval (cast a wide net)
        vector_results = self.vector_rag.retrieve(question, top_k=top_k)

        # Stage 2: Tree-based retrieval (precision picks)
        self.vectorless_rag._last_nav_usage = None
        tree_results = self.vectorless_rag.retrieve(question)
        nav_usage = self.vectorless_rag._last_nav_usage

        # Stage 3: Merge candidates
        all_candidates = []
        seen_texts = set()

        for r in tree_results:
            text_key = r["text"][:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_candidates.append({
                    "text": r["text"],
                    "source": "tree_reasoning",
                    "detail": r.get("section_title", ""),
                })

        for r in vector_results:
            text_key = r["text"][:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_candidates.append({
                    "text": r["text"],
                    "source": "vector_search",
                    "detail": f"similarity={r['score']}",
                })

        # Stage 4: LLM reranks the merged candidates
        candidate_summaries = []
        for i, c in enumerate(all_candidates):
            preview = c["text"][:300].replace("\n", " ")
            candidate_summaries.append(
                f"[{i}] (via {c['source']}) {preview}..."
            )

        rerank_response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance judge. Given a question and numbered "
                        "text candidates, return a JSON array of the indices (numbers) "
                        "of the TOP 3 most relevant candidates, ordered by relevance. "
                        "Example: [2, 0, 4]. Return ONLY the JSON array."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Candidates:\n" +
                        "\n\n".join(candidate_summaries)
                    ),
                },
            ],
            temperature=0.0,
        )
        rerank_usage = rerank_response.usage

        # Parse reranked indices
        raw = rerank_response.choices[0].message.content.strip()
        try:
            json_match = re.search(r"\[.*?\]", raw)
            if json_match:
                ranked_indices = json.loads(json_match.group())
            else:
                ranked_indices = list(range(min(3, len(all_candidates))))
        except (json.JSONDecodeError, ValueError):
            ranked_indices = list(range(min(3, len(all_candidates))))

        # Filter valid indices
        ranked_indices = [
            i for i in ranked_indices
            if isinstance(i, int) and 0 <= i < len(all_candidates)
        ][:3]

        if not ranked_indices:
            ranked_indices = list(range(min(3, len(all_candidates))))

        final_candidates = [all_candidates[i] for i in ranked_indices]

        # Stage 5: Generate answer
        context = "\n\n---\n\n".join([c["text"] for c in final_candidates])

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

        gen_usage = response.usage
        all_usages = [u for u in [nav_usage, rerank_usage, gen_usage] if u]
        tokens = {
            "prompt_tokens": sum(u.prompt_tokens for u in all_usages),
            "completion_tokens": sum(u.completion_tokens for u in all_usages),
            "total_tokens": sum(u.prompt_tokens + u.completion_tokens for u in all_usages),
            "llm_calls": 3,
        }

        return {
            "answer": response.choices[0].message.content,
            "tokens": tokens,
            "retrieved_chunks": [
                {
                    "text": c["text"][:500] + "..." if len(c["text"]) > 500 else c["text"],
                    "source": c["source"],
                    "detail": c["detail"],
                }
                for c in final_candidates
            ],
            "method": "Hybrid RAG",
            "details": {
                "vector_candidates": len(vector_results),
                "tree_candidates": len(tree_results),
                "total_merged": len(all_candidates),
                "final_selected": len(final_candidates),
                "strategy": "Vector broad search → Tree reasoning → LLM rerank",
            },
        }
