"""
Benchmark Runner
----------------
Runs a set of questions against all 3 RAG engines and aggregates metrics.
"""

import json
import re
import time


# Pricing per 1M tokens (USD) — gpt-4o-mini rates
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-5-mini": {"input": 1.25, "output": 5.00},
    "default": {"input": 0.15, "output": 0.60},
}


def estimate_cost(tokens: dict, model: str) -> float:
    """Estimate USD cost from token usage."""
    rates = PRICING.get(model, PRICING["default"])
    input_cost = (tokens.get("prompt_tokens", 0) / 1_000_000) * rates["input"]
    output_cost = (tokens.get("completion_tokens", 0) / 1_000_000) * rates["output"]
    return round(input_cost + output_cost, 6)


def generate_benchmark_questions(client, model: str, document_text: str, n: int = 5) -> list[str]:
    """Use LLM to auto-generate benchmark questions from the document."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Generate exactly {n} diverse factual questions about the following document. "
                    "Questions should test different sections and vary in difficulty: "
                    "some specific (exact numbers/names), some requiring synthesis. "
                    "Return ONLY a JSON array of question strings."
                ),
            },
            {
                "role": "user",
                "content": document_text[:6000],
            },
        ],
        temperature=0.7,
    )
    raw = response.choices[0].message.content.strip()
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())[:n]
    except (json.JSONDecodeError, ValueError):
        pass
    return [line.strip().lstrip("0123456789.-) ") for line in raw.split("\n") if line.strip() and "?" in line][:n]


def run_benchmark(engines: dict, questions: list[str], model: str, progress_callback=None) -> dict:
    """Run all questions against all engines, collect metrics."""
    results = {"vector": [], "vectorless": [], "hybrid": []}
    total_steps = len(questions) * 3
    step = 0

    for q_idx, question in enumerate(questions):
        for engine_name in ["vector", "vectorless", "hybrid"]:
            step += 1
            if progress_callback:
                progress_callback(step / total_steps, f"Q{q_idx+1}: {engine_name}...")

            start = time.time()
            try:
                result = engines[engine_name].query(question)
                elapsed = time.time() - start
                tokens = result.get("tokens", {})
                cost = estimate_cost(tokens, model)

                results[engine_name].append({
                    "question": question,
                    "answer": result["answer"],
                    "time": round(elapsed, 2),
                    "tokens": tokens,
                    "cost": cost,
                    "chunks_retrieved": len(result.get("retrieved_chunks", [])),
                    "success": True,
                })
            except Exception as e:
                elapsed = time.time() - start
                results[engine_name].append({
                    "question": question,
                    "answer": f"Error: {e}",
                    "time": round(elapsed, 2),
                    "tokens": {},
                    "cost": 0,
                    "chunks_retrieved": 0,
                    "success": False,
                })

    # Aggregate
    summary = {}
    for engine_name, runs in results.items():
        successful = [r for r in runs if r["success"]]
        total_tokens = sum(r["tokens"].get("total_tokens", 0) for r in successful)
        total_cost = sum(r["cost"] for r in successful)
        avg_time = sum(r["time"] for r in successful) / max(len(successful), 1)
        avg_llm_calls = sum(r["tokens"].get("llm_calls", 0) for r in successful) / max(len(successful), 1)

        summary[engine_name] = {
            "total_questions": len(runs),
            "successful": len(successful),
            "failed": len(runs) - len(successful),
            "avg_time_sec": round(avg_time, 2),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_tokens_per_query": total_tokens // max(len(successful), 1),
            "avg_cost_per_query": round(total_cost / max(len(successful), 1), 6),
            "avg_llm_calls": round(avg_llm_calls, 1),
            "cost_per_1000_queries": round((total_cost / max(len(successful), 1)) * 1000, 4),
        }

    return {"details": results, "summary": summary}
