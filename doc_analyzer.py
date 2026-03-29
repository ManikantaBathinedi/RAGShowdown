"""
Document Structure Analyzer
----------------------------
Analyzes an uploaded document and recommends the best RAG approach.
"""

import re


def analyze_document(text: str) -> dict:
    """Analyze document structure and return metrics + recommendation."""
    lines = text.strip().split("\n")
    total_lines = len(lines)
    total_words = len(text.split())
    total_chars = len(text)

    # Count headings by level
    headings = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            headings.append({"level": len(m.group(1)), "title": m.group(2).strip()})

    heading_count = len(headings)
    max_depth = max((h["level"] for h in headings), default=0)

    # Calculate structure ratio (headings per 1000 words)
    structure_ratio = (heading_count / max(total_words, 1)) * 1000

    # Nesting depth analysis
    level_counts = {}
    for h in headings:
        level_counts[h["level"]] = level_counts.get(h["level"], 0) + 1
    has_nesting = max_depth >= 2

    # Average section length (words between headings)
    section_lengths = []
    heading_lines = [i for i, line in enumerate(lines) if re.match(r"^#{1,6}\s+", line)]
    for idx, hl in enumerate(heading_lines):
        end = heading_lines[idx + 1] if idx + 1 < len(heading_lines) else total_lines
        section_text = "\n".join(lines[hl:end])
        section_lengths.append(len(section_text.split()))
    avg_section_len = sum(section_lengths) / max(len(section_lengths), 1)

    # Lists, tables, code blocks
    list_lines = sum(1 for line in lines if re.match(r"^\s*[-*+]\s+|^\s*\d+\.\s+", line))
    table_lines = sum(1 for line in lines if "|" in line and line.strip().startswith("|"))
    code_blocks = len(re.findall(r"```", text)) // 2

    # Scoring
    structure_score = min(100, int(structure_ratio * 10 + (max_depth * 10) + (20 if has_nesting else 0)))
    structure_score = max(0, structure_score)

    # Recommendation logic
    if structure_score >= 60 and heading_count >= 5 and has_nesting:
        best = "Vectorless RAG"
        reason = (
            f"Your document is well-structured with {heading_count} headings "
            f"across {max_depth} nesting levels. The LLM can navigate the heading "
            f"tree effectively — no embeddings needed."
        )
        second = "Hybrid RAG"
    elif structure_score < 25 or heading_count < 3:
        best = "Vector RAG"
        reason = (
            f"Your document has minimal structure ({heading_count} headings). "
            f"Vector similarity search will work better since there's no clear "
            f"heading hierarchy for the LLM to navigate."
        )
        second = "Hybrid RAG"
    else:
        best = "Hybrid RAG"
        reason = (
            f"Your document has moderate structure ({heading_count} headings, "
            f"depth {max_depth}). Combining vector search for broad retrieval "
            f"with tree reasoning for precision gives the best results."
        )
        second = "Vectorless RAG" if structure_score >= 40 else "Vector RAG"

    return {
        "total_words": total_words,
        "total_lines": total_lines,
        "total_chars": total_chars,
        "heading_count": heading_count,
        "max_depth": max_depth,
        "has_nesting": has_nesting,
        "level_counts": level_counts,
        "avg_section_words": round(avg_section_len),
        "structure_score": structure_score,
        "list_lines": list_lines,
        "table_lines": table_lines,
        "code_blocks": code_blocks,
        "recommendation": best,
        "recommendation_reason": reason,
        "runner_up": second,
    }
