"""
Vectorless RAG Engine
---------------------
PageIndex-inspired approach: Parse headings → Build tree index → LLM reasons
through the tree → Retrieve relevant sections → LLM generates answer.

No embeddings. No vector database. Pure reasoning-based retrieval.
"""

import json
import re
from openai import OpenAI


def _parse_markdown_tree(text: str) -> dict:
    """Parse a markdown document into a hierarchical tree based on headings."""
    lines = text.split("\n")
    root = {
        "title": "Document Root",
        "level": 0,
        "content": "",
        "children": [],
        "start_line": 0,
        "end_line": len(lines),
    }

    stack = [root]
    current_content_lines = []
    current_node = root

    for i, line in enumerate(lines):
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save accumulated content to current node
            if current_content_lines:
                current_node["content"] = "\n".join(current_content_lines).strip()
                current_content_lines = []

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            node = {
                "title": title,
                "level": level,
                "content": "",
                "children": [],
                "start_line": i,
                "end_line": len(lines),
            }

            # Find parent: pop stack until we find a node with lower level
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()

            # Set end_line of previous sibling
            parent = stack[-1]
            if parent["children"]:
                parent["children"][-1]["end_line"] = i

            parent["children"].append(node)
            stack.append(node)
            current_node = node
        else:
            current_content_lines.append(line)

    # Save last content
    if current_content_lines:
        current_node["content"] = "\n".join(current_content_lines).strip()

    return root


def _parse_plaintext_tree(text: str) -> dict:
    """Fallback parser for plain text (PDF/DOCX) with no markdown headings.
    Detects headings via heuristics, falls back to paragraph-based sections."""
    lines = text.split("\n")
    root = {
        "title": "Document Root",
        "level": 0,
        "content": "",
        "children": [],
        "start_line": 0,
        "end_line": len(lines),
    }

    candidates = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) < 3:
            continue

        prev_blank = (i == 0) or not lines[i - 1].strip()
        next_blank = (i >= len(lines) - 1) or not lines[i + 1].strip()

        # Pattern 1: Numbered heading  "1. Title", "2.1 Sub", "1.1.1 Deep"
        nm = re.match(r"^(\d+(?:\.\d+)*)[.\)]\s+(.+)", s)
        if nm and len(s) < 120:
            level = nm.group(1).count(".") + 1
            candidates.append((i, s, level))
            continue

        # Pattern 2: ALL CAPS line (at least 2 words)
        if s.isupper() and len(s) < 100 and len(s.split()) >= 2:
            candidates.append((i, s, 1))
            continue

        # Pattern 3: "Part N", "Chapter N", "Section N" prefixes
        if re.match(r"^(Part|Chapter|Section|Appendix|Module)\s+\w", s, re.IGNORECASE) and len(s) < 120:
            if prev_blank:
                candidates.append((i, s, 1))
                continue

        # Pattern 4: Short standalone line between blank lines, not a sentence
        if len(s) < 80 and prev_blank and next_blank:
            if not s.endswith((".", ",", ";", "?")):
                if len(s.split()) <= 10:
                    candidates.append((i, s, 1))
                    continue

    if len(candidates) >= 2:
        # Build sections from detected heading candidates
        for idx, (line_num, title, level) in enumerate(candidates):
            end_line = candidates[idx + 1][0] if idx + 1 < len(candidates) else len(lines)
            content = "\n".join(lines[line_num + 1 : end_line]).strip()
            node = {
                "title": title,
                "level": level,
                "content": content,
                "children": [],
                "start_line": line_num,
                "end_line": end_line,
            }
            root["children"].append(node)
    else:
        # Last resort: split into paragraph groups → ~8-12 sections
        paragraphs = []
        current_start = None
        for i, line in enumerate(lines):
            if line.strip():
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    paragraphs.append((current_start, i))
                    current_start = None
        if current_start is not None:
            paragraphs.append((current_start, len(lines)))

        if not paragraphs:
            return root

        target = min(12, max(4, len(paragraphs) // 3))
        group_size = max(1, len(paragraphs) // target)

        for j in range(0, len(paragraphs), group_size):
            group = paragraphs[j : j + group_size]
            start = group[0][0]
            end = group[-1][1]
            first = lines[start].strip()
            title = first[:60] + ("..." if len(first) > 60 else "")
            content = "\n".join(lines[start:end]).strip()
            node = {
                "title": title,
                "level": 1,
                "content": content,
                "children": [],
                "start_line": start,
                "end_line": end,
            }
            root["children"].append(node)

    return root


def _tree_to_toc(node: dict, indent: int = 0) -> str:
    """Convert tree to a table-of-contents string for LLM reasoning."""
    lines = []
    if node["level"] > 0:
        prefix = "  " * (node["level"] - 1)
        summary = node["content"][:120].replace("\n", " ") if node["content"] else ""
        lines.append(f"{prefix}[{node['level']}] {node['title']} — {summary}...")
    for child in node["children"]:
        lines.extend([_tree_to_toc(child, indent + 1)])
    return "\n".join(lines)


def _get_section_content(node: dict, doc_lines: list[str]) -> str:
    """Get full text content of a section including all its children."""
    return "\n".join(doc_lines[node["start_line"]:node["end_line"]])


def _find_node_by_title(node: dict, title: str) -> dict | None:
    """Find a node by its title (case-insensitive partial match)."""
    if title.lower() in node["title"].lower():
        return node
    for child in node["children"]:
        found = _find_node_by_title(child, title)
        if found:
            return found
    return None


def _get_all_sections(node: dict, sections: list | None = None) -> list[dict]:
    """Flatten tree into a list of all sections."""
    if sections is None:
        sections = []
    if node["level"] > 0:
        sections.append(node)
    for child in node["children"]:
        _get_all_sections(child, sections)
    return sections


class VectorlessRAG:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini"):
        self.openai_client = openai_client
        self.model = model
        self.tree = None
        self.doc_lines = []
        self.toc = ""

    def index_document(self, text: str) -> dict:
        """Build the hierarchical tree index (no embeddings)."""
        self.doc_lines = text.split("\n")
        self.tree = _parse_markdown_tree(text)
        all_sections = _get_all_sections(self.tree)

        # Fallback: if markdown parsing found nothing, try plain-text heuristics
        if not all_sections:
            self.tree = _parse_plaintext_tree(text)
            all_sections = _get_all_sections(self.tree)

        self.toc = _tree_to_toc(self.tree)

        return {
            "total_sections": len(all_sections),
            "tree_depth": max((s["level"] for s in all_sections), default=0),
            "index_type": "Hierarchical Tree (no vectors)",
        }

    def retrieve(self, query: str) -> list[dict]:
        """Use LLM reasoning to navigate the tree and find relevant sections."""
        # Step 1: LLM picks the most relevant sections from the TOC
        nav_response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document navigation expert. Given a table of contents "
                        "and a question, identify the 1-3 most relevant section titles "
                        "that would contain the answer. Respond with ONLY a JSON array "
                        "of section titles, e.g. [\"Section Title 1\", \"Section Title 2\"]. "
                        "Pick the most SPECIFIC sections, not broad parent sections."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Table of Contents:\n{self.toc}\n\n"
                        f"Question: {query}\n\n"
                        "Which sections are most relevant? Return JSON array of titles only."
                    ),
                },
            ],
            temperature=0.0,
        )
        self._last_nav_usage = nav_response.usage

        # Parse the LLM's section picks
        raw = nav_response.choices[0].message.content.strip()
        # Extract JSON array from response
        try:
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                section_titles = json.loads(json_match.group())
            else:
                section_titles = [raw]
        except json.JSONDecodeError:
            section_titles = [raw]

        # Step 2: Retrieve content of the identified sections
        retrieved = []
        for title in section_titles:
            node = _find_node_by_title(self.tree, title)
            if node:
                content = _get_section_content(node, self.doc_lines)
                retrieved.append({
                    "text": content,
                    "section_title": node["title"],
                    "reasoning": f"LLM navigated tree → selected '{node['title']}'",
                })

        # Fallback: if nothing matched, try broader search
        if not retrieved:
            all_sections = _get_all_sections(self.tree)
            for section in all_sections:
                if any(word.lower() in section["title"].lower()
                       for word in query.split() if len(word) > 3):
                    content = _get_section_content(section, self.doc_lines)
                    retrieved.append({
                        "text": content,
                        "section_title": section["title"],
                        "reasoning": f"Keyword fallback → matched '{section['title']}'",
                    })

        return retrieved[:3]

    def query(self, question: str) -> dict:
        """Full Vectorless RAG pipeline: reason through tree then generate."""
        self._last_nav_usage = None
        retrieved = self.retrieve(question)
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

        nav_usage = self._last_nav_usage
        gen_usage = response.usage
        tokens = {
            "prompt_tokens": (nav_usage.prompt_tokens if nav_usage else 0) + (gen_usage.prompt_tokens if gen_usage else 0),
            "completion_tokens": (nav_usage.completion_tokens if nav_usage else 0) + (gen_usage.completion_tokens if gen_usage else 0),
            "total_tokens": 0,
            "llm_calls": 2,
        }
        tokens["total_tokens"] = tokens["prompt_tokens"] + tokens["completion_tokens"]

        return {
            "answer": response.choices[0].message.content,
            "tokens": tokens,
            "retrieved_chunks": [
                {
                    "text": r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"],
                    "section": r["section_title"],
                    "reasoning": r["reasoning"],
                }
                for r in retrieved
            ],
            "method": "Vectorless RAG",
            "details": {
                "tree_sections": len(_get_all_sections(self.tree)),
                "sections_retrieved": len(retrieved),
                "retrieval_type": "LLM reasoning over tree index",
            },
        }
