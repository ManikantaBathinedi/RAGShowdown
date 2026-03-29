"""
RAG Showdown — Vector vs Vectorless vs Hybrid
==============================================
Upload any document. Compare 3 RAG approaches side-by-side.
Get metrics, cost estimates, and a recommendation for which works best.
"""

import csv
import io
import json
import os
import time
from datetime import datetime

from fpdf import FPDF

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

from vector_rag import VectorRAG
from vectorless_rag import VectorlessRAG
from hybrid_rag import HybridRAG
from doc_analyzer import analyze_document
from benchmark import estimate_cost, generate_benchmark_questions, run_benchmark

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Showdown: Vector vs Vectorless vs Hybrid",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS for clean professional look ───────────────────────────────────
st.markdown("""
<style>
    /* Tighter spacing */
    .block-container { padding-top: 2rem; }
    /* Card styling */
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .engine-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        padding: 6px 0;
    }
    .vector-border { border-left: 4px solid #4287f5; }
    .vectorless-border { border-left: 4px solid #42f56f; }
    .hybrid-border { border-left: 4px solid #f5a442; }
    .retrieval-card {
        padding: 10px 14px;
        margin-bottom: 6px;
        border-radius: 6px;
        background: rgba(255,255,255,0.03);
    }
    .winner-banner {
        padding: 12px 20px;
        border-radius: 8px;
        border: 1px solid rgba(66,135,245,0.3);
        background: rgba(66,135,245,0.05);
        margin-bottom: 12px;
    }
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

LABELS = [("vector", "🟦 Vector RAG"), ("vectorless", "🟩 Vectorless RAG"), ("hybrid", "🟨 Hybrid RAG")]
COLORS = {"vector": "#4287f5", "vectorless": "#42f56f", "hybrid": "#f5a442"}
ENGINE_NAMES = {"vector": "Vector RAG", "vectorless": "Vectorless RAG", "hybrid": "Hybrid RAG"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if name.endswith((".md", ".txt")):
        return raw.decode("utf-8", errors="replace")
    elif name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            st.error("Install PyPDF2: `pip install PyPDF2`")
            return ""
    elif name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            st.error("Install python-docx: `pip install python-docx`")
            return ""
    return raw.decode("utf-8", errors="replace")


def _create_client(provider_name, **kwargs):
    if provider_name == "Azure OpenAI":
        return AzureOpenAI(
            azure_endpoint=kwargs["azure_endpoint"],
            api_key=kwargs["api_key"],
            api_version=kwargs["api_version"],
        )
    elif provider_name == "Ollama (Local, Free)":
        return OpenAI(base_url=kwargs["ollama_url"], api_key="ollama")
    return OpenAI(api_key=kwargs["api_key"])


def _engine_cache_key(provider_name, model_name, doc_key, chunk_size, chunk_overlap):
    return f"{provider_name}|{model_name}|{doc_key}|{chunk_size}|{chunk_overlap}"


def _engines_cached(cache_key):
    if st.session_state.get("_engine_key") == cache_key and st.session_state.get("engines"):
        try:
            st.session_state["engines"]["vector"].collection.count()
            return True
        except Exception:
            pass
    return False


def init_engines(provider_name, model_name, doc_text, doc_key, chunk_size=400, chunk_overlap=80, **kwargs):
    cache_key = _engine_cache_key(provider_name, model_name, doc_key, chunk_size, chunk_overlap)
    if _engines_cached(cache_key):
        return st.session_state["engines"], st.session_state["index_info"], st.session_state["_client"]

    client = _create_client(provider_name, **kwargs)
    vec = VectorRAG(client, model_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vless = VectorlessRAG(client, model_name)
    hyb = HybridRAG(client, model_name, vector_rag=vec, vectorless_rag=vless)
    engines = {"vector": vec, "vectorless": vless, "hybrid": hyb}
    idx = {
        "vector": vec.index_document(doc_text),
        "vectorless": vless.index_document(doc_text),
        "hybrid": hyb.index_document(doc_text),
    }
    st.session_state.update({"engines": engines, "index_info": idx, "_engine_key": cache_key, "_client": client})
    return engines, idx, client


def build_single_query_csv(question, results):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Engine", "Answer", "Latency (s)", "Tokens", "LLM Calls", "Cost (USD)"])
    for ename in ["vector", "vectorless", "hybrid"]:
        r = results.get(ename, {})
        if "error" in r:
            w.writerow([ENGINE_NAMES[ename], f"Error: {r['error']}", f"{r['time']:.2f}", "", "", ""])
        else:
            t = r.get("tokens", {})
            w.writerow([
                ENGINE_NAMES[ename], r["answer"], f"{r['time']:.2f}",
                t.get("total_tokens", 0), t.get("llm_calls", 0), f"{r.get('cost', 0):.6f}",
            ])
    return buf.getvalue()


def build_single_query_json(question, results, analysis):
    export = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "document_analysis": {
            "words": analysis["total_words"],
            "headings": analysis["heading_count"],
            "structure_score": analysis["structure_score"],
            "recommendation": analysis["recommendation"],
        },
        "results": {},
    }
    for ename in ["vector", "vectorless", "hybrid"]:
        r = results.get(ename, {})
        if "error" in r:
            export["results"][ename] = {"error": r["error"]}
        else:
            export["results"][ename] = {
                "answer": r["answer"],
                "latency_sec": round(r["time"], 3),
                "tokens": r.get("tokens", {}),
                "cost_usd": r.get("cost", 0),
                "chunks_retrieved": len(r.get("retrieved_chunks", [])),
            }
    return json.dumps(export, indent=2)


def build_benchmark_csv(summary, details, questions):
    buf = io.StringIO()
    w = csv.writer(buf)
    # Summary
    w.writerow(["=== SUMMARY ==="])
    w.writerow(["Metric", "Vector RAG", "Vectorless RAG", "Hybrid RAG"])
    metrics = [
        ("Avg Latency (s)", "avg_time_sec"), ("Avg LLM Calls", "avg_llm_calls"),
        ("Avg Tokens/Query", "avg_tokens_per_query"), ("Total Tokens", "total_tokens"),
        ("Total Cost (USD)", "total_cost_usd"), ("Cost/1K Queries", "cost_per_1000_queries"),
        ("Success Rate", None),
    ]
    for label, key in metrics:
        if key:
            w.writerow([label] + [summary[e][key] for e in ["vector", "vectorless", "hybrid"]])
        else:
            w.writerow([label] + [f"{summary[e]['successful']}/{summary[e]['total_questions']}" for e in ["vector", "vectorless", "hybrid"]])
    w.writerow([])
    # Details
    w.writerow(["=== PER-QUESTION DETAILS ==="])
    w.writerow(["Question", "Engine", "Answer", "Latency (s)", "Tokens", "Cost (USD)", "Success"])
    for q_idx, q in enumerate(questions):
        for ename in ["vector", "vectorless", "hybrid"]:
            r = details[ename][q_idx]
            w.writerow([q, ENGINE_NAMES[ename], r["answer"][:500], r["time"], r["tokens"].get("total_tokens", 0), r["cost"], r["success"]])
    return buf.getvalue()


def _safe(text):
    """Strip characters that latin-1 (fpdf default) can't encode."""
    if not text:
        return ""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_single_query_pdf(question, results, analysis, doc_name=""):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(66, 135, 245)
    pdf.cell(0, 12, "RAG Comparison Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 6, f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}  |  Document: {_safe(doc_name)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Document overview ──
    pdf.set_draw_color(200, 200, 200)
    pdf.set_fill_color(245, 245, 250)
    pdf.rect(10, pdf.get_y(), 190, 28, style="DF")
    y0 = pdf.get_y() + 3
    pdf.set_xy(14, y0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 5, "Document Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(45, 5, f"Words: {analysis['total_words']:,}")
    pdf.cell(45, 5, f"Headings: {analysis['heading_count']}")
    pdf.cell(50, 5, f"Structure Score: {analysis['structure_score']}/100")
    pdf.cell(0, 5, f"Depth: {analysis['max_depth']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(14)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(66, 135, 245)
    pdf.cell(0, 5, f"Recommended: {analysis['recommendation']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(y0 + 28)
    pdf.ln(4)

    # ── Question ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, "Question", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 5, _safe(question))
    pdf.ln(4)

    # ── Comparison table ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, "Performance Comparison", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    col_w = [45, 30, 30, 30, 35]
    headers = ["Engine", "Latency", "Tokens", "LLM Calls", "Cost (USD)"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(66, 135, 245)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    colors = {"vector": (66, 135, 245), "vectorless": (34, 180, 85), "hybrid": (245, 164, 66)}
    pdf.set_font("Helvetica", "", 9)
    for ename in ["vector", "vectorless", "hybrid"]:
        r = results.get(ename, {})
        pdf.set_text_color(*colors[ename])
        pdf.cell(col_w[0], 6, ENGINE_NAMES[ename], border=1, align="C")
        pdf.set_text_color(60, 60, 60)
        if "error" in r:
            pdf.cell(sum(col_w[1:]), 6, f"Error: {_safe(r['error'])[:60]}", border=1)
        else:
            t = r.get("tokens", {})
            pdf.cell(col_w[1], 6, f"{r['time']:.2f}s", border=1, align="C")
            pdf.cell(col_w[2], 6, f"{t.get('total_tokens', 0):,}", border=1, align="C")
            pdf.cell(col_w[3], 6, str(t.get("llm_calls", "?")), border=1, align="C")
            pdf.cell(col_w[4], 6, f"${r.get('cost', 0):.6f}", border=1, align="C")
        pdf.ln()
    pdf.ln(6)

    # ── Answers ──
    for ename in ["vector", "vectorless", "hybrid"]:
        r = results.get(ename, {})
        if "error" in r:
            continue
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*colors[ename])
        pdf.cell(0, 7, ENGINE_NAMES[ename], new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 4.5, _safe(r["answer"]))
        pdf.ln(4)

    # ── Footer ──
    pdf.set_y(-25)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 5, "RAG Showdown - Vector vs Vectorless vs Hybrid", align="C")

    return bytes(pdf.output())


def build_benchmark_pdf(summary, details, questions, doc_name=""):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(66, 135, 245)
    pdf.cell(0, 12, "RAG Benchmark Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 6, f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}  |  {len(questions)} questions", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Document: {_safe(doc_name)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Summary table ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, "Aggregate Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    col_w = [50, 40, 50, 40]
    headers = ["Metric", "Vector", "Vectorless", "Hybrid"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(66, 135, 245)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    rows = [
        ("Avg Latency", lambda e: f"{summary[e]['avg_time_sec']:.2f}s"),
        ("LLM Calls/Query", lambda e: str(summary[e]["avg_llm_calls"])),
        ("Tokens/Query", lambda e: f"{summary[e]['avg_tokens_per_query']:,}"),
        ("Total Tokens", lambda e: f"{summary[e]['total_tokens']:,}"),
        ("Total Cost", lambda e: f"${summary[e]['total_cost_usd']:.4f}"),
        ("Cost/1K Queries", lambda e: f"${summary[e]['cost_per_1000_queries']:.4f}"),
        ("Success Rate", lambda e: f"{summary[e]['successful']}/{summary[e]['total_questions']}"),
    ]

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    alt = False
    for label, fn in rows:
        if alt:
            pdf.set_fill_color(245, 245, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_w[0], 6, label, border=1, fill=True)
        for ename in ["vector", "vectorless", "hybrid"]:
            pdf.cell(col_w[1] if ename == "vector" else (col_w[2] if ename == "vectorless" else col_w[3]),
                     6, fn(ename), border=1, fill=True, align="C")
        pdf.ln()
        alt = not alt
    pdf.ln(6)

    # ── Per-question detail ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, "Per-Question Results", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    colors = {"vector": (66, 135, 245), "vectorless": (34, 180, 85), "hybrid": (245, 164, 66)}
    short = {"vector": "Vector", "vectorless": "Vectorless", "hybrid": "Hybrid"}
    for qi, q in enumerate(questions):
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, _safe(f"Q{qi+1}. {q}"))
        pdf.ln(1)
        for ename in ["vector", "vectorless", "hybrid"]:
            r = details[ename][qi]
            tok = r["tokens"].get("total_tokens", 0)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*colors[ename])
            pdf.cell(22, 4, short[ename], new_x="END")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, f"{r['time']}s  |  {tok} tok  |  ${r['cost']:.6f}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 3.5, _safe("   " + r["answer"][:200]))
            pdf.ln(0.5)
        pdf.ln(3)

    # ── Footer ──
    pdf.set_y(-25)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 5, "RAG Showdown - Benchmark Report", align="C")

    return bytes(pdf.output())


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("LLM Provider", ["Azure OpenAI", "OpenAI", "Ollama (Local, Free)"])

    # Use .env credentials silently if available; otherwise show BYOK inputs
    _env_azure_ep = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    _env_azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    _env_openai_key = os.getenv("OPENAI_API_KEY", "")

    if provider == "Azure OpenAI":
        _server_has_azure = bool(_env_azure_ep and _env_azure_key)
        use_own_azure = False
        if _server_has_azure:
            use_own_azure = st.toggle("Use my own Azure credentials", value=False)
        if _server_has_azure and not use_own_azure:
            azure_endpoint = _env_azure_ep
            api_key = _env_azure_key
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            st.success("Connected via server config", icon="🔒")
        else:
            azure_endpoint = st.text_input("Azure Endpoint")
            api_key = st.text_input("Azure API Key", type="password")
            model = st.text_input("Deployment Name", value="gpt-4o-mini")
            api_version = st.text_input("API Version", value="2024-12-01-preview")
    elif provider == "OpenAI":
        _server_has_openai = bool(_env_openai_key)
        use_own_openai = False
        if _server_has_openai:
            use_own_openai = st.toggle("Use my own OpenAI key", value=False)
        if _server_has_openai and not use_own_openai:
            api_key = _env_openai_key
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
            st.success("Connected via server config", icon="🔒")
        else:
            api_key = st.text_input("OpenAI API Key", type="password")
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434/v1")
        model = st.text_input("Ollama Model", value="llama3.2")
        api_key = "ollama"

    st.divider()
    st.header("📄 Document")
    doc_source = st.radio("Source", ["Sample (Acme Corp Handbook)", "Upload your own"])
    uploaded_file = None
    if doc_source == "Upload your own":
        uploaded_file = st.file_uploader("Upload", type=["md", "txt", "pdf", "docx"])

    st.divider()
    st.header("🎯 Mode")
    app_mode = st.radio("Choose mode", ["Single Query", "Benchmark (Multi-Query)"])

    st.divider()
    with st.expander("⚙️ Advanced Settings"):
        # Initialize applied values in session state
        if "_applied_chunk_size" not in st.session_state:
            st.session_state["_applied_chunk_size"] = 400
        if "_applied_chunk_overlap" not in st.session_state:
            st.session_state["_applied_chunk_overlap"] = 80

        new_chunk_size = st.slider("Chunk size (words)", 100, 1000,
                                   st.session_state["_applied_chunk_size"], step=50,
                                   help="Larger chunks = more context per retrieval but fewer chunks. Smaller = more precise but may miss context.")
        new_chunk_overlap = st.slider("Chunk overlap (words)", 0, 200,
                                      st.session_state["_applied_chunk_overlap"], step=10,
                                      help="Overlap between consecutive chunks to avoid cutting context at boundaries.")

        has_changes = (new_chunk_size != st.session_state["_applied_chunk_size"] or
                       new_chunk_overlap != st.session_state["_applied_chunk_overlap"])

        if st.button("Apply", disabled=not has_changes, use_container_width=True, type="primary"):
            st.session_state["_applied_chunk_size"] = new_chunk_size
            st.session_state["_applied_chunk_overlap"] = new_chunk_overlap
            # Clear engine cache to force re-index
            st.session_state.pop("_engine_key", None)
            st.rerun()

        if has_changes:
            st.caption(f"Pending: {new_chunk_size}w chunks, {new_chunk_overlap}w overlap → click Apply")
        else:
            st.caption("Affects Vector & Hybrid RAG only. Vectorless uses heading-based sections.")

    st.divider()
    st.caption(
        "**Vector**: Chunks → embeddings → cosine similarity\n\n"
        "**Vectorless**: Headings → tree → LLM reasoning\n\n"
        "**Hybrid**: Both + LLM reranker"
    )

    st.divider()
    with st.expander("ℹ️ About & Disclaimer", expanded=False):
        st.markdown("""
**Learning & comparison tool** — not a production RAG system.

**Key Limitations:**
- 📄 Basic text extraction (PyPDF2/python-docx) — production uses Azure Document Intelligence, LlamaParse
- 💾 In-memory vector store (lost on restart) — production uses Pinecone, Azure AI Search, pgvector
- ✂️ Simple word-count chunking — production uses semantic/recursive chunking
- 🔢 Basic embeddings (MiniLM, 384d) — production uses text-embedding-3-large or fine-tuned models
- 🌳 Heuristic heading detection — production uses LLM-generated summaries (RAPTOR)
- ❌ No reranking — production adds Cohere Rerank or cross-encoders
- ❌ No guardrails — production adds hallucination detection & citation verification

**Accuracy:**
- Vector: ~70–80% (production: 85–95% with reranking)
- Vectorless: ~60–85% (depends on doc structure)
- Hybrid: Generally best, but our merge is basic

*Architecture is sound — gaps are in component quality, not design.*
""")

    st.markdown(
        "<div style='text-align:center; padding:8px 0; opacity:0.7;'>"
        "<small>Crafted with ❤️ by <b>Manime</b></small><br>"
        "<a href='https://github.com/ManikantaBathinedi/RAGShowdown' target='_blank' style='color:#4287f5; text-decoration:none;'>"
        "<small>⭐ GitHub</small></a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
st.title("🔍 RAG Showdown")
st.markdown("**Vector RAG** vs **Vectorless RAG** vs **Hybrid RAG**")
st.caption("Same document, same question — three retrieval strategies compared with real metrics.")

# Credentials check
if provider == "Azure OpenAI" and (not api_key or not azure_endpoint):
    st.warning("👈 Enter Azure endpoint and API key.")
    st.stop()
elif provider == "OpenAI" and not api_key:
    st.warning("👈 Enter your OpenAI API key.")
    st.stop()

# Load document
if doc_source == "Upload your own":
    if not uploaded_file:
        st.info("👈 Upload a document to get started.")
        st.stop()
    doc_text = extract_text(uploaded_file)
    doc_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
    if not doc_text.strip():
        st.error("Could not extract text from file.")
        st.stop()
    doc_name = uploaded_file.name
else:
    with open(os.path.join(os.path.dirname(__file__), "sample_document.md"), "r", encoding="utf-8") as f:
        doc_text = f.read()
    doc_key = "sample_acme"
    doc_name = "Acme Corp Employee Handbook (sample)"

client_kwargs = {"api_key": api_key}
if provider == "Azure OpenAI":
    client_kwargs["azure_endpoint"] = azure_endpoint
    client_kwargs["api_version"] = api_version
elif provider == "Ollama (Local, Free)":
    client_kwargs["ollama_url"] = ollama_url

# Clear stale results on doc change
if st.session_state.get("_last_doc_key") != doc_key:
    for key in ["bench_results", "bench_questions", "user_question", "query_results"]:
        st.session_state.pop(key, None)
    st.session_state["_last_doc_key"] = doc_key


# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
analysis = analyze_document(doc_text)

with st.expander("📋 Document Analysis & Recommendation", expanded=True):
    left, right = st.columns([3, 2])

    with left:
        best = analysis["recommendation"]
        color_map = {"Vector RAG": "🟦", "Vectorless RAG": "🟩", "Hybrid RAG": "🟨"}

        # Live verdict if we have query results
        prev = st.session_state.get("query_results")
        if prev and all(e in prev and "error" not in prev[e] for e in ["vector", "vectorless", "hybrid"]):
            times = {e: prev[e]["time"] for e in ["vector", "vectorless", "hybrid"]}
            costs = {e: prev[e].get("cost", 0) for e in ["vector", "vectorless", "hybrid"]}
            toks = {e: prev[e].get("tokens", {}).get("total_tokens", 0) for e in ["vector", "vectorless", "hybrid"]}
            mx_t, mx_c, mx_k = max(times.values()) or 1, max(costs.values()) or 1, max(toks.values()) or 1
            scores = {e: (costs[e]/mx_c)*0.4 + (times[e]/mx_t)*0.3 + (toks[e]/mx_k)*0.3 for e in times}
            winner = min(scores, key=scores.get)

            st.markdown(f"### {color_map.get(ENGINE_NAMES[winner], '🔍')} Winner: **{ENGINE_NAMES[winner]}**")
            st.caption("Based on this query · Run Benchmark for a multi-query verdict")
            st.markdown(
                f"<div class='winner-banner'>"
                f"Fastest at <b>{times[winner]:.2f}s</b>, "
                f"used <b>{toks[winner]:,}</b> tokens, "
                f"cost <b>${costs[winner]:.5f}</b> per query."
                f"</div>",
                unsafe_allow_html=True,
            )

            for e, lbl in LABELS:
                bar = max(0.05, 1.0 - scores[e])
                medal = " 👑" if e == winner else ""
                st.markdown(f"**{lbl}{medal}** — {times[e]:.2f}s · {toks[e]:,} tokens · ${costs[e]:.5f}")
                st.progress(bar)
            st.caption(f"Structure prediction: {best} · Score: {analysis['structure_score']}/100")
        else:
            st.markdown(f"### {color_map.get(best, '🔍')} Predicted: **{best}**")
            st.info(analysis["recommendation_reason"])
            st.caption(f"Runner-up: {analysis['runner_up']} · Ask a question for a live verdict.")

    with right:
        score = analysis["structure_score"]
        level = "Well-structured" if score >= 60 else "Moderate" if score >= 30 else "Low structure"
        st.markdown(f"**Structure Score: {score}/100** ({level})")
        st.progress(min(score, 100) / 100)

        c1, c2 = st.columns(2)
        c1.metric("Words", f"{analysis['total_words']:,}")
        c2.metric("Headings", analysis["heading_count"])
        c3, c4 = st.columns(2)
        c3.metric("Depth", analysis["max_depth"])
        c4.metric("Avg Section", f"{analysis['avg_section_words']}w")


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE INIT
# ═════════════════════════════════════════════════════════════════════════════
_ck = _engine_cache_key(provider, model, doc_key,
                        st.session_state["_applied_chunk_size"],
                        st.session_state["_applied_chunk_overlap"])

if _engines_cached(_ck):
    engines, index_info, client = init_engines(provider, model, doc_text, doc_key,
                                               chunk_size=st.session_state["_applied_chunk_size"],
                                               chunk_overlap=st.session_state["_applied_chunk_overlap"],
                                               **client_kwargs)
else:
    with st.status("🔄 Preparing engines...", expanded=True) as status:
        try:
            st.write("Loading embedding model...")
            engines, index_info, client = init_engines(provider, model, doc_text, doc_key,
                                                       chunk_size=st.session_state["_applied_chunk_size"],
                                                       chunk_overlap=st.session_state["_applied_chunk_overlap"],
                                                       **client_kwargs)
            st.write(f"Indexed — chunk size: {st.session_state['_applied_chunk_size']}w, overlap: {st.session_state['_applied_chunk_overlap']}w")
            status.update(label="✅ Engines ready", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Init failed", state="error")
            st.error(f"Init failed: {e}")
            st.stop()

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# HOW EACH ENGINE SEES THE DOCUMENT
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("🔬 How Each Engine Sees the Document", expanded=False):
    vtab, vltab = st.tabs(["🟦 Vector RAG — Chunks & Embeddings", "🟩 Vectorless RAG — Heading Tree"])

    # ── Vector RAG: chunk visualization ──────────────────────────────────────
    with vtab:
        chunks = engines["vector"].chunks
        cs = st.session_state["_applied_chunk_size"]
        co = st.session_state["_applied_chunk_overlap"]
        st.markdown(f"**{len(chunks)} chunks** created with **{cs}w** chunk size and **{co}w** overlap")
        st.caption(f"Embedding model: all-MiniLM-L6-v2 (384 dimensions) · Stored in ChromaDB (cosine similarity)")

        # Chunk size distribution bar
        word_counts = [len(c["text"].split()) for c in chunks]
        st.markdown("**Chunk sizes (words):**")
        chart_cols = st.columns(min(len(chunks), 20))
        max_wc = max(word_counts) if word_counts else 1
        for i, (col, wc) in enumerate(zip(chart_cols, word_counts[:20])):
            with col:
                st.progress(wc / max_wc)
                st.caption(f"{wc}w")
        if len(chunks) > 20:
            st.caption(f"... and {len(chunks) - 20} more chunks")

        st.markdown("---")
        st.markdown("**Chunk contents:**")
        chunk_labels = [f"Chunk {i+1} — {len(c['text'].split())}w" for i, c in enumerate(chunks)]
        selected_chunk = st.selectbox("Select chunk to inspect:", chunk_labels, key="chunk_sel")
        ci = chunk_labels.index(selected_chunk)
        st.code(chunks[ci]["text"], language="markdown")

        # Show embedding space (2D projection of first few chunks)
        st.markdown("---")
        st.markdown("**Embedding space (2D projection):**")
        st.caption("Each dot is a chunk. Closer dots = more similar content.")
        try:
            import numpy as np
            embeddings = engines["vector"].embedder.encode([c["text"] for c in chunks])
            # Simple 2D projection using the first 2 principal components
            centered = embeddings - embeddings.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            top2 = eigenvectors[:, -2:]
            projected = centered @ top2
            import pandas as pd
            df = pd.DataFrame({
                "x": projected[:, 0],
                "y": projected[:, 1],
                "chunk": [f"Chunk {i+1}: {c['text'][:60]}..." for i, c in enumerate(chunks)],
            })
            st.scatter_chart(df, x="x", y="y", height=300)
        except Exception:
            st.caption("Could not render embedding projection.")

    # ── Vectorless RAG: tree visualization ───────────────────────────────────
    with vltab:
        tree = engines["vectorless"].tree
        toc = engines["vectorless"].toc

        def count_nodes(node):
            return 1 + sum(count_nodes(c) for c in node.get("children", []))

        def max_depth(node, depth=0):
            if not node.get("children"):
                return depth
            return max(max_depth(c, depth + 1) for c in node["children"])

        total_sections = count_nodes(tree) - 1
        depth = max_depth(tree)
        st.markdown(f"**{total_sections} sections** across **{depth} levels** of nesting")
        st.caption("The LLM reads this table of contents, then reasons about which section(s) to retrieve — no embeddings needed.")

        # Build Graphviz tree
        st.markdown("**Document tree:**")
        level_colors = {1: "#42f56f", 2: "#3dd4a0", 3: "#38b5d1", 4: "#6b8ef5", 5: "#a97bf5", 6: "#d46bf5"}

        def _sanitize(text):
            return text.replace('"', '\\"').replace('\n', ' ')[:50]

        dot_lines = [
            'digraph {',
            '  rankdir=TB;',
            '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.07"];',
            '  edge [color="#666666", arrowsize=0.6];',
            '  bgcolor="transparent";',
            f'  root [label="📄 Document", fillcolor="#1a1a2e", fontcolor="white", fontsize=11];',
        ]
        node_id = [0]

        def add_nodes(node, parent_id=None):
            for child in node.get("children", []):
                nid = f"n{node_id[0]}"
                node_id[0] += 1
                words = len(child["content"].split()) if child["content"] else 0
                color = level_colors.get(child["level"], "#888888")
                label = _sanitize(child["title"])
                dot_lines.append(f'  {nid} [label="{label}\\n{words}w", fillcolor="{color}", fontcolor="#1a1a2e"];')
                pid = parent_id if parent_id else "root"
                dot_lines.append(f'  {pid} -> {nid};')
                add_nodes(child, nid)

        add_nodes(tree)
        dot_lines.append('}')
        dot_src = '\n'.join(dot_lines)

        try:
            st.graphviz_chart(dot_src, use_container_width=True)
        except Exception:
            # Fallback to text tree if graphviz not available
            def render_tree(node, indent=0):
                if node["level"] > 0:
                    prefix = "│   " * (indent - 1) + "├── " if indent > 0 else ""
                    words = len(node["content"].split()) if node["content"] else 0
                    st.markdown(f"`{prefix}{'#' * node['level']} {node['title']}` — *{words}w*")
                for child in node.get("children", []):
                    render_tree(child, indent + 1)
            render_tree(tree)

        st.markdown("---")
        st.markdown("**Raw table of contents (what the LLM sees):**")
        st.code(toc, language="text")

        st.markdown("---")
        st.markdown("**Section contents:**")
        def collect_sections(node, out=None):
            if out is None:
                out = []
            if node["level"] > 0 and node["content"].strip():
                out.append(node)
            for child in node.get("children", []):
                collect_sections(child, out)
            return out

        sections = collect_sections(tree)
        if sections:
            sec_labels = [f"{'#' * s['level']} {s['title']} — {len(s['content'].split())}w" for s in sections]
            selected_sec = st.selectbox("Select section to inspect:", sec_labels, key="sec_sel")
            si = sec_labels.index(selected_sec)
            st.code(sections[si]["content"][:800], language="markdown")
        else:
            st.caption("No sections with content found.")

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK MODE
# ═════════════════════════════════════════════════════════════════════════════
if app_mode == "Benchmark (Multi-Query)":
    st.subheader("🏋️ Benchmark")

    b1, b2 = st.columns([2, 1])
    with b1:
        num_q = st.slider("Questions", 3, 10, 5)
    with b2:
        auto_gen = st.checkbox("Auto-generate", value=True)

    if auto_gen:
        if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
            with st.spinner("Generating questions..."):
                questions = generate_benchmark_questions(client, model, doc_text, num_q)
            bar = st.progress(0, text="Running...")
            res = run_benchmark(engines, questions, model, progress_callback=lambda p, t: bar.progress(p, text=t))
            bar.empty()
            st.session_state["bench_results"] = res
            st.session_state["bench_questions"] = questions
    else:
        custom = st.text_area("Questions (one per line)", height=120)
        if st.button("🚀 Run Benchmark", type="primary", use_container_width=True) and custom.strip():
            questions = [q.strip() for q in custom.strip().split("\n") if q.strip()]
            bar = st.progress(0, text="Running...")
            res = run_benchmark(engines, questions, model, progress_callback=lambda p, t: bar.progress(p, text=t))
            bar.empty()
            st.session_state["bench_results"] = res
            st.session_state["bench_questions"] = questions

    if "bench_results" not in st.session_state:
        st.stop()

    bench = st.session_state["bench_results"]
    summary = bench["summary"]
    questions = st.session_state["bench_questions"]

    st.divider()

    # ── Benchmark summary table ──────────────────────────────────────────────
    st.markdown("#### 📊 Results Summary")

    # Build a clean comparison table with HTML
    table_rows = [
        ("Avg Latency", [f"{summary[e]['avg_time_sec']:.2f}s" for e in ["vector", "vectorless", "hybrid"]]),
        ("LLM Calls / Query", [f"{summary[e]['avg_llm_calls']}" for e in ["vector", "vectorless", "hybrid"]]),
        ("Tokens / Query", [f"{summary[e]['avg_tokens_per_query']:,}" for e in ["vector", "vectorless", "hybrid"]]),
        ("Total Tokens", [f"{summary[e]['total_tokens']:,}" for e in ["vector", "vectorless", "hybrid"]]),
        ("Total Cost", [f"${summary[e]['total_cost_usd']:.4f}" for e in ["vector", "vectorless", "hybrid"]]),
        ("Cost / 1K Queries", [f"${summary[e]['cost_per_1000_queries']:.4f}" for e in ["vector", "vectorless", "hybrid"]]),
        ("Success", [f"{summary[e]['successful']}/{summary[e]['total_questions']}" for e in ["vector", "vectorless", "hybrid"]]),
    ]

    html = """<table style='width:100%; border-collapse:collapse; font-size:0.9rem;'>
    <thead><tr style='border-bottom:2px solid rgba(255,255,255,0.15);'>
        <th style='text-align:left; padding:8px;'>Metric</th>
        <th style='text-align:center; padding:8px; color:#4287f5;'>🟦 Vector</th>
        <th style='text-align:center; padding:8px; color:#42f56f;'>🟩 Vectorless</th>
        <th style='text-align:center; padding:8px; color:#f5a442;'>🟨 Hybrid</th>
    </tr></thead><tbody>"""
    for label, vals in table_rows:
        html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
        html += f"<td style='padding:6px 8px; font-weight:500;'>{label}</td>"
        for v in vals:
            html += f"<td style='padding:6px 8px; text-align:center;'>{v}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

    # ── Cost projection ──────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("#### 💰 Cost Projection")
    daily = st.slider("Daily queries", 10, 10000, 100, step=10, key="bench_daily")

    pc = st.columns(3)
    for col, (ename, elabel) in zip(pc, LABELS):
        with col:
            avg = summary[ename]["avg_cost_per_query"]
            st.markdown(f"**{elabel}**")
            st.metric("Monthly (30d)", f"${avg * daily * 30:.2f}")
            st.caption(f"${avg:.6f}/query · ${avg * daily:.4f}/day")

    # ── Per-question ─────────────────────────────────────────────────────────
    with st.expander("📝 Per-Question Detail"):
        for qi, q in enumerate(questions):
            st.markdown(f"**Q{qi+1}.** {q}")
            dc = st.columns(3)
            for col, ename in zip(dc, ["vector", "vectorless", "hybrid"]):
                with col:
                    r = bench["details"][ename][qi]
                    if r["success"]:
                        st.caption(f"{r['time']}s · {r['tokens'].get('total_tokens', 0)} tok · ${r['cost']:.6f}")
                        st.success(r["answer"][:250])
                    else:
                        st.error(r["answer"][:150])
            if qi < len(questions) - 1:
                st.markdown("---")

    # ── Export benchmark ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Export Benchmark")
    ex1, ex2, ex3, _ = st.columns([1, 1, 1, 1])
    with ex1:
        csv_data = build_benchmark_csv(summary, bench["details"], questions)
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name=f"rag_benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        json_data = json.dumps({"summary": summary, "questions": questions, "timestamp": datetime.now().isoformat()}, indent=2)
        st.download_button(
            "📥 Download JSON",
            data=json_data,
            file_name=f"rag_benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with ex3:
        pdf_bytes = build_benchmark_pdf(summary, bench["details"], questions, doc_name)
        st.download_button(
            "📥 Download PDF",
            data=pdf_bytes,
            file_name=f"rag_benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE QUERY MODE
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("💬 Ask a Question")

if doc_source == "Sample (Acme Corp Handbook)":
    samples = [
        "How many PTO days does a 4-year employee get?",
        "What is the on-call compensation?",
        "What are the promotion criteria?",
        "What technology stack does the company use?",
        "How does the 401k matching work?",
        "What is the parental leave policy?",
        "What is the expense approval chain for a $3000 expense?",
        "When was the company founded and by whom?",
    ]
    cols = st.columns(4)
    for i, q in enumerate(samples):
        with cols[i % 4]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["user_question"] = q

question = st.text_input(
    "Your question:",
    value=st.session_state.get("user_question", ""),
    placeholder="e.g., How many PTO days does a 4-year employee get?",
    label_visibility="collapsed",
)

if not question:
    st.stop()

st.divider()

# ── Run engines ──────────────────────────────────────────────────────────────
results = {}
with st.status("⏳ Running all 3 engines...", expanded=True) as qstatus:
    for name, label in [("vector", "🟦 Vector RAG"), ("vectorless", "🟩 Vectorless RAG"), ("hybrid", "🟨 Hybrid RAG")]:
        st.write(f"Querying {label}...")
        start = time.time()
        try:
            result = engines[name].query(question)
            elapsed = time.time() - start
            tokens = result.get("tokens", {})
            cost = estimate_cost(tokens, model)
            results[name] = {**result, "time": elapsed, "cost": cost}
            st.write(f"{label} done — {elapsed:.1f}s")
        except Exception as e:
            results[name] = {"error": str(e), "time": time.time() - start}
            st.write(f"{label} error: {e}")
    qstatus.update(label="✅ All engines complete", state="complete", expanded=False)

st.session_state["query_results"] = results

# ── Find winner ──────────────────────────────────────────────────────────────
valid = {e: r for e, r in results.items() if "error" not in r}
winner = None
if len(valid) == 3:
    t = {e: r["time"] for e, r in valid.items()}
    c = {e: r.get("cost", 0) for e, r in valid.items()}
    k = {e: r.get("tokens", {}).get("total_tokens", 0) for e, r in valid.items()}
    mx_t, mx_c, mx_k = max(t.values()) or 1, max(c.values()) or 1, max(k.values()) or 1
    sc = {e: (c[e]/mx_c)*0.4 + (t[e]/mx_t)*0.3 + (k[e]/mx_k)*0.3 for e in t}
    winner = min(sc, key=sc.get)


# ── Answers ──────────────────────────────────────────────────────────────────
answer_cols = st.columns(3)
for col, (ename, elabel) in zip(answer_cols, LABELS):
    with col:
        r = results.get(ename, {})
        is_winner = ename == winner
        border = COLORS[ename]
        crown = " 👑" if is_winner else ""

        if "error" in r:
            st.markdown(f"**{elabel}**")
            st.error(r["error"])
            continue

        tok = r.get("tokens", {})
        st.markdown(
            f"<div class='metric-card' style='border-left: 4px solid {border};'>"
            f"<div class='engine-header'>{elabel}{crown}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Answer
        st.markdown(r["answer"])

        # Inline metrics
        st.caption(
            f"⏱️ {r['time']:.2f}s  ·  "
            f"🪙 {tok.get('total_tokens', 0):,} tokens  ·  "
            f"📞 {tok.get('llm_calls', '?')} LLM calls  ·  "
            f"💵 ${r.get('cost', 0):.5f}"
        )

st.divider()

# ── Retrieval Map ────────────────────────────────────────────────────────────
st.subheader("🗺️ What Each Engine Retrieved")

viz_cols = st.columns(3)
for col, (ename, elabel) in zip(viz_cols, LABELS):
    with col:
        r = results.get(ename, {})
        if "error" in r:
            continue

        chunks = r.get("retrieved_chunks", [])
        st.markdown(f"**{len(chunks)} chunks retrieved**")

        for i, chunk in enumerate(chunks):
            section = chunk.get("section", chunk.get("detail", chunk.get("source", f"Chunk {i+1}")))
            reasoning = chunk.get("reasoning", "")
            source = chunk.get("source", "")
            score = chunk.get("score")

            if ename == "vector":
                tag = f"Similarity: {score}" if score else "Vector match"
            elif ename == "vectorless":
                tag = reasoning if reasoning else "Tree nav"
            else:
                tag = f"via {source}" if source else "Hybrid"

            st.markdown(
                f"<div class='retrieval-card' style='border-left: 3px solid {COLORS[ename]};'>"
                f"<strong>{section}</strong><br>"
                f"<small style='opacity:0.7;'>{tag}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.expander(f"View ({len(chunk['text'].split())}w)", expanded=False):
                st.code(chunk["text"][:600], language="markdown")

st.divider()

# ── Cost at Scale ────────────────────────────────────────────────────────────
st.subheader("💰 Cost at Scale")
daily_q = st.slider("Expected daily queries", 10, 10000, 100, step=10, key="scale")

cost_cols = st.columns(3)
for col, (ename, elabel) in zip(cost_cols, LABELS):
    with col:
        r = results.get(ename, {})
        if "error" in r:
            continue
        pq = r.get("cost", 0)
        st.markdown(f"**{elabel}**")
        st.metric("Monthly (30d)", f"${pq * daily_q * 30:.2f}")
        st.caption(f"${pq:.6f}/query · ${pq * daily_q:.4f}/day · ${pq * daily_q * 365:.2f}/year")

st.divider()

# ── Export ───────────────────────────────────────────────────────────────────
st.subheader("📥 Export Results")
ex1, ex2, ex3, ex4 = st.columns(4)

with ex1:
    csv_out = build_single_query_csv(question, results)
    st.download_button(
        "📥 Download CSV",
        data=csv_out,
        file_name=f"rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with ex2:
    json_out = build_single_query_json(question, results, analysis)
    st.download_button(
        "📥 Download JSON",
        data=json_out,
        file_name=f"rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=True,
    )
with ex3:
    pdf_bytes = build_single_query_pdf(question, results, analysis, doc_name)
    st.download_button(
        "📥 Download PDF",
        data=pdf_bytes,
        file_name=f"rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
with ex4:
    # Plain-text report
    report = f"RAG COMPARISON REPORT\n{'='*50}\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    report += f"Question: {question}\n"
    report += f"Document: {analysis['total_words']} words, {analysis['heading_count']} headings\n"
    report += f"Structure Score: {analysis['structure_score']}/100\n"
    report += f"Recommended: {analysis['recommendation']}\n\n"
    for ename in ["vector", "vectorless", "hybrid"]:
        r = results.get(ename, {})
        report += f"--- {ENGINE_NAMES[ename]} ---\n"
        if "error" in r:
            report += f"Error: {r['error']}\n\n"
        else:
            t = r.get("tokens", {})
            report += f"Answer: {r['answer']}\n"
            report += f"Latency: {r['time']:.2f}s | Tokens: {t.get('total_tokens', 0)} | Cost: ${r.get('cost', 0):.6f}\n\n"

    st.download_button(
        "📥 Download TXT",
        data=report,
        file_name=f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )



