from __future__ import annotations

import html
import json
import math
import re
import textwrap
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from bs4 import BeautifulSoup
from langgraph.graph import END, START, StateGraph

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
INDEX_HTML = REPO_ROOT / "index.html"
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/agentic-rag"
SOURCE_TITLE = "Build a custom RAG agent with LangGraph"


class DemoState(TypedDict, total=False):
    question: str
    route_decision: Literal["8b", "14b"]
    profile: dict
    search_query: str
    search_query_history: list[str]
    top_chunks: list[dict]
    chunk_stats: dict
    attempts_8b: int
    attempts_14b: int
    retry_reason: str
    quality: dict
    final_answer: str
    final_model: str
    restart_count: int
    logs: list[dict]
    trace_label: str
    retry_action: str
    force_large_retry_once: bool


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "use",
    "uses",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}

SECTION_HINTS = {
    "preprocess": ["preprocess", "documents", "fetch", "split", "index"],
    "retriever": ["retriever", "tool", "vectorstore", "semantic", "search"],
    "query": ["query", "generate", "question", "rewrite"],
    "grading": ["grade", "documents", "relevance", "quality"],
    "answer": ["generate", "answer", "agent", "tool", "respond"],
    "graph": ["assemble", "graph", "conditional", "edge", "state", "workflow"],
}

EXPANSION_RULES = {
    "weird": ["grade documents", "rewrite question", "relevance", "quality"],
    "low": ["grade documents", "relevance", "quality"],
    "quality": ["grade documents", "rewrite question", "relevance"],
    "tool": ["retriever tool", "vectorstore", "semantic search"],
    "decide": ["agent decision", "respond directly", "retrieve context"],
    "answer": ["generate an answer", "retriever tool", "agent"],
    "documents": ["preprocess documents", "vectorstore", "splitters"],
}


@dataclass
class Chunk:
    chunk_id: str
    section: str
    level: int
    text: str
    tokens: list[str]

    @property
    def preview(self) -> str:
        return textwrap.shorten(self.text, width=180, placeholder="…")


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()) if tok not in STOPWORDS]


def clean_excerpt(text: str) -> str:
    text = re.sub(r"Code example:.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    picked = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 25:
            continue
        picked.append(sentence)
        if len(picked) == 2:
            break
    excerpt = " ".join(picked) if picked else text
    return textwrap.shorten(excerpt, width=280, placeholder="…")


def log_event(state: DemoState, node: str, message: str, **payload) -> None:
    state.setdefault("logs", []).append({"node": node, "message": message, "payload": payload})


def fetch_source_document() -> tuple[str, str]:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        html_text = response.read().decode("utf-8", "ignore")
    return SOURCE_URL, html_text


def extract_chunks(html_text: str) -> list[Chunk]:
    soup = BeautifulSoup(html_text, "lxml")
    main = soup.select_one("main") or soup.body
    content_root = None
    for candidate in main.find_all(["h2", "h3"]):
        if "overview" in candidate.get_text(" ", strip=True).lower():
            content_root = candidate.parent
            break
    if content_root is None:
        content_root = main

    chunks: list[Chunk] = []
    current_h2 = "Overview"
    current_h3 = ""
    buffer: list[str] = []
    level = 2
    counter = 0

    def flush_buffer(section_name: str, current_level: int) -> None:
        nonlocal counter, buffer
        text = " ".join(part.strip() for part in buffer if part.strip())
        if not text:
            buffer = []
            return
        counter += 1
        chunks.append(
            Chunk(
                chunk_id=f"C{counter:02d}",
                section=section_name,
                level=current_level,
                text=text,
                tokens=tokenize(section_name + " " + text),
            )
        )
        buffer = []

    for node in content_root.find_all(["h2", "h3", "p", "li", "pre"]):
        name = node.name.lower()
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if name == "h2":
            flush_buffer(f"{current_h2} / {current_h3}" if current_h3 else current_h2, level)
            current_h2 = text
            current_h3 = ""
            level = 2
        elif name == "h3":
            flush_buffer(f"{current_h2} / {current_h3}" if current_h3 else current_h2, level)
            current_h3 = text
            level = 3
        elif name == "pre":
            cleaned = re.sub(r"\s+", " ", text)
            if len(cleaned) > 260:
                cleaned = cleaned[:260] + " …"
            buffer.append(f"Code example: {cleaned}")
        else:
            buffer.append(text)
            if len(" ".join(buffer)) > 900:
                flush_buffer(f"{current_h2} / {current_h3}" if current_h3 else current_h2, level)

    flush_buffer(f"{current_h2} / {current_h3}" if current_h3 else current_h2, level)
    return chunks


def classify_question(question: str) -> tuple[str, dict]:
    tokens = tokenize(question)
    raw_lower = question.lower()
    hard_terms = {"why", "how", "decision", "decide", "flow", "compare", "quality", "weird", "restart", "retry", "grading", "rewrite"}
    matched_hard = sorted(term for term in hard_terms if term in raw_lower or term in tokens)
    complexity = 0
    complexity += 2 if len(tokens) > 10 else 0
    complexity += 2 if any(term in raw_lower for term in ["how", "why", "compare", "decision", "flow"]) else 0
    complexity += 2 if any(term in raw_lower for term in ["quality", "weird", "restart", "retry", "grading", "rewrite"]) else 0
    complexity += 1 if question.count("?") > 1 or " and " in raw_lower else 0
    route = "14b" if complexity >= 4 else "8b"
    profile = {
        "token_count": len(tokens),
        "matched_hard_terms": matched_hard,
        "complexity_score": complexity,
        "reason": "multi-step/diagnostic" if route == "14b" else "direct factual retrieval",
    }
    return route, profile


def build_search_query(question: str, model_name: str, attempt: int, retry_reason: str | None = None) -> str:
    tokens = tokenize(question)
    repair_terms: list[str] = []
    retry_lower = (retry_reason or "").lower()
    if "grade" in retry_lower or "relevance" in retry_lower:
        repair_terms.extend(["grade documents", "relevance"])
    if "rewrite" in retry_lower or "recover" in retry_lower:
        repair_terms.extend(["rewrite question", "recover"])
    if "quality" in retry_lower or "weak lexical grounding" in retry_lower:
        repair_terms.extend(["quality", "semantic search"])

    if model_name == "8b":
        base = tokens[:8]
        for term in repair_terms:
            if term not in base:
                base.append(term)
        return " ".join(base)

    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(EXPANSION_RULES.get(token, []))
    expanded.extend(repair_terms)
    if not any(term in expanded for term in ["graph", "retriever", "documents", "answer"]):
        expanded.extend(["graph", "retriever", "documents"])
    deduped = []
    seen = set()
    for item in expanded:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return " ".join(deduped[:24])


def score_chunk(chunk: Chunk, query_terms: list[str], model_name: str) -> float:
    counts = Counter(chunk.tokens)
    section_tokens = set(tokenize(chunk.section))
    score = 0.0
    for term in query_terms:
        if not term:
            continue
        term_count = counts.get(term, 0)
        if term_count:
            score += 1.2 + math.log1p(term_count)
            if term in section_tokens:
                score += 1.4
    for section_name, hints in SECTION_HINTS.items():
        if any(hint in query_terms for hint in hints) and any(hint in chunk.section.lower() for hint in hints):
            score += 1.1
    if model_name == "14b":
        score += min(len(set(query_terms).intersection(set(chunk.tokens))), 6) * 0.15
    return score


def retrieve(chunks: list[Chunk], query: str, model_name: str) -> tuple[list[dict], dict]:
    query_terms = tokenize(query)
    ranked = []
    for chunk in chunks:
        score = score_chunk(chunk, query_terms, model_name)
        if score > 0:
            ranked.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "score": round(score, 3),
                    "preview": chunk.preview,
                    "text": chunk.text,
                    "term_hits": sorted(set(query_terms).intersection(set(chunk.tokens))),
                }
            )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    top_k = 4 if model_name == "8b" else 6
    selected = ranked[:top_k]
    stats = {
        "ranked_count": len(ranked),
        "top_score": selected[0]["score"] if selected else 0,
        "distinct_sections": len({item["section"] for item in selected}),
        "query_terms": query_terms,
    }
    return selected, stats


def grade_search(question: str, search_query: str, retrieved: list[dict], stats: dict, model_name: str) -> tuple[str, dict, str]:
    if not retrieved:
        return "retry_same_model", {"ok": False, "reason": "no chunks retrieved"}, "No hits came back."

    question_terms = set(tokenize(question))
    coverage_hits = set()
    for item in retrieved[:3]:
        coverage_hits.update(item["term_hits"])
    coverage = len(coverage_hits.intersection(question_terms)) / max(len(question_terms), 1)
    top_score = stats.get("top_score", 0)
    weird_question = any(term in question.lower() for term in ["weird", "odd", "bad", "low quality", "이상"])
    quality = {
        "ok": True,
        "coverage": round(coverage, 3),
        "top_score": top_score,
        "distinct_sections": stats.get("distinct_sections", 0),
        "reason": "retrieval looks grounded",
    }

    if coverage < 0.28 or top_score < 2.2 or stats.get("distinct_sections", 0) < 1:
        quality["ok"] = False
        quality["reason"] = "weak lexical grounding"
    if weird_question and not (
        any("grade" in item["text"].lower() for item in retrieved[:2])
        and any("rewrite" in item["text"].lower() for item in retrieved[:3])
    ):
        quality["ok"] = False
        quality["reason"] = "question asks about bad search quality but retrieved chunks do not cover both grading and rewrite recovery"

    if (
        model_name == "14b"
        and weird_question
        and any("grade" in item["text"].lower() for item in retrieved[:2])
        and any("rewrite" in item["text"].lower() for item in retrieved[:3])
        and top_score >= 6
    ):
        quality["ok"] = True
        quality["reason"] = "14B recovery search covered grading + rewrite path"

    if quality["ok"]:
        return "answer", quality, "Search quality passed."
    if model_name == "8b":
        return "retry_same_model", quality, "8B retrieval looked shaky; retrying with a repaired query."
    return "answer_after_large_retry", quality, "14B retrieval is still imperfect; answer with best available evidence."


def synthesize_answer(question: str, retrieved: list[dict], model_name: str, quality: dict) -> str:
    bullets = []
    for item in retrieved[:3]:
        cleaned = clean_excerpt(item["text"])
        bullets.append(f"- [{item['chunk_id']}] {item['section']}: {cleaned}")
    header = f"Route: {model_name.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage', 'n/a')}"
    return header + "\n" + "\n".join(bullets)


def make_graph(chunks: list[Chunk]):
    graph = StateGraph(DemoState)

    def classify_node(state: DemoState) -> DemoState:
        route, profile = classify_question(state["question"])
        state["route_decision"] = route
        state["profile"] = profile
        state["final_model"] = route
        log_event(state, "classify_8b", f"8B router classified the question for {route.upper()} search.", profile=profile)
        return state

    def search_8b_node(state: DemoState) -> DemoState:
        state["attempts_8b"] = state.get("attempts_8b", 0) + 1
        query = build_search_query(state["question"], "8b", state["attempts_8b"], state.get("retry_reason"))
        state["search_query"] = query
        state.setdefault("search_query_history", []).append(f"8B#{state['attempts_8b']}: {query}")
        retrieved, stats = retrieve(chunks, query, "8b")
        state["top_chunks"] = retrieved
        state["chunk_stats"] = stats
        state["final_model"] = "8b"
        log_event(state, "search_8b", "8B search completed.", attempt=state["attempts_8b"], query=query, stats=stats, top_chunks=retrieved[:3])
        return state

    def search_14b_node(state: DemoState) -> DemoState:
        state["attempts_14b"] = state.get("attempts_14b", 0) + 1
        query = build_search_query(state["question"], "14b", state["attempts_14b"], state.get("retry_reason"))
        state["search_query"] = query
        state.setdefault("search_query_history", []).append(f"14B#{state['attempts_14b']}: {query}")
        retrieved, stats = retrieve(chunks, query, "14b")
        state["top_chunks"] = retrieved
        state["chunk_stats"] = stats
        state["final_model"] = "14b"
        log_event(state, "search_14b", "14B search completed.", attempt=state["attempts_14b"], query=query, stats=stats, top_chunks=retrieved[:4])
        return state

    def grade_node(state: DemoState) -> DemoState:
        model_name = state.get("final_model", state["route_decision"])
        action, quality, message = grade_search(
            state["question"],
            state.get("search_query", ""),
            state.get("top_chunks", []),
            state.get("chunk_stats", {}),
            model_name,
        )
        if state.get("force_large_retry_once") and model_name == "14b" and state.get("attempts_14b", 0) == 1:
            action = "retry_same_model"
            quality["ok"] = False
            quality["reason"] = "strict demo gate requested one extra 14B verification pass"
            message = "Strict gate requested one extra 14B restart before answering."
        elif action == "retry_same_model":
            if model_name == "8b" and state.get("attempts_8b", 0) >= 2:
                action = "escalate_to_14b"
                message = "8B already retried once; escalate to 14B search."
            elif model_name == "14b" and state.get("attempts_14b", 0) >= 2:
                action = "answer_after_large_retry"
                message = "14B already retried once; answer with best available evidence."
        state["quality"] = quality
        state["retry_action"] = action
        state["retry_reason"] = quality.get("reason", "")
        if action in {"retry_same_model", "escalate_to_14b"}:
            state["restart_count"] = state.get("restart_count", 0) + 1
        log_event(state, "grade_results", message, action=action, quality=quality)
        return state

    def answer_node(state: DemoState) -> DemoState:
        state["final_answer"] = synthesize_answer(
            state["question"],
            state.get("top_chunks", []),
            state.get("final_model", state["route_decision"]),
            state.get("quality", {}),
        )
        log_event(state, "answer", "Generated grounded summary from retrieved chunks.", final_model=state.get("final_model"), answer=state["final_answer"])
        return state

    graph.add_node("classify_8b", classify_node)
    graph.add_node("search_8b", search_8b_node)
    graph.add_node("search_14b", search_14b_node)
    graph.add_node("grade_results", grade_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "classify_8b")
    graph.add_conditional_edges("classify_8b", lambda state: "search_8b" if state["route_decision"] == "8b" else "search_14b")
    graph.add_edge("search_8b", "grade_results")
    graph.add_edge("search_14b", "grade_results")

    def route_after_grade(state: DemoState) -> str:
        action = state.get("retry_action")
        if action == "retry_same_model":
            return "search_8b" if state.get("final_model") == "8b" else "search_14b"
        if action == "escalate_to_14b":
            state["final_model"] = "14b"
            return "search_14b"
        return "answer"

    graph.add_conditional_edges("grade_results", route_after_grade, {"search_8b": "search_8b", "search_14b": "search_14b", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()


def build_demo_runs(app, chunks: list[Chunk]) -> list[dict]:
    questions = [
        {
            "label": "Easy route",
            "question": "In the preprocess documents step, what source material does the tutorial fetch for retrieval?",
        },
        {
            "label": "Large-model route",
            "question": "Compare the direct-answer path with the retriever-tool path and explain where document grading and question rewrite sit in the workflow.",
        },
        {
            "label": "Retry + restart route",
            "question": "What happens if the search results look weird or low quality?",
        },
        {
            "label": "14B strict retry route",
            "question": "Compare the normal answer path with the recovery path after low-quality retrieval, including grading and question rewrite.",
            "force_large_retry_once": True,
        },
    ]
    runs = []
    for item in questions:
        state: DemoState = {
            "question": item["question"],
            "logs": [],
            "attempts_8b": 0,
            "attempts_14b": 0,
            "restart_count": 0,
            "trace_label": item["label"],
            "force_large_retry_once": item.get("force_large_retry_once", False),
        }
        result = app.invoke(state)
        runs.append(
            {
                "label": item["label"],
                "question": item["question"],
                "route": result.get("route_decision"),
                "final_model": result.get("final_model"),
                "profile": result.get("profile"),
                "search_query_history": result.get("search_query_history", []),
                "restart_count": result.get("restart_count", 0),
                "quality": result.get("quality", {}),
                "top_chunks": result.get("top_chunks", []),
                "answer": result.get("final_answer", ""),
                "logs": result.get("logs", []),
            }
        )
    return runs


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_run_card(run: dict) -> str:
    chunk_items = "".join(
        f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])}"
        f" <span class='score'>score {chunk['score']}</span><br>{html.escape(chunk['preview'])}</li>"
        for chunk in run["top_chunks"][:4]
    )
    log_lines = []
    for event in run["logs"]:
        payload = json.dumps(event["payload"], ensure_ascii=False, indent=2)
        log_lines.append(f"[{event['node']}] {event['message']}\n{payload}")
    log_text = "\n\n".join(log_lines)
    return f"""
    <article class='card run-card'>
      <div class='run-head'>
        <div>
          <div class='eyebrow'>{html.escape(run['label'])}</div>
          <h3>{html.escape(run['question'])}</h3>
        </div>
        <div class='run-meta'>
          <span class='pill'>initial route: {html.escape(run['route'].upper())}</span>
          <span class='pill'>final model: {html.escape(run['final_model'].upper())}</span>
          <span class='pill'>restarts: {run['restart_count']}</span>
        </div>
      </div>
      <div class='grid grid-2 compact-gap'>
        <div>
          <h4>Search path</h4>
          <ul>{''.join(f'<li>{html.escape(q)}</li>' for q in run['search_query_history'])}</ul>
          <h4>Retrieved chunks</h4>
          <ul>{chunk_items}</ul>
        </div>
        <div>
          <h4>Final grounded answer</h4>
          <pre class='answer-box'>{html.escape(run['answer'])}</pre>
          <h4>Quality gate</h4>
          <pre class='log-box'>{html.escape(json.dumps(run['quality'], ensure_ascii=False, indent=2))}</pre>
        </div>
      </div>
      <h4>Detailed node logs</h4>
      <pre class='log-box big'>{html.escape(log_text)}</pre>
    </article>
    """


def render_index_html(source_meta: dict, chunks: list[Chunk], runs: list[dict]) -> str:
    top_sections = "".join(
        f"<li><strong>{html.escape(chunk.chunk_id)}</strong> · {html.escape(chunk.section)}<br>{html.escape(chunk.preview)}</li>"
        for chunk in chunks[:6]
    )
    run_cards = "\n".join(render_run_card(run) for run in runs)
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>8B/14B 라우팅 RAG LangGraph 데모</title>
  <meta name=\"description\" content=\"8B 질문 분류, 8B/14B 검색 라우팅, 검색 품질 게이트, 재시작 로직까지 포함한 LangGraph RAG 데모\" />
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121933;
      --panel-2: #182244;
      --text: #eef3ff;
      --muted: #a9b6d3;
      --line: #2a3768;
      --accent: #7cc9ff;
      --accent-2: #8effc8;
      --warn: #ffd479;
      --danger: #ff9f9f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, \"Pretendard\", \"Apple SD Gothic Neo\", \"Noto Sans KR\", sans-serif;
      background: linear-gradient(180deg, #09101d 0%, #0b1020 100%);
      color: var(--text);
      line-height: 1.6;
    }}
    a {{ color: var(--accent); }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 88px; }}
    .hero, .card {{
      background: radial-gradient(circle at top left, rgba(124,201,255,.12), transparent 36%), var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 14px 40px rgba(0,0,0,.2);
    }}
    .hero h1 {{ font-size: clamp(30px, 4.8vw, 54px); line-height: 1.18; margin: 0 0 12px; }}
    .hero p {{ max-width: 900px; color: var(--muted); font-size: 18px; }}
    .eyebrow {{ display: inline-block; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin-bottom: 10px; }}
    h2 {{ font-size: clamp(22px, 3vw, 34px); margin: 34px 0 14px; }}
    h3 {{ margin: 0 0 12px; font-size: 22px; }}
    h4 {{ margin: 16px 0 10px; font-size: 16px; color: var(--accent-2); }}
    .grid {{ display: grid; gap: 16px; }}
    .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .compact-gap {{ gap: 12px; }}
    .pillrow, .run-meta {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.03);
      color: var(--muted);
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
    }}
    .stats strong {{ color: var(--accent-2); display: block; font-size: 28px; margin-bottom: 8px; }}
    ul {{ padding-left: 20px; margin: 0; }}
    li {{ color: var(--muted); margin-bottom: 10px; }}
    .flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .step {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 18px; padding: 16px; min-height: 132px; }}
    .step small {{ display: block; color: var(--accent); margin-bottom: 8px; }}
    .run-card {{ margin-top: 18px; }}
    .run-head {{ display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap; align-items: start; }}
    .answer-box, .log-box {{
      margin: 0;
      background: #0a0f1d;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      color: #dbe8ff;
      white-space: pre-wrap;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.55;
    }}
    .log-box.big {{ max-height: 460px; }}
    .score {{ color: var(--warn); font-size: 12px; }}
    .footer {{ margin-top: 32px; color: var(--muted); font-size: 14px; }}
  </style>
</head>
<body>
  <main class=\"wrap\">
    <section class=\"hero\">
      <div class=\"eyebrow\">LangGraph × Routed RAG</div>
      <h1>8B로 질문을 분류하고, 필요하면 14B로 올려 보내는 RAG 그래프</h1>
      <p>요구한 흐름을 그대로 넣었다. <strong>1) 8B 질문 파악 → 2) 8B/14B 검색 분기 → 3) 검색 품질 게이트 → 4) 이상하면 같은 노드 재시작 또는 14B로 승격</strong>. 그리고 그 실행 로그를 HTML에 그대로 붙였다.</p>
      <div class=\"pillrow\">
        <span class=\"pill\">8B router</span>
        <span class=\"pill\">8B search node</span>
        <span class=\"pill\">14B search node</span>
        <span class=\"pill\">quality gate</span>
        <span class=\"pill\">restart / retry</span>
        <span class=\"pill\">GitHub Pages</span>
      </div>
    </section>

    <section>
      <h2>이번 데모에서 실제로 쓴 문서</h2>
      <div class=\"grid grid-2\">
        <article class=\"card\">
          <h3>{html.escape(source_meta['title'])}</h3>
          <p>소스 URL: <a href=\"{html.escape(source_meta['url'])}\">{html.escape(source_meta['url'])}</a></p>
          <ul>
            <li>문서 성격: LangGraph 공식 튜토리얼</li>
            <li>왜 골랐나: RAG, retriever tool, grading, question rewrite, graph assembly가 한 문서 안에 다 들어 있음</li>
            <li>청크 수: {len(chunks)}개</li>
            <li>문서 안 키 섹션: preprocess documents, retriever tool, grade documents, rewrite question, assemble the graph</li>
          </ul>
        </article>
        <article class=\"card\">
          <h3>초기 청크 샘플</h3>
          <ul>{top_sections}</ul>
        </article>
      </div>
    </section>

    <section>
      <h2>그래프 구조</h2>
      <div class=\"flow\">
        <div class=\"step\"><small>Node 1</small><strong>classify_8b</strong><p>질문 길이·복잡도·진단성 단어를 보고 8B/14B 중 어디로 보낼지 결정.</p></div>
        <div class=\"step\"><small>Node 2</small><strong>search_8b</strong><p>짧은 키워드 쿼리로 빠르게 검색. 결과가 약하면 같은 노드를 다시 돌림.</p></div>
        <div class=\"step\"><small>Node 3</small><strong>search_14b</strong><p>확장 쿼리와 넓은 top-k로 재검색. 8B로 부족한 질문을 받음.</p></div>
        <div class=\"step\"><small>Node 4</small><strong>grade_results</strong><p>coverage, top score, 섹션 다양성, 품질 관련 키워드 포함 여부를 평가.</p></div>
        <div class=\"step\"><small>Node 5</small><strong>answer</strong><p>최종 청크에서 근거 문장을 뽑아 answer를 만듦.</p></div>
      </div>
    </section>

    <section>
      <h2>요구사항 매핑</h2>
      <div class=\"grid grid-4\">
        <article class=\"card stats\"><strong>1</strong><p>8B 분류 로그를 남긴다.</p></article>
        <article class=\"card stats\"><strong>2</strong><p>할 만한 질문은 8B 검색, 아니면 14B 검색으로 보낸다.</p></article>
        <article class=\"card stats\"><strong>3</strong><p>검색이 이상하면 같은 노드를 재시작하거나 14B로 올린다.</p></article>
        <article class=\"card stats\"><strong>4</strong><p>RAG용 실제 문서를 웹에서 가져와 청크화했다.</p></article>
      </div>
    </section>

    <section>
      <h2>실제 실행 로그</h2>
      {run_cards}
    </section>

    <section>
      <h2>운영 메모</h2>
      <div class=\"grid grid-2\">
        <article class=\"card\">
          <h3>지금 이 데모가 보여주는 것</h3>
          <ul>
            <li>LangGraph로 <strong>재시도/재시작 edge</strong>를 실제로 걸 수 있다는 점</li>
            <li>작은 모델을 router로 두고 큰 모델을 비용 높은 fallback으로 두는 구조</li>
            <li>최종 HTML에서 <strong>중간 로그를 그대로 확인</strong>할 수 있다는 점</li>
          </ul>
        </article>
        <article class=\"card\">
          <h3>실제 로컬 8B/14B로 바꾸려면</h3>
          <ul>
            <li>현재는 <strong>LangGraph 제어 흐름을 검증하기 위한 역할 기반 데모</strong>다.</li>
            <li>search/query rewrite 부분의 함수만 OpenAI-compatible local endpoint(예: Ollama, llama.cpp server)로 교체하면 된다.</li>
            <li>회사 4070 환경에서는 8B를 router, 14B를 difficult-retrieval fallback으로 두는 구성이 가장 현실적이다.</li>
          </ul>
        </article>
      </div>
      <p class=\"footer\">Generated from a real LangGraph execution over a fetched public document. Build artifacts are stored in <code>artifacts/</code>.</p>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    source_url, html_text = fetch_source_document()
    chunks = extract_chunks(html_text)
    source_meta = {"title": SOURCE_TITLE, "url": source_url, "chunk_count": len(chunks)}
    app = make_graph(chunks)
    runs = build_demo_runs(app, chunks)

    save_json(ARTIFACTS_DIR / "source_document.json", {
        **source_meta,
        "chunks": [chunk.__dict__ | {"preview": chunk.preview} for chunk in chunks],
    })
    save_json(ARTIFACTS_DIR / "demo_runs.json", runs)
    save_json(ARTIFACTS_DIR / "build_summary.json", {
        "source": source_meta,
        "questions": [run["question"] for run in runs],
        "routes": [run["route"] for run in runs],
        "restarts": [run["restart_count"] for run in runs],
    })
    INDEX_HTML.write_text(render_index_html(source_meta, chunks, runs), encoding="utf-8")
    print(json.dumps({
        "source": source_meta,
        "runs": [{
            "label": run["label"],
            "route": run["route"],
            "final_model": run["final_model"],
            "restart_count": run["restart_count"],
            "quality": run["quality"],
        } for run in runs],
        "index_html": str(INDEX_HTML),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
