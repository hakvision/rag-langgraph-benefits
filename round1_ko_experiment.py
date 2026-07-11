from __future__ import annotations

import html
import json
import re
import textwrap
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORT_HTML = REPO_ROOT / "round1_ko.html"
RESULTS_JSON = ARTIFACTS_DIR / "round1_ko_results.json"
SUMMARY_JSON = ARTIFACTS_DIR / "round1_ko_summary.json"
SOURCE_JSON = ARTIFACTS_DIR / "round1_ko_source_document.json"

SOURCE_URL = "https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/"
SOURCE_TITLE = "RAG란? - 검색 증강 생성 AI 설명 - AWS"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in", "into", "is", "it", "of", "on", "or",
    "the", "to", "what", "when", "where", "which", "with", "you", "your",
    "것", "가", "을", "를", "은", "는", "이", "의", "에", "와", "과", "로", "으로", "도", "좀", "더", "수", "및", "등", "때", "왜", "무엇", "어떻게", "설명", "하나요", "인가요", "있는", "합니다", "대한", "에서", "하는", "하는지", "무엇인가요",
}

HARD_TERMS = {"차이", "비교", "왜", "어떻게", "작동", "흐름", "업데이트", "신뢰", "제어", "시맨틱", "이점", "한계"}

QUESTIONS = [
    {"label": "Q1", "category": "definition", "question": "검색 증강 생성(RAG)을 AWS 문서는 어떻게 정의하나요?"},
    {"label": "Q2", "category": "importance", "question": "AWS 문서가 말하는 RAG의 필요성은 무엇이며, LLM의 어떤 문제를 줄이려 하나요?"},
    {"label": "Q3", "category": "benefits", "question": "RAG의 주요 이점을 문서 기준으로 정리해줘."},
    {"label": "Q4", "category": "cost", "question": "왜 AWS 문서는 RAG를 비용 효율적인 구현 방식이라고 설명하나요?"},
    {"label": "Q5", "category": "trust", "question": "RAG는 사용자 신뢰를 어떻게 강화한다고 설명하나요?"},
    {"label": "Q6", "category": "workflow", "question": "RAG는 전체적으로 어떤 단계로 작동하나요?"},
    {"label": "Q7", "category": "retrieval", "question": "문서에서 말하는 관련 정보 검색 단계는 어떤 식으로 이뤄지나요?"},
    {"label": "Q8", "category": "prompting", "question": "LLM 프롬프트 확장 단계는 왜 필요하고 무엇을 추가하나요?"},
    {"label": "Q9", "category": "comparison", "question": "AWS 문서 기준으로 RAG와 시맨틱 검색의 차이를 비교해줘."},
    {"label": "Q10", "category": "aws-support", "question": "AWS는 RAG 구축을 위해 어떤 서비스들을 어떻게 지원한다고 설명하나요?"},
]


@dataclass
class Chunk:
    chunk_id: str
    section: str
    text: str
    tokens: list[str]

    @property
    def preview(self) -> str:
        return textwrap.shorten(self.text, width=180, placeholder="…")


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[가-힣A-Za-z0-9]+", text.lower()):
        raw = raw.strip()
        if not raw or raw in STOPWORDS:
            continue
        if len(raw) == 1 and not raw.isdigit():
            continue
        tokens.append(raw)
    return tokens


def clean_excerpt(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    sentences = re.split(r"(?<=[.!?다요])\s+", compact)
    picked = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 24:
            continue
        picked.append(sentence)
        if len(picked) == 2:
            break
    excerpt = " ".join(picked) if picked else compact
    return textwrap.shorten(excerpt, width=280, placeholder="…")


def fetch_source_document() -> tuple[str, str]:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        html_text = response.read().decode("utf-8", "ignore")
    return SOURCE_URL, html_text


def extract_chunks(html_text: str) -> list[Chunk]:
    soup = BeautifulSoup(html_text, "lxml")
    main = soup.find("main") or soup.body
    chunks: list[Chunk] = []
    current_h2 = "개요"
    current_h3 = ""
    buffer: list[str] = []
    counter = 0
    started = False

    def flush() -> None:
        nonlocal counter, buffer
        text = " ".join(part.strip() for part in buffer if part.strip())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            buffer = []
            return
        counter += 1
        section = current_h2 if not current_h3 else f"{current_h2} / {current_h3}"
        chunks.append(Chunk(chunk_id=f"K{counter:02d}", section=section, text=text, tokens=tokenize(section + " " + text)))
        buffer = []

    for node in main.find_all(["h1", "h2", "h3", "p", "li"]):
        name = node.name.lower()
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if name == "h1":
            started = True
            current_h2 = text
            current_h3 = ""
            continue
        if not started:
            continue
        if name == "h2":
            flush()
            current_h2 = text
            current_h3 = ""
            continue
        if name == "h3":
            flush()
            current_h3 = text
            continue
        buffer.append(text)
        if len(" ".join(buffer)) > 900:
            flush()

    flush()
    return chunks


def score_chunk(chunk: Chunk, query_tokens: list[str]) -> tuple[float, list[str]]:
    token_counts = {}
    for token in chunk.tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    hits: list[str] = []
    score = 0.0
    section_lower = chunk.section.lower()
    for token in query_tokens:
        if token in token_counts:
            hits.append(token)
            score += 2.2 + min(token_counts[token] * 0.45, 1.8)
        if token in section_lower:
            score += 1.5

    if any(term in chunk.section for term in ["이점", "작동", "차이", "지원", "중요"]):
        score += 0.4
    if len(set(hits)) >= 3:
        score += 1.2
    return round(score, 3), sorted(set(hits))


def retrieve(chunks: list[Chunk], query: str, model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_tokens = tokenize(query)
    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        score, hits = score_chunk(chunk, query_tokens)
        if score <= 0:
            continue
        scored.append(
            {
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "text": chunk.text,
                "preview": chunk.preview,
                "score": score,
                "term_hits": hits,
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    limit = 4 if model_name == "8b" else 6
    top = scored[:limit]
    stats = {
        "query_tokens": query_tokens,
        "top_score": top[0]["score"] if top else 0,
        "distinct_sections": len({item["section"] for item in top}),
        "candidate_count": len(scored),
    }
    return top, stats


def classify_question(question: str) -> tuple[Literal["8b", "14b"], dict[str, Any]]:
    tokens = tokenize(question)
    matched = sorted(term for term in HARD_TERMS if term in question)
    complexity = 0
    complexity += 2 if len(tokens) >= 8 else 0
    complexity += 2 if any(term in question for term in ["비교", "차이", "왜", "어떻게", "작동", "흐름"]) else 0
    complexity += 1 if any(term in question for term in ["신뢰", "제어", "업데이트", "지원", "이점"]) else 0
    route: Literal["8b", "14b"] = "14b" if complexity >= 4 else "8b"
    return route, {
        "token_count": len(tokens),
        "matched_hard_terms": matched,
        "complexity_score": complexity,
        "reason": "multi-step/comparison" if route == "14b" else "direct factual retrieval",
        "source": "heuristic-mock",
    }


def build_search_query(question: str, model_name: Literal["8b", "14b"], attempt: int, retry_reason: str = "") -> str:
    base = tokenize(question)
    if model_name == "8b":
        selected = base[:8]
        if attempt > 1 and retry_reason:
            selected.extend(tokenize(retry_reason)[:4])
        return " ".join(dict.fromkeys(selected))

    expanded = list(base)
    if any(term in question for term in ["비교", "차이"]):
        expanded.extend(["차이", "비교", "시맨틱", "검색"])
    if any(term in question for term in ["작동", "흐름"]):
        expanded.extend(["작동", "단계", "검색", "프롬프트", "업데이트"])
    if any(term in question for term in ["이점", "신뢰", "제어"]):
        expanded.extend(["이점", "신뢰", "제어", "최신", "비용"])
    if any(term in question for term in ["지원", "서비스", "AWS"]):
        expanded.extend(["aws", "bedrock", "kendra", "sagemaker"])
    if retry_reason:
        expanded.extend(tokenize(retry_reason)[:6])
    return " ".join(dict.fromkeys(expanded[:18]))


def grade_search(question: str, chunks: list[dict[str, Any]], stats: dict[str, Any], model_name: Literal["8b", "14b"]) -> tuple[str, dict[str, Any], str]:
    question_tokens = set(tokenize(question))
    chunk_hits = set()
    for item in chunks:
        chunk_hits.update(item.get("term_hits", []))
    coverage = round(len(question_tokens & chunk_hits) / max(len(question_tokens), 1), 3)
    quality = {
        "ok": False,
        "coverage": coverage,
        "top_score": stats.get("top_score", 0),
        "distinct_sections": stats.get("distinct_sections", 0),
        "reason": "",
    }

    if stats.get("top_score", 0) >= 6.0 and coverage >= 0.25:
        quality["ok"] = True
        quality["reason"] = "retrieval looks grounded"
        return "answer", quality, "품질 게이트 통과: 상위 청크가 질문과 충분히 맞물림."

    if model_name == "8b":
        quality["reason"] = "8B retrieval looked shallow; escalate to 14B"
        return "escalate_to_14b", quality, "품질 게이트가 8B 검색을 얕다고 판단해 14B로 승격함."

    quality["reason"] = "14B retrieval still partial; answer with current evidence"
    return "answer", quality, "14B 검색도 완벽하진 않지만 현재 근거로 answer 단계로 진행함."


def build_answer(question: str, final_model: str, chunks: list[dict[str, Any]], quality: dict[str, Any]) -> str:
    lines = [
        f"Route: {final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}",
        f"질문: {question}",
        "문서 근거 요약:",
    ]
    for item in chunks[:4]:
        lines.append(f"- [{item['chunk_id']}] {item['section']}: {clean_excerpt(item['text'])}")
    return "\n".join(lines)


def run_one(question_item: dict[str, Any], chunks: list[Chunk]) -> dict[str, Any]:
    question = question_item["question"]
    flow: list[str] = []
    logs: list[dict[str, Any]] = []
    node_timings: list[dict[str, Any]] = []
    explanation: list[str] = []
    search_attempts: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    t0 = time.perf_counter()
    route, analysis = classify_question(question)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    flow.append(f"classify_8b → {route}")
    logs.append({"node": "classify_8b", "message": "질문 분류 완료", "payload": {"analysis": analysis, "elapsed_ms": elapsed}})
    node_timings.append({"node": "classify_8b", "elapsed_ms": elapsed, "details": analysis})
    explanation.append(f"처음에는 classify_8b가 질문을 {route.upper()} 경로로 분류했다. 이유: {analysis['reason']}")

    current_model: Literal["8b", "14b"] = route
    restart_count = 0
    retry_reason = ""
    attempt_counts = {"8b": 0, "14b": 0}
    final_chunks: list[dict[str, Any]] = []
    final_quality: dict[str, Any] = {}

    while True:
        attempt_counts[current_model] += 1
        query = build_search_query(question, current_model, attempt_counts[current_model], retry_reason)
        t0 = time.perf_counter()
        top_chunks, stats = retrieve(chunks, query, current_model)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_chunks = top_chunks
        flow.append(f"search_{current_model}")
        logs.append({
            "node": f"search_{current_model}",
            "message": f"{current_model.upper()} 검색 완료",
            "payload": {"attempt": attempt_counts[current_model], "query": query, "stats": stats, "selected_chunk_ids": [item['chunk_id'] for item in top_chunks], "elapsed_ms": elapsed},
        })
        node_timings.append({"node": f"search_{current_model}", "elapsed_ms": elapsed, "details": {"attempt": attempt_counts[current_model], "query": query, **stats}})
        search_attempts.append({"node": f"search_{current_model}", "elapsed_ms": elapsed, "attempt": attempt_counts[current_model], "query": query, "chunks": top_chunks})

        t0 = time.perf_counter()
        action, quality, message = grade_search(question, top_chunks, stats, current_model)
        grade_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_quality = quality
        flow.append(f"grade_results → {action}")
        logs.append({"node": "grade_results", "message": message, "payload": {"action": action, "quality": quality, "elapsed_ms": grade_elapsed}})
        node_timings.append({"node": "grade_results", "elapsed_ms": grade_elapsed, "details": {"action": action, "quality": quality}})
        explanation.append(f"search_{current_model}에서 {', '.join(item['chunk_id'] for item in top_chunks[:4]) or '청크 없음'}을 가져왔고, grade_results가 '{quality['reason']}'로 판단해 {action}을 선택했다.")

        if action == "answer":
            break
        if action == "escalate_to_14b":
            restart_count += 1
            retry_reason = quality["reason"]
            current_model = "14b"
            flow.append("route_upgrade 8b→14b")
            logs.append({"node": "route_upgrade", "message": "품질 게이트가 14B 승격을 요청함", "payload": {"reason": retry_reason}})
            node_timings.append({"node": "route_upgrade", "elapsed_ms": 0.0, "details": {"from": "8b", "to": "14b", "reason": retry_reason}})
            explanation.append("8B 검색이 얕다고 판단되어 14B 검색으로 승격했다.")
            continue
        break

    t0 = time.perf_counter()
    final_answer = build_answer(question, current_model, final_chunks, final_quality)
    answer_elapsed = round((time.perf_counter() - t0) * 1000, 2)
    flow.append("answer")
    logs.append({"node": "answer", "message": "최종 답변 생성", "payload": {"final_model": current_model, "elapsed_ms": answer_elapsed, "answer_preview": textwrap.shorten(final_answer, width=240, placeholder='…')}})
    node_timings.append({"node": "answer", "elapsed_ms": answer_elapsed, "details": {"final_model": current_model}})
    explanation.append(f"마지막에는 {current_model.upper()} answer 단계가 현재 상위 청크를 근거로 응답을 만들었다.")

    return {
        "label": question_item["label"],
        "category": question_item["category"],
        "question": question,
        "provider_mode": "mock",
        "backend_available": False,
        "route_decision": route,
        "question_analysis": analysis,
        "search_query_history": [attempt["query"] for attempt in search_attempts],
        "search_attempts": search_attempts,
        "top_chunks": final_chunks,
        "restart_count": restart_count,
        "final_model": current_model,
        "quality": final_quality,
        "final_answer": final_answer,
        "logs": logs,
        "node_timings": node_timings,
        "flow": flow,
        "explanation": explanation,
        "total_ms": round((time.perf_counter() - run_start) * 1000, 2),
    }


def render_timing_table(run: dict[str, Any]) -> str:
    rows = []
    for item in run["node_timings"]:
        rows.append(f"<tr><td>{html.escape(item['node'])}</td><td>{item['elapsed_ms']:.2f} ms</td><td><pre>{html.escape(json.dumps(item['details'], ensure_ascii=False, indent=2))}</pre></td></tr>")
    return "".join(rows)


def render_search_attempts(run: dict[str, Any]) -> str:
    blocks = []
    for attempt in run["search_attempts"]:
        chunk_cards = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>score {chunk['score']}</span><br>{html.escape(chunk['preview'])}<br><span class='hits'>term hits: {html.escape(', '.join(chunk.get('term_hits', [])) or '-')}</span></li>"
            for chunk in attempt["chunks"]
        ) or "<li>no chunks</li>"
        blocks.append(
            f"<article class='attempt'><div class='attempt-head'><strong>{html.escape(attempt['node'])}</strong><span>{attempt['elapsed_ms']:.2f} ms</span></div><div class='query'>{html.escape(attempt['query'])}</div><ul>{chunk_cards}</ul></article>"
        )
    return "\n".join(blocks)


def render_run(run: dict[str, Any]) -> str:
    return f"""
    <section class='card run'>
      <div class='run-head'>
        <div>
          <div class='eyebrow'>{html.escape(run['label'])} · {html.escape(run['category'])}</div>
          <h2>{html.escape(run['question'])}</h2>
        </div>
        <div class='pillrow'>
          <span class='pill'>initial route: {html.escape(run['route_decision'].upper())}</span>
          <span class='pill'>final model: {html.escape(run['final_model'].upper())}</span>
          <span class='pill'>restarts: {run['restart_count']}</span>
          <span class='pill'>total: {run['total_ms']:.2f} ms</span>
        </div>
      </div>
      <div class='grid two'>
        <article class='panel'>
          <h3>LangGraph-style flow</h3>
          <ol>{''.join(f'<li>{html.escape(step)}</li>' for step in run['flow'])}</ol>
          <h3>왜 이런 결과가 나왔나</h3>
          <ul>{''.join(f'<li>{html.escape(line)}</li>' for line in run['explanation'])}</ul>
        </article>
        <article class='panel'>
          <h3>Final answer</h3>
          <pre>{html.escape(run['final_answer'])}</pre>
          <h3>Quality gate</h3>
          <pre>{html.escape(json.dumps(run['quality'], ensure_ascii=False, indent=2))}</pre>
        </article>
      </div>
      <article class='panel'>
        <h3>Search attempts + retrieved chunks</h3>
        {render_search_attempts(run)}
      </article>
      <article class='panel'>
        <h3>Per-node timing</h3>
        <table>
          <thead><tr><th>node</th><th>elapsed</th><th>details</th></tr></thead>
          <tbody>{render_timing_table(run)}</tbody>
        </table>
      </article>
      <article class='panel'>
        <h3>Raw logs</h3>
        <pre>{html.escape(json.dumps(run['logs'], ensure_ascii=False, indent=2))}</pre>
      </article>
    </section>
    """


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    runs = payload["runs"]
    run_html = "\n".join(render_run(run) for run in runs)
    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Round1 KO · AWS RAG 문서 실험</title>
  <style>
    :root {{ --bg:#09101d; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; --warn:#ffd479; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo","Noto Sans KR",sans-serif; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); line-height:1.6; }}
    .wrap {{ max-width:1280px; margin:0 auto; padding:28px 18px 80px; }}
    .hero,.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 12px 36px rgba(0,0,0,.22); }}
    .hero h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,52px); }}
    .hero p,.muted,li {{ color:var(--muted); }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; margin-bottom:8px; }}
    .pillrow {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .pill {{ border:1px solid var(--line); border-radius:999px; padding:7px 12px; font-size:13px; color:var(--muted); background:rgba(255,255,255,.03); }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-top:18px; }}
    .stats .panel strong {{ display:block; color:var(--ok); font-size:28px; margin-bottom:6px; }}
    .run {{ margin-top:18px; }}
    .run-head {{ display:flex; justify-content:space-between; gap:14px; flex-wrap:wrap; align-items:start; }}
    .grid.two {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .attempt {{ background:var(--panel2); border:1px solid var(--line); border-radius:16px; padding:14px; margin-top:12px; }}
    .attempt-head {{ display:flex; justify-content:space-between; gap:10px; align-items:center; margin-bottom:8px; }}
    .query {{ background:#0a0f1d; border:1px solid var(--line); border-radius:12px; padding:10px 12px; margin-bottom:10px; color:#dbe8ff; font-size:14px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ border-top:1px solid var(--line); padding:10px; text-align:left; vertical-align:top; }}
    th {{ color:var(--accent); font-size:13px; }}
    pre {{ margin:0; white-space:pre-wrap; word-break:break-word; background:#0a0f1d; border:1px solid var(--line); border-radius:14px; padding:12px; color:#dbe8ff; font-size:13px; }}
    h2,h3 {{ margin:0 0 10px; }}
    ol,ul {{ margin:0; padding-left:20px; }}
    .score {{ color:var(--warn); font-size:12px; }}
    .hits {{ color:var(--accent); font-size:12px; }}
    a {{ color:var(--accent); }}
    @media (max-width: 980px) {{ .grid.two {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class='wrap'>
    <section class='hero'>
      <div class='eyebrow'>round1 · qwen3 routed rag · ko</div>
      <h1>AWS 한국어 RAG 문서를 기준으로 돌린 round1</h1>
      <p>이번 round1은 이 맥에서 실제 Qwen3/Ollama 백엔드가 없어서 <strong>mock 라우팅/검색 실험</strong>으로 실행했다. 대신 문서는 실제 AWS 한국어 원문을 가져와 청크화했고, 질문/검색/품질 게이트/흐름/HTML 리포트는 전부 실제 산출물로 남겼다.</p>
      <div class='pillrow'>
        <span class='pill'>provider: {html.escape(summary['provider_mode'])}</span>
        <span class='pill'>router: {html.escape(summary['models']['router_8b'])}</span>
        <span class='pill'>large: {html.escape(summary['models']['large_14b'])}</span>
        <span class='pill'>doc: <a href='{html.escape(summary['doc_url'])}' target='_blank'>{html.escape(summary['doc_title'])}</a></span>
      </div>
      <div class='stats'>
        <article class='panel'><strong>{summary['question_count']}</strong><span>questions</span></article>
        <article class='panel'><strong>{summary['routes']['8b']}</strong><span>initial 8B routes</span></article>
        <article class='panel'><strong>{summary['routes']['14b']}</strong><span>initial 14B routes</span></article>
        <article class='panel'><strong>{summary['total_restarts']}</strong><span>total restarts</span></article>
        <article class='panel'><strong>{summary['avg_total_ms']:.2f} ms</strong><span>avg end-to-end time</span></article>
        <article class='panel'><strong>{summary['chunk_count']}</strong><span>document chunks</span></article>
      </div>
    </section>
    {run_html}
  </main>
</body>
</html>
"""


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    doc_url, html_text = fetch_source_document()
    chunks = extract_chunks(html_text)
    SOURCE_JSON.write_text(json.dumps({
        "title": SOURCE_TITLE,
        "url": doc_url,
        "chunk_count": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    runs = [run_one(item, chunks) for item in QUESTIONS]
    summary = {
        "round": "round1-ko",
        "doc_title": SOURCE_TITLE,
        "doc_url": doc_url,
        "chunk_count": len(chunks),
        "provider_mode": "mock",
        "backend_available": False,
        "models": {"router_8b": "qwen3:8b", "large_14b": "qwen3:14b"},
        "question_count": len(runs),
        "routes": {"8b": sum(1 for run in runs if run['route_decision'] == '8b'), "14b": sum(1 for run in runs if run['route_decision'] == '14b')},
        "total_restarts": sum(run['restart_count'] for run in runs),
        "avg_total_ms": round(sum(run['total_ms'] for run in runs) / max(len(runs), 1), 2),
    }
    payload = {"summary": summary, "runs": runs}
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_HTML.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({
        "report_html": str(REPORT_HTML),
        "results_json": str(RESULTS_JSON),
        "summary_json": str(SUMMARY_JSON),
        "source_json": str(SOURCE_JSON),
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
