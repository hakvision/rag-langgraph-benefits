from __future__ import annotations

import asyncio
import html
import json
import shutil
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Literal

from server import (
    DOC_STORE,
    BackendSettings,
    analyze_question,
    append_log,
    grade_search,
    make_initial_state,
    model_answer,
    run_search_node,
    should_force_14b_retry,
)

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORT_HTML = REPO_ROOT / "round1.html"
RESULTS_JSON = ARTIFACTS_DIR / "round1_results.json"
SUMMARY_JSON = ARTIFACTS_DIR / "round1_summary.json"

QUESTIONS = [
    {
        "label": "Q1",
        "category": "preprocess",
        "question": "How does the tutorial fetch and preprocess documents before retrieval starts?",
    },
    {
        "label": "Q2",
        "category": "chunking",
        "question": "Why does the tutorial split the fetched documents into smaller chunks before indexing?",
    },
    {
        "label": "Q3",
        "category": "retriever",
        "question": "How is the retriever tool created and what components does it depend on?",
    },
    {
        "label": "Q4",
        "category": "router",
        "question": "What does the generate_query_or_respond step decide before the graph calls the retriever tool?",
    },
    {
        "label": "Q5",
        "category": "grading",
        "question": "What exactly is the grade_documents node checking when it looks at retrieved results?",
    },
    {
        "label": "Q6",
        "category": "rewrite",
        "question": "When the retrieved documents are not relevant enough, how does the rewrite_question node help recovery?",
    },
    {
        "label": "Q7",
        "category": "answer",
        "question": "How does the final answer generation step combine the original question with retrieved context?",
    },
    {
        "label": "Q8",
        "category": "graph",
        "question": "How do conditional edges connect query generation, retrieval, grading, and answer generation in this graph?",
    },
    {
        "label": "Q9",
        "category": "comparison",
        "question": "Compare the direct response path with the retrieval path and explain where grading and question rewrite sit in the workflow.",
    },
    {
        "label": "Q10",
        "category": "recovery",
        "question": "What happens if the search results look weird or low quality, and how does the graph recover before answering?",
    },
]


async def run_one(question_item: dict[str, Any], settings: BackendSettings, backend_available: bool) -> dict[str, Any]:
    question = question_item["question"]
    state = make_initial_state(question)
    state["backend_available"] = backend_available
    node_timings: list[dict[str, Any]] = []
    search_attempts: list[dict[str, Any]] = []
    flow: list[str] = []
    explanation: list[str] = []
    run_start = time.perf_counter()

    t0 = time.perf_counter()
    route, analysis = await analyze_question(question, settings)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    state["route_decision"] = route
    state["question_analysis"] = analysis
    state["final_model"] = route
    append_log(state, "classify_8b", f"8B router classified the question for {route.upper()} search.", analysis=analysis, elapsed_ms=elapsed_ms)
    node_timings.append({"node": "classify_8b", "elapsed_ms": elapsed_ms, "details": {"route": route, "reason": analysis.get("reason"), "source": analysis.get("source")}})
    flow.append(f"classify_8b → {route}")
    explanation.append(f"처음에는 classify_8b가 질문을 {route.upper()} 경로로 분류했다. 이유: {analysis.get('reason', 'n/a')}")

    current_model: Literal["8b", "14b"] = route  # type: ignore[assignment]
    while True:
        search_node = f"search_{current_model}"
        t0 = time.perf_counter()
        before_count = len(state["logs"])
        await run_search_node(state, settings, current_model)
        search_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        flow.append(search_node)
        latest_search_log = state["logs"][-1]
        latest_search_log["payload"]["elapsed_ms"] = search_elapsed_ms
        latest_search_log["payload"]["selected_chunk_ids"] = [chunk["chunk_id"] for chunk in state.get("top_chunks", [])]
        node_timings.append({
            "node": search_node,
            "elapsed_ms": search_elapsed_ms,
            "details": {
                "attempt": state["attempts_8b"] if current_model == "8b" else state["attempts_14b"],
                "query": state["search_query_history"][-1],
                "selected_chunk_ids": [chunk["chunk_id"] for chunk in state.get("top_chunks", [])],
                "top_score": state["top_chunks"][0]["score"] if state.get("top_chunks") else 0,
            },
        })
        search_attempts.append({
            "node": search_node,
            "elapsed_ms": search_elapsed_ms,
            "attempt": state["attempts_8b"] if current_model == "8b" else state["attempts_14b"],
            "query": state["search_query_history"][-1],
            "chunks": state.get("top_chunks", []),
            "log": latest_search_log,
        })

        t0 = time.perf_counter()
        action, quality, message = grade_search(
            state["question"],
            state["search_query_history"][-1],
            state["top_chunks"],
            {
                "top_score": state["top_chunks"][0]["score"] if state["top_chunks"] else 0,
                "distinct_sections": len({item["section"] for item in state["top_chunks"]}),
            },
            current_model,
        )
        if should_force_14b_retry(state):
            action = "retry_same_model"
            quality["ok"] = False
            quality["reason"] = "strict gate requested one extra 14B verification pass"
            message = "Strict gate requested one extra 14B restart before answering."
        grade_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        state["quality"] = quality
        append_log(state, "grade_results", message, action=action, quality=quality, elapsed_ms=grade_elapsed_ms)
        node_timings.append({"node": "grade_results", "elapsed_ms": grade_elapsed_ms, "details": {"action": action, "quality": quality}})
        flow.append(f"grade_results → {action}")
        explanation.append(
            f"{search_node}에서 {', '.join(chunk['chunk_id'] for chunk in state.get('top_chunks', [])[:4]) or 'no chunks'}를 가져왔고, grade_results가 '{quality.get('reason')}'로 판단해 {action} 액션을 선택했다."
        )

        if action == "answer":
            break
        if action == "retry_same_model":
            state["restart_count"] += 1
            state["retry_reason"] = quality.get("reason", "")
            if current_model == "8b" and state["attempts_8b"] >= 2:
                current_model = "14b"
                append_log(state, "route_upgrade", "8B retries exhausted; escalating to 14B.", reason=state["retry_reason"], elapsed_ms=0)
                node_timings.append({"node": "route_upgrade", "elapsed_ms": 0.0, "details": {"from": "8b", "to": "14b", "reason": state["retry_reason"]}})
                flow.append("route_upgrade 8b→14b")
                explanation.append("8B가 두 번 시도했는데도 부족해서 14B로 승격했다.")
            elif current_model == "14b" and state["attempts_14b"] >= 2:
                explanation.append("14B도 재시도 한도에 도달해서 현재 근거로 답변 단계로 넘어가게 됐다.")
                break
            continue
        if action == "escalate_to_14b":
            state["restart_count"] += 1
            state["retry_reason"] = quality.get("reason", "")
            current_model = "14b"
            append_log(state, "route_upgrade", "Quality gate escalated retrieval to 14B.", reason=state["retry_reason"], elapsed_ms=0)
            node_timings.append({"node": "route_upgrade", "elapsed_ms": 0.0, "details": {"from": "8b", "to": "14b", "reason": state["retry_reason"]}})
            flow.append("route_upgrade 8b→14b")
            explanation.append("품질 게이트가 14B 승격을 요구해서 큰 모델 검색으로 넘어갔다.")
            continue
        explanation.append("14B 검색 품질이 완벽하진 않지만 현재 근거로 answer 단계에 진입했다.")
        break

    t0 = time.perf_counter()
    state["final_answer"] = await model_answer(state["question"], state["final_model"], settings, state["top_chunks"], state["quality"])
    answer_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    append_log(state, "answer", "Generated final answer.", final_model=state["final_model"], answer_preview=textwrap.shorten(state["final_answer"], width=260, placeholder="…"), elapsed_ms=answer_elapsed_ms)
    node_timings.append({"node": "answer", "elapsed_ms": answer_elapsed_ms, "details": {"final_model": state["final_model"]}})
    flow.append("answer")
    explanation.append(f"마지막에는 {state['final_model'].upper()} answer 단계가 현재 상위 청크를 근거로 응답을 만들었다.")

    total_ms = round((time.perf_counter() - run_start) * 1000, 2)
    return {
        "label": question_item["label"],
        "category": question_item["category"],
        "question": question,
        "provider_mode": settings.provider_mode,
        "backend_available": backend_available,
        "route_decision": state["route_decision"],
        "question_analysis": state["question_analysis"],
        "search_query_history": state["search_query_history"],
        "search_attempts": search_attempts,
        "top_chunks": state["top_chunks"],
        "restart_count": state["restart_count"],
        "final_model": state["final_model"],
        "quality": state["quality"],
        "final_answer": state["final_answer"],
        "logs": state["logs"],
        "node_timings": node_timings,
        "flow": flow,
        "explanation": explanation,
        "total_ms": total_ms,
    }


async def run_all() -> dict[str, Any]:
    backend_available = bool(shutil.which("ollama"))
    if backend_available:
        try:
            response = subprocess.run(
                ["curl", "-s", "http://127.0.0.1:11434/api/tags"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            backend_available = response.returncode == 0 and "models" in response.stdout
        except Exception:
            backend_available = False

    settings = BackendSettings(
        provider_mode="openai-compatible" if backend_available else "mock",
        model_8b="qwen3:8b",
        model_14b="qwen3:14b",
    )
    runs = []
    for item in QUESTIONS:
        runs.append(await run_one(item, settings, backend_available))

    summary = {
        "round": "round1",
        "doc_title": DOC_STORE.title,
        "doc_url": DOC_STORE.url,
        "chunk_count": len(DOC_STORE.chunks),
        "provider_mode": settings.provider_mode,
        "backend_available": backend_available,
        "models": {"router_8b": settings.model_8b, "large_14b": settings.model_14b},
        "question_count": len(runs),
        "routes": {"8b": sum(1 for run in runs if run["route_decision"] == "8b"), "14b": sum(1 for run in runs if run["route_decision"] == "14b")},
        "total_restarts": sum(run["restart_count"] for run in runs),
        "avg_total_ms": round(sum(run["total_ms"] for run in runs) / max(len(runs), 1), 2),
    }
    return {"summary": summary, "runs": runs}


def render_timing_table(run: dict[str, Any]) -> str:
    rows = []
    for item in run["node_timings"]:
        rows.append(
            f"<tr><td>{html.escape(item['node'])}</td><td>{item['elapsed_ms']:.2f} ms</td><td><pre>{html.escape(json.dumps(item['details'], ensure_ascii=False, indent=2))}</pre></td></tr>"
        )
    return "".join(rows)


def render_search_attempts(run: dict[str, Any]) -> str:
    blocks = []
    for attempt in run["search_attempts"]:
        chunk_cards = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>score {chunk['score']}</span><br>{html.escape(chunk['preview'])}<br><span class='hits'>term hits: {html.escape(', '.join(chunk.get('term_hits', [])) or '-')}</span></li>"
            for chunk in attempt["chunks"]
        ) or "<li>no chunks</li>"
        blocks.append(
            f"""
            <article class='attempt'>
              <div class='attempt-head'>
                <strong>{html.escape(attempt['node'])}</strong>
                <span>{attempt['elapsed_ms']:.2f} ms</span>
              </div>
              <div class='query'>{html.escape(attempt['query'])}</div>
              <ul>{chunk_cards}</ul>
            </article>
            """
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
          <h3>LangGraph flow</h3>
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
    note = (
        "실제 로컬 Qwen3 백엔드가 감지되지 않아 이번 round1은 mock 모드로 실행했다. 즉 Qwen3 모델명과 LangGraph 흐름은 Qwen3 기준으로 맞췄지만, 실제 토큰 생성 시간은 로컬 Ollama/llama.cpp를 붙였을 때와 다르다."
        if summary["provider_mode"] == "mock"
        else "로컬 OpenAI-compatible Qwen3 백엔드가 감지되어 실제 모델 호출로 round1을 실행했다."
    )
    run_html = "\n".join(render_run(run) for run in runs)
    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Round1 · Qwen3 Routed RAG Experiment</title>
  <style>
    :root {{ --bg:#09101d; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; --warn:#ffd479; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo","Noto Sans KR",sans-serif; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); line-height:1.6; }}
    .wrap {{ max-width:1280px; margin:0 auto; padding:28px 18px 80px; }}
    .hero,.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 12px 36px rgba(0,0,0,.22); }}
    .hero h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,52px); }}
    .hero p,.muted, li {{ color:var(--muted); }}
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
      <div class='eyebrow'>round1 · qwen3 routed rag</div>
      <h1>Qwen3 8B/14B 기준으로 10개 질문을 돌린 round1 실험</h1>
      <p>{html.escape(note)}</p>
      <div class='pillrow'>
        <span class='pill'>provider: {html.escape(summary['provider_mode'])}</span>
        <span class='pill'>router: {html.escape(summary['models']['router_8b'])}</span>
        <span class='pill'>large: {html.escape(summary['models']['large_14b'])}</span>
        <span class='pill'>doc: <a href='{html.escape(summary['doc_url'])}' target='_blank'>LangGraph agentic-rag</a></span>
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
    payload = asyncio.run(run_all())
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_HTML.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({
        "report_html": str(REPORT_HTML),
        "results_json": str(RESULTS_JSON),
        "summary": payload["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
