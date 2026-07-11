from __future__ import annotations

import html
import json
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from round1_ko_experiment import (
    ARTIFACTS_DIR,
    QUESTIONS,
    REPO_ROOT,
    SOURCE_TITLE,
    build_answer,
    build_search_query,
    classify_question,
    clean_excerpt,
    evaluate_run,
    extract_chunks,
    fetch_source_document,
    grade_search,
    retrieve,
    tokenize,
)

REPORT_HTML = REPO_ROOT / "round2_ko.html"
RESULTS_JSON = ARTIFACTS_DIR / "round2_ko_results.json"
SUMMARY_JSON = ARTIFACTS_DIR / "round2_ko_summary.json"
SOURCE_JSON = ARTIFACTS_DIR / "round2_ko_source_document.json"


def rerank_chunks(question: str, candidates: list[dict[str, Any]], model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_tokens = set(tokenize(question))
    reranked: list[dict[str, Any]] = []
    for idx, chunk in enumerate(candidates):
        chunk_tokens = set(tokenize(chunk["section"] + " " + chunk["text"]))
        overlap = sorted(question_tokens & chunk_tokens)
        overlap_score = len(overlap) * 1.7
        section_bonus = 0.0
        question_lower = question.lower()
        section_lower = chunk["section"].lower()
        if "정의" in question or "무엇" in question:
            section_bonus += 1.4 if "무엇" in section_lower else 0.0
        if "이점" in question:
            section_bonus += 1.6 if "이점" in section_lower else 0.0
        if "작동" in question or "단계" in question or "흐름" in question:
            section_bonus += 1.6 if "작동" in section_lower else 0.0
        if "검색" in question and "관련 정보 검색" in chunk["section"]:
            section_bonus += 2.4
        if "프롬프트" in question and "프롬프트 확장" in chunk["section"]:
            section_bonus += 2.4
        if "차이" in question and "차이점" in section_lower:
            section_bonus += 2.0
        if "aws" in question_lower and "지원" in section_lower:
            section_bonus += 2.2
        rank_bonus = max(0.0, (len(candidates) - idx) * 0.08)
        rerank_score = round(chunk["score"] + overlap_score + section_bonus + rank_bonus, 3)
        reranked.append({
            **chunk,
            "rerank_score": rerank_score,
            "rerank_overlap": overlap,
            "base_score": chunk["score"],
        })

    reranked.sort(key=lambda item: (-item["rerank_score"], -item["score"], item["chunk_id"]))
    final_limit = 4 if model_name == "8b" else 6
    selected = reranked[:final_limit]
    stats = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "top_rerank_score": selected[0]["rerank_score"] if selected else 0,
        "rerank_strategy": "question-token overlap + section-aware heuristic reranker",
    }
    return selected, stats


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
        before_cards = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>base {chunk['score']}</span></li>"
            for chunk in attempt["initial_candidates"]
        ) or "<li>no candidates</li>"
        after_cards = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>rerank {chunk['rerank_score']}</span><br>{html.escape(chunk['preview'])}<br><span class='hits'>overlap: {html.escape(', '.join(chunk.get('rerank_overlap', [])) or '-')}</span></li>"
            for chunk in attempt["chunks"]
        ) or "<li>no selected chunks</li>"
        blocks.append(
            f"""
            <article class='attempt'>
              <div class='attempt-head'>
                <strong>{html.escape(attempt['node'])}</strong>
                <span>{attempt['elapsed_ms']:.2f} ms</span>
              </div>
              <div class='query'>{html.escape(attempt['query'])}</div>
              <h4>retrieval candidates</h4>
              <ul>{before_cards}</ul>
              <h4>reranked top chunks</h4>
              <ul>{after_cards}</ul>
            </article>
            """
        )
    return "\n".join(blocks)


def render_structure_section() -> str:
    return """
    <section class='card run'>
      <div class='eyebrow'>structure · round1 vs round2</div>
      <h2>Round1 / Round2 구조 비교</h2>
      <div class='grid two'>
        <article class='panel'>
          <h3>Round1</h3>
          <ol>
            <li>classify_8b</li>
            <li>search_8b 또는 search_14b</li>
            <li>grade_results</li>
            <li>필요 시 14B 승격</li>
            <li>answer</li>
            <li>judge</li>
          </ol>
          <ul>
            <li>retrieval 1단계만 있고 reranker 없음</li>
            <li>top score와 coverage 위주로 품질을 판정</li>
            <li>baseline 확인용 구조</li>
          </ul>
        </article>
        <article class='panel'>
          <h3>Round2</h3>
          <ol>
            <li>classify_8b</li>
            <li>search_8b 또는 search_14b로 넓게 후보 수집</li>
            <li>rerank_{model}: 질문-청크 overlap + section-aware reranker 적용</li>
            <li>grade_results</li>
            <li>필요 시 14B 승격 후 다시 retrieval+rereank</li>
            <li>answer</li>
            <li>judge</li>
          </ol>
          <ul>
            <li>retrieval과 reranking을 분리해서 구조를 더 명확하게 봄</li>
            <li>질문 의도와 맞는 section을 앞으로 당겨 score 개선을 노림</li>
            <li>즉 round2는 reranker 추가 효과를 보는 실험 구조</li>
          </ul>
        </article>
      </div>
    </section>
    """


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
          <span class='pill'>reranker applied</span>
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
          <h3>Expected answer</h3>
          <pre>{html.escape(run['expected_answer'])}</pre>
          <h3>Judge score</h3>
          <pre>{html.escape(json.dumps(run['evaluation'], ensure_ascii=False, indent=2))}</pre>
          <h3>Quality gate</h3>
          <pre>{html.escape(json.dumps(run['quality'], ensure_ascii=False, indent=2))}</pre>
        </article>
      </div>
      <article class='panel'>
        <h3>Search + rerank attempts</h3>
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
  <title>Round2 KO · AWS RAG 문서 실험</title>
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
    h2,h3,h4 {{ margin:0 0 10px; }}
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
      <div class='eyebrow'>round2 · qwen3 routed rag · ko</div>
      <h1>Reranker를 추가한 AWS 한국어 RAG round2</h1>
      <p>이번 round2는 round1의 baseline retrieval 위에 <strong>reranker 단계</strong>를 따로 넣어, 먼저 넓게 후보를 뽑고 그다음 질문-청크 정합성으로 다시 정렬하는 구조로 분리했다. 여전히 실백엔드는 없어서 mock 실험이지만, 구조 차이와 judged score 변화를 보기 좋게 만들었다.</p>
      <div class='pillrow'>
        <span class='pill'>provider: mock</span>
        <span class='pill'>router: qwen3:8b</span>
        <span class='pill'>large: qwen3:14b</span>
        <span class='pill'>doc: <a href='{html.escape(summary['doc_url'])}' target='_blank'>{html.escape(summary['doc_title'])}</a></span>
        <span class='pill'>reranker: heuristic section-aware reranker</span>
      </div>
      <div class='stats'>
        <article class='panel'><strong>{summary['question_count']}</strong><span>questions</span></article>
        <article class='panel'><strong>{summary['routes']['8b']}</strong><span>initial 8B routes</span></article>
        <article class='panel'><strong>{summary['routes']['14b']}</strong><span>initial 14B routes</span></article>
        <article class='panel'><strong>{summary['total_restarts']}</strong><span>total restarts</span></article>
        <article class='panel'><strong>{summary['avg_total_ms']:.2f} ms</strong><span>avg end-to-end time</span></article>
        <article class='panel'><strong>{summary['avg_judge_score']:.1f}</strong><span>avg judge score</span></article>
        <article class='panel'><strong>{summary['avg_rerank_lift']:.2f}</strong><span>avg rerank lift</span></article>
        <article class='panel'><strong>{summary['score_bands']['good']}</strong><span>good verdicts</span></article>
      </div>
    </section>
    {render_structure_section()}
    {run_html}
  </main>
</body>
</html>
"""


def run_one(question_item: dict[str, Any], chunks: list[Any]) -> dict[str, Any]:
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
    rerank_lifts: list[float] = []

    while True:
        attempt_counts[current_model] += 1
        query = build_search_query(question, current_model, attempt_counts[current_model], retry_reason)

        t0 = time.perf_counter()
        initial_candidates, base_stats = retrieve(chunks, query, current_model)
        retrieval_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        flow.append(f"search_{current_model}")
        logs.append({
            "node": f"search_{current_model}",
            "message": f"{current_model.upper()} 검색 후보 수집 완료",
            "payload": {"attempt": attempt_counts[current_model], "query": query, "stats": base_stats, "selected_chunk_ids": [item['chunk_id'] for item in initial_candidates], "elapsed_ms": retrieval_elapsed},
        })
        node_timings.append({"node": f"search_{current_model}", "elapsed_ms": retrieval_elapsed, "details": {"attempt": attempt_counts[current_model], "query": query, **base_stats}})

        t0 = time.perf_counter()
        reranked_chunks, rerank_stats = rerank_chunks(question, initial_candidates, current_model)
        rerank_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_chunks = reranked_chunks
        flow.append(f"rerank_{current_model}")
        rerank_lift = (reranked_chunks[0]["rerank_score"] - reranked_chunks[0]["base_score"]) if reranked_chunks else 0.0
        rerank_lifts.append(rerank_lift)
        logs.append({
            "node": f"rerank_{current_model}",
            "message": f"{current_model.upper()} reranker가 후보를 재정렬함",
            "payload": {"stats": rerank_stats, "selected_chunk_ids": [item['chunk_id'] for item in reranked_chunks], "top_rerank_lift": rerank_lift, "elapsed_ms": rerank_elapsed},
        })
        node_timings.append({"node": f"rerank_{current_model}", "elapsed_ms": rerank_elapsed, "details": {**rerank_stats, "top_rerank_lift": rerank_lift}})
        search_attempts.append({
            "node": f"search_{current_model}+rerank_{current_model}",
            "elapsed_ms": round(retrieval_elapsed + rerank_elapsed, 2),
            "attempt": attempt_counts[current_model],
            "query": query,
            "initial_candidates": initial_candidates,
            "chunks": reranked_chunks,
        })
        explanation.append(
            f"search_{current_model}가 먼저 후보를 모으고 rerank_{current_model}가 질문 overlap과 section 힌트로 {', '.join(item['chunk_id'] for item in reranked_chunks[:4]) or '청크 없음'} 순서로 다시 정렬했다."
        )

        grade_stats = {
            "top_score": reranked_chunks[0]["rerank_score"] if reranked_chunks else 0,
            "distinct_sections": len({item["section"] for item in reranked_chunks}),
        }
        t0 = time.perf_counter()
        action, quality, message = grade_search(question, reranked_chunks, grade_stats, current_model)
        grade_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_quality = quality
        flow.append(f"grade_results → {action}")
        logs.append({"node": "grade_results", "message": message, "payload": {"action": action, "quality": quality, "elapsed_ms": grade_elapsed}})
        node_timings.append({"node": "grade_results", "elapsed_ms": grade_elapsed, "details": {"action": action, "quality": quality}})
        explanation.append(f"grade_results가 reranked top chunks를 보고 '{quality['reason']}'로 판단해 {action}을 선택했다.")

        if action == "answer":
            break
        if action == "escalate_to_14b":
            restart_count += 1
            retry_reason = quality["reason"]
            current_model = "14b"
            flow.append("route_upgrade 8b→14b")
            logs.append({"node": "route_upgrade", "message": "품질 게이트가 14B 승격을 요청함", "payload": {"reason": retry_reason}})
            node_timings.append({"node": "route_upgrade", "elapsed_ms": 0.0, "details": {"from": "8b", "to": "14b", "reason": retry_reason}})
            explanation.append("8B 검색 결과가 부족해서 14B retrieval+rereank 경로로 승격했다.")
            continue
        break

    t0 = time.perf_counter()
    final_answer = build_answer(question, current_model, final_chunks, final_quality)
    answer_elapsed = round((time.perf_counter() - t0) * 1000, 2)
    evaluation = evaluate_run(question_item, final_answer, final_chunks, final_quality)
    flow.append("answer")
    logs.append({"node": "answer", "message": "최종 답변 생성", "payload": {"final_model": current_model, "elapsed_ms": answer_elapsed, "answer_preview": textwrap.shorten(final_answer, width=240, placeholder='…')}})
    logs.append({"node": "judge", "message": "미리 정한 정답 기준으로 응답 평가 완료", "payload": evaluation})
    node_timings.append({"node": "answer", "elapsed_ms": answer_elapsed, "details": {"final_model": current_model}})
    explanation.append(f"마지막에는 {current_model.upper()} answer 단계가 reranked top chunks를 근거로 응답을 만들었다.")
    explanation.append(evaluation["judge_comment"])

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
        "expected_answer": evaluation["gold_answer"],
        "evaluation": evaluation,
        "final_answer": final_answer,
        "logs": logs,
        "node_timings": node_timings,
        "flow": flow,
        "explanation": explanation,
        "rerank_lift_avg": round(sum(rerank_lifts) / max(len(rerank_lifts), 1), 3),
        "total_ms": round((time.perf_counter() - run_start) * 1000, 2),
    }


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
        "round": "round2-ko",
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
        "avg_judge_score": round(sum(run['evaluation']['total_score'] for run in runs) / max(len(runs), 1), 1),
        "avg_rerank_lift": round(sum(run['rerank_lift_avg'] for run in runs) / max(len(runs), 1), 2),
        "score_bands": {
            "good": sum(1 for run in runs if run['evaluation']['verdict'] == '좋음'),
            "okay": sum(1 for run in runs if run['evaluation']['verdict'] == '무난'),
            "weak": sum(1 for run in runs if run['evaluation']['verdict'] == '아쉬움'),
            "poor": sum(1 for run in runs if run['evaluation']['verdict'] == '미흡'),
        },
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
