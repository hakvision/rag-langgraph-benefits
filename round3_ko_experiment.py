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
    evaluate_run,
    extract_chunks,
    fetch_source_document,
    grade_search,
    retrieve,
    tokenize,
)
from round2_ko_experiment import rerank_chunks

REPORT_HTML = REPO_ROOT / "round3_ko.html"
RESULTS_JSON = ARTIFACTS_DIR / "round3_ko_results.json"
SUMMARY_JSON = ARTIFACTS_DIR / "round3_ko_summary.json"
SOURCE_JSON = ARTIFACTS_DIR / "round3_ko_source_document.json"


def build_semantic_query(question: str) -> str:
    tokens = list(dict.fromkeys(tokenize(question)))
    expansions: list[str] = []
    q = question.lower()
    if "정의" in question or "무엇" in question:
        expansions.extend(["개념", "정의", "프로세스"])
    if "필요성" in question or "문제" in question:
        expansions.extend(["한계", "문제", "최신", "신뢰"])
    if "이점" in question:
        expansions.extend(["장점", "비용", "최신", "신뢰", "제어"])
    if "작동" in question or "흐름" in question or "단계" in question:
        expansions.extend(["외부 데이터", "관련 정보 검색", "프롬프트 확장", "업데이트"])
    if "검색" in question:
        expansions.extend(["연관성", "벡터", "데이터베이스", "시맨틱"])
    if "프롬프트" in question:
        expansions.extend(["보강", "컨텍스트", "정확한 답변"])
    if "차이" in question or "비교" in question:
        expansions.extend(["시맨틱 검색", "키워드 검색", "지식 준비"])
    if "aws" in q or "지원" in question:
        expansions.extend(["bedrock", "kendra", "sagemaker", "지식 기반"])
    merged = list(dict.fromkeys(tokens + expansions))
    return " ".join(merged[:24])


def reciprocal_rank_fusion(rank_lists: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for list_name, items in zip(["lexical", "semantic"], rank_lists):
        for rank, item in enumerate(items, start=1):
            chunk_id = item["chunk_id"]
            score = 1.0 / (k + rank)
            bucket = fused.setdefault(
                chunk_id,
                {
                    **item,
                    "rrf_score": 0.0,
                    "fusion_sources": [],
                    "source_ranks": {},
                    "base_scores": {},
                },
            )
            bucket["rrf_score"] += score
            bucket["fusion_sources"].append(list_name)
            bucket["source_ranks"][list_name] = rank
            bucket["base_scores"][list_name] = item["score"]
            if item["score"] > bucket.get("score", 0):
                bucket["score"] = item["score"]
                bucket["preview"] = item["preview"]
                bucket["text"] = item["text"]
                bucket["section"] = item["section"]
                bucket["term_hits"] = item.get("term_hits", [])
    merged = list(fused.values())
    merged.sort(key=lambda item: (-item["rrf_score"], -item.get("score", 0), item["chunk_id"]))
    return merged


def hybrid_retrieve(chunks: list[Any], lexical_query: str, semantic_query: str, model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lexical_candidates, lexical_stats = retrieve(chunks, lexical_query, model_name)
    semantic_candidates, semantic_stats = retrieve(chunks, semantic_query, model_name)
    fused_candidates = reciprocal_rank_fusion([lexical_candidates, semantic_candidates])
    limit = 6 if model_name == "8b" else 8
    selected = fused_candidates[:limit]
    stats = {
        "lexical_query": lexical_query,
        "semantic_query": semantic_query,
        "lexical_candidate_count": len(lexical_candidates),
        "semantic_candidate_count": len(semantic_candidates),
        "fused_candidate_count": len(fused_candidates),
        "top_rrf_score": round(selected[0]["rrf_score"], 4) if selected else 0,
        "retrieval_strategy": "hybrid lexical+dense-like heuristic with RRF",
        "lexical_top_score": lexical_stats.get("top_score", 0),
        "semantic_top_score": semantic_stats.get("top_score", 0),
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
        fused_cards = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>rrf {chunk['rrf_score']:.4f}</span><br><span class='hits'>sources: {html.escape(', '.join(chunk.get('fusion_sources', [])) or '-')}</span></li>"
            for chunk in attempt["hybrid_candidates"]
        ) or "<li>no fused candidates</li>"
        reranked_cards = "".join(
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
              <div class='query'>lexical: {html.escape(attempt['query'])}</div>
              <div class='query'>semantic: {html.escape(attempt['semantic_query'])}</div>
              <h4>hybrid fused candidates</h4>
              <ul>{fused_cards}</ul>
              <h4>reranked top chunks</h4>
              <ul>{reranked_cards}</ul>
            </article>
            """
        )
    return "\n".join(blocks)


def render_structure_section() -> str:
    return """
    <section class='card run'>
      <div class='eyebrow'>structure · round2 vs round3</div>
      <h2>Round2 / Round3 구조 비교</h2>
      <div class='grid two'>
        <article class='panel'>
          <h3>Round2</h3>
          <ol>
            <li>classify_8b</li>
            <li>search_8b 또는 search_14b</li>
            <li>rerank_{model}</li>
            <li>grade_results</li>
            <li>answer</li>
            <li>judge</li>
          </ol>
          <ul>
            <li>retrieval은 한 종류의 lexical scoring에 의존</li>
            <li>reranker는 잘하지만 후보군 recall ceiling이 낮을 수 있음</li>
            <li>즉 rerank 전 후보 pool 개선이 다음 병목</li>
          </ul>
        </article>
        <article class='panel'>
          <h3>Round3</h3>
          <ol>
            <li>classify_8b</li>
            <li>lexical retrieve</li>
            <li>semantic-like retrieve</li>
            <li>RRF fusion으로 hybrid candidate pool 생성</li>
            <li>rerank_{model}</li>
            <li>grade_results</li>
            <li>answer</li>
            <li>judge</li>
          </ol>
          <ul>
            <li>후보군 recall을 먼저 넓히고 그 위에 reranker를 태움</li>
            <li>hybrid retrieval + reranker의 분업 구조를 실험</li>
            <li>즉 round3는 reranker 앞단 후보군 품질 개선용 구조</li>
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
          <span class='pill'>hybrid retrieval</span>
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
        <h3>Hybrid retrieval + rerank attempts</h3>
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
  <title>Round3 KO · AWS RAG 문서 실험</title>
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
      <div class='eyebrow'>round3 · qwen3 routed rag · ko</div>
      <h1>Hybrid retrieval + reranker를 적용한 AWS 한국어 RAG round3</h1>
      <p>이번 round3는 round2의 reranker 구조를 유지하되, 앞단 retrieval을 <strong>lexical + semantic-like hybrid</strong>로 바꿨다. 먼저 두 종류의 후보를 만들고 RRF로 합친 다음 reranker를 태워서, 후보군 recall을 넓힌 뒤 순위를 다듬는 구조로 실험했다.</p>
      <div class='pillrow'>
        <span class='pill'>provider: mock</span>
        <span class='pill'>router: qwen3:8b</span>
        <span class='pill'>large: qwen3:14b</span>
        <span class='pill'>doc: <a href='{html.escape(summary['doc_url'])}' target='_blank'>{html.escape(summary['doc_title'])}</a></span>
        <span class='pill'>hybrid: lexical + semantic-like + RRF</span>
      </div>
      <div class='stats'>
        <article class='panel'><strong>{summary['question_count']}</strong><span>questions</span></article>
        <article class='panel'><strong>{summary['routes']['8b']}</strong><span>initial 8B routes</span></article>
        <article class='panel'><strong>{summary['routes']['14b']}</strong><span>initial 14B routes</span></article>
        <article class='panel'><strong>{summary['total_restarts']}</strong><span>total restarts</span></article>
        <article class='panel'><strong>{summary['avg_total_ms']:.2f} ms</strong><span>avg end-to-end time</span></article>
        <article class='panel'><strong>{summary['avg_judge_score']:.1f}</strong><span>avg judge score</span></article>
        <article class='panel'><strong>{summary['avg_rrf_score']:.4f}</strong><span>avg top RRF score</span></article>
        <article class='panel'><strong>{summary['avg_rerank_lift']:.2f}</strong><span>avg rerank lift</span></article>
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
    rrf_tops: list[float] = []

    while True:
        attempt_counts[current_model] += 1
        lexical_query = build_search_query(question, current_model, attempt_counts[current_model], retry_reason)
        semantic_query = build_semantic_query(question)

        t0 = time.perf_counter()
        hybrid_candidates, hybrid_stats = hybrid_retrieve(chunks, lexical_query, semantic_query, current_model)
        hybrid_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        flow.append(f"hybrid_search_{current_model}")
        logs.append({
            "node": f"hybrid_search_{current_model}",
            "message": f"{current_model.upper()} hybrid retrieval 완료",
            "payload": {**hybrid_stats, "selected_chunk_ids": [item['chunk_id'] for item in hybrid_candidates], "elapsed_ms": hybrid_elapsed},
        })
        node_timings.append({"node": f"hybrid_search_{current_model}", "elapsed_ms": hybrid_elapsed, "details": hybrid_stats})
        rrf_tops.append(hybrid_stats.get("top_rrf_score", 0))

        t0 = time.perf_counter()
        reranked_chunks, rerank_stats = rerank_chunks(question, hybrid_candidates, current_model)
        rerank_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_chunks = reranked_chunks
        flow.append(f"rerank_{current_model}")
        rerank_lift = (reranked_chunks[0]["rerank_score"] - reranked_chunks[0].get("score", 0)) if reranked_chunks else 0.0
        rerank_lifts.append(rerank_lift)
        logs.append({
            "node": f"rerank_{current_model}",
            "message": f"{current_model.upper()} reranker가 hybrid 후보를 재정렬함",
            "payload": {"stats": rerank_stats, "selected_chunk_ids": [item['chunk_id'] for item in reranked_chunks], "top_rerank_lift": rerank_lift, "elapsed_ms": rerank_elapsed},
        })
        node_timings.append({"node": f"rerank_{current_model}", "elapsed_ms": rerank_elapsed, "details": {**rerank_stats, "top_rerank_lift": rerank_lift}})
        search_attempts.append({
            "node": f"hybrid_search_{current_model}+rerank_{current_model}",
            "elapsed_ms": round(hybrid_elapsed + rerank_elapsed, 2),
            "attempt": attempt_counts[current_model],
            "query": lexical_query,
            "semantic_query": semantic_query,
            "hybrid_candidates": hybrid_candidates,
            "chunks": reranked_chunks,
        })
        explanation.append(
            f"hybrid_search_{current_model}가 lexical/semantic-like 후보를 합친 뒤 RRF로 묶었고, rerank_{current_model}가 그중 {', '.join(item['chunk_id'] for item in reranked_chunks[:4]) or '청크 없음'} 순서로 다시 정렬했다."
        )

        grade_stats = {
            "top_score": reranked_chunks[0]["rerank_score"] if reranked_chunks else 0,
            "distinct_sections": len({item['section'] for item in reranked_chunks}),
        }
        t0 = time.perf_counter()
        action, quality, message = grade_search(question, reranked_chunks, grade_stats, current_model)
        grade_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_quality = quality
        flow.append(f"grade_results → {action}")
        logs.append({"node": "grade_results", "message": message, "payload": {"action": action, "quality": quality, "elapsed_ms": grade_elapsed}})
        node_timings.append({"node": "grade_results", "elapsed_ms": grade_elapsed, "details": {"action": action, "quality": quality}})
        explanation.append(f"grade_results가 hybrid+rereank 결과를 보고 '{quality['reason']}'로 판단해 {action}을 선택했다.")

        if action == "answer":
            break
        if action == "escalate_to_14b":
            restart_count += 1
            retry_reason = quality["reason"]
            current_model = "14b"
            flow.append("route_upgrade 8b→14b")
            logs.append({"node": "route_upgrade", "message": "품질 게이트가 14B 승격을 요청함", "payload": {"reason": retry_reason}})
            node_timings.append({"node": "route_upgrade", "elapsed_ms": 0.0, "details": {"from": "8b", "to": "14b", "reason": retry_reason}})
            explanation.append("8B 경로의 hybrid retrieval 결과가 부족해서 14B hybrid+rereank 경로로 승격했다.")
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
    explanation.append(f"마지막에는 {current_model.upper()} answer 단계가 hybrid+rereanked top chunks를 근거로 응답을 만들었다.")
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
        "rrf_top_avg": round(sum(rrf_tops) / max(len(rrf_tops), 1), 4),
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
        "round": "round3-ko",
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
        "avg_rrf_score": round(sum(run['rrf_top_avg'] for run in runs) / max(len(runs), 1), 4),
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
