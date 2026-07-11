from __future__ import annotations

import html
import json
import textwrap
import time
from dataclasses import asdict
from typing import Any, Literal

from round1_ko_experiment import (
    ARTIFACTS_DIR,
    QUESTIONS,
    REPO_ROOT,
    SOURCE_TITLE,
    build_search_query,
    classify_question,
    clean_excerpt,
    evaluate_run,
    extract_chunks,
    fetch_source_document,
    grade_search,
)
from round3_ko_experiment import hybrid_retrieve, rerank_chunks

REPORT_HTML = REPO_ROOT / "round4_ko.html"
RESULTS_JSON = ARTIFACTS_DIR / "round4_ko_results.json"
SUMMARY_JSON = ARTIFACTS_DIR / "round4_ko_summary.json"
SOURCE_JSON = ARTIFACTS_DIR / "round4_ko_source_document.json"


def build_answer(question_item: dict[str, Any], final_model: str, chunks: list[dict[str, Any]], quality: dict[str, Any], mode: Literal["draft", "revised"] = "draft") -> str:
    question = question_item["question"]
    category = question_item["category"]
    citations = ", ".join(chunk["chunk_id"] for chunk in chunks[:3]) or "근거 없음"
    key_points = [clean_excerpt(item["text"]) for item in chunks[:3]]

    direct_map = {
        "definition": "RAG는 응답 생성 전에 외부의 신뢰할 수 있는 지식 소스를 참조하도록 LLM 출력을 최적화하는 방식이며, 모델 재학습 없이 정확성과 유용성을 높이는 접근이다.",
        "importance": "AWS 문서 기준 RAG의 필요성은 LLM의 허위정보, 오래된 정보, 신뢰하기 어려운 출처 문제를 줄이고 더 통제된 답변을 만들기 위해서다.",
        "benefits": "문서 기준 핵심 이점은 비용 효율적인 구현, 최신 정보 반영, 사용자 신뢰 강화, 개발자 제어 강화다.",
        "cost": "AWS 문서는 파운데이션 모델을 다시 학습시키는 대신 외부 지식을 연결해 새 데이터를 주입하므로 컴퓨팅 및 재정 비용을 크게 줄일 수 있다고 본다.",
        "trust": "RAG는 정확한 정보를 출처와 함께 제시하고 사용자가 원문을 직접 확인할 수 있게 해 신뢰와 확신을 높인다.",
        "workflow": "문서는 외부 데이터 생성, 관련 정보 검색, LLM 프롬프트 확장, 외부 데이터 업데이트 흐름으로 RAG 작동 방식을 설명한다.",
        "retrieval": "관련 정보 검색 단계에서는 사용자 쿼리를 벡터 표현으로 바꾸고 벡터 데이터베이스와 매칭해 연관성 높은 문서나 구절을 찾는다.",
        "prompting": "프롬프트 확장 단계는 검색된 관련 데이터를 사용자 입력에 덧붙여 LLM이 더 정확하고 근거 있는 답변을 생성하게 하기 위해 필요하다.",
        "comparison": "RAG는 외부 지식을 붙여 답변을 생성하는 전체 접근이고, 시맨틱 검색은 그 성능을 높이기 위해 관련 구절을 더 정확히 찾는 검색 기술이다.",
        "aws-support": "AWS는 Amazon Bedrock 지식 기반, Amazon Kendra, SageMaker JumpStart를 통해 데이터 연결, 엔터프라이즈 검색, 빠른 배포를 지원한다고 설명한다.",
    }
    direct_answer = direct_map[category]

    if mode == "draft":
        lines = [
            f"결론: {direct_answer}",
            f"질문: {question}",
            f"근거 청크: {citations}",
            f"모드: draft-{final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}",
            "핵심 근거:",
        ]
        for idx, point in enumerate(key_points, start=1):
            lines.append(f"- 근거 {idx}: {point}")
        return "\n".join(lines)

    # revised answer: explicit direct answer + why + evidence + limitation
    why_map = {
        "definition": "핵심은 외부 지식을 참조한다는 점과, 모델 재학습 없이도 특정 도메인 정보를 붙일 수 있다는 점이다.",
        "importance": "즉 RAG는 허위정보, 오래된 정보, 불분명한 출처 문제를 줄이면서 최신성과 통제력을 보강한다.",
        "benefits": "정리하면 비용, 최신성, 신뢰, 제어라는 네 축에서 가치가 난다.",
        "cost": "따라서 대규모 재학습보다 훨씬 비용 효율적인 구현 경로가 된다.",
        "trust": "사용자는 출처와 원문 확인 경로가 있어 결과를 검증할 수 있다.",
        "workflow": "이 순서 덕분에 검색된 외부 데이터가 생성 단계에 직접 연결된다.",
        "retrieval": "여기서 핵심은 벡터화와 벡터 데이터베이스 매칭으로 연관성 높은 구절을 찾는 점이다.",
        "prompting": "즉 관련 데이터를 프롬프트에 붙여 답변 정확성과 groundedness를 높인다.",
        "comparison": "그래서 RAG는 상위 아키텍처이고, 시맨틱 검색은 그 안의 retrieval 품질을 끌어올리는 하위 기술이다.",
        "aws-support": "그래서 데이터 연결, 검색, 배포 가속을 AWS 관리형 서비스로 나눠 지원하는 구조다.",
    }
    lines = [
        f"직답: {direct_answer}",
        f"왜 이렇게 말하나: {why_map[category]}",
        f"질문: {question}",
        f"근거 청크: {citations}",
        f"모드: revised-{final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}",
        "핵심 포인트:",
    ]
    for idx, point in enumerate(key_points, start=1):
        lines.append(f"- 포인트 {idx}: {point}")
    lines.append("한계: 이번 round는 실백엔드가 아니라 mock 파이프라인이므로 모델 추론 성능 자체를 뜻하지는 않는다.")
    return "\n".join(lines)


def judge_answer(question_item: dict[str, Any], answer: str, chunks: list[dict[str, Any]], quality: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    evaluation = evaluate_run(question_item, answer, chunks, quality)
    if evaluation["total_score"] >= 70:
        return "finish", evaluation, "answer judge 통과: 응답이 정답 기준을 충분히 충족함."
    if not chunks:
        return "finish", evaluation, "answer judge 종료: 근거 청크가 부족해서 재생성보다 종료가 낫다고 판단함."
    return "regenerate_once", evaluation, "answer judge가 응답 형식/키워드 충족이 약하다고 판단해 한 번 더 생성함."


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
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>hybrid {chunk['score']}</span> <span class='hits'>lex {html.escape(str(chunk.get('lexical_rank')))} / sem {html.escape(str(chunk.get('semantic_rank')))}</span></li>"
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
              <h4>hybrid retrieval candidates</h4>
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
      <div class='eyebrow'>structure · round3 vs round4</div>
      <h2>Round3 / Round4 구조 비교</h2>
      <div class='grid two'>
        <article class='panel'>
          <h3>Round3</h3>
          <ol>
            <li>classify_8b</li>
            <li>hybrid retrieval</li>
            <li>reranker</li>
            <li>grade_results</li>
            <li>필요 시 14B 승격</li>
            <li>answer / judge</li>
          </ol>
          <ul>
            <li>retrieval quality와 answer quality가 한 덩어리로 보임</li>
            <li>답변 형식이 약해도 retrieval 문제처럼 보일 수 있음</li>
          </ul>
        </article>
        <article class='panel'>
          <h3>Round4</h3>
          <ol>
            <li>classify_8b</li>
            <li>hybrid_retrieve + rerank</li>
            <li>judge_retrieval</li>
            <li>필요 시 14B 승격</li>
            <li>answer_draft 생성</li>
            <li>judge_answer</li>
            <li>약하면 answer_revise 1회</li>
            <li>final judge 저장</li>
          </ol>
          <ul>
            <li>retrieval 문제와 answer formatting 문제를 분리해서 본다</li>
            <li>round4는 retrieval judge / answer judge 분리 효과를 보는 구조</li>
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
          <span class='pill'>answer regenerations: {run['answer_regeneration_count']}</span>
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
          <h3>Retrieval judge</h3>
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
  <title>Round4 KO · AWS RAG 문서 실험</title>
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
      <div class='eyebrow'>round4 · qwen3 routed rag · ko</div>
      <h1>Retrieval judge / answer judge를 분리한 AWS 한국어 RAG round4</h1>
      <p>이번 round4는 round3의 hybrid retrieval과 reranker를 유지하면서, <strong>judge_retrieval</strong>과 <strong>judge_answer</strong>를 분리했다. 검색 품질과 답변 형식 문제를 따로 보고, 답변 judge가 약하다고 판단하면 한 번 더 정리해서 내보내는 구조다. 실백엔드는 없어서 mock 실험이지만, 실제 AWS 한국어 원문 청크를 다시 가져와 새 라운드를 실행했다.</p>
      <div class='pillrow'>
        <span class='pill'>provider: mock</span>
        <span class='pill'>router: qwen3:8b</span>
        <span class='pill'>large: qwen3:14b</span>
        <span class='pill'>doc: <a href='{html.escape(summary['doc_url'])}' target='_blank'>{html.escape(summary['doc_title'])}</a></span>
        <span class='pill'>retrieval judge + answer judge</span>
      </div>
      <div class='stats'>
        <article class='panel'><strong>{summary['question_count']}</strong><span>questions</span></article>
        <article class='panel'><strong>{summary['routes']['8b']}</strong><span>initial 8B routes</span></article>
        <article class='panel'><strong>{summary['routes']['14b']}</strong><span>initial 14B routes</span></article>
        <article class='panel'><strong>{summary['total_restarts']}</strong><span>total restarts</span></article>
        <article class='panel'><strong>{summary['answer_regenerations']}</strong><span>answer regenerations</span></article>
        <article class='panel'><strong>{summary['avg_total_ms']:.2f} ms</strong><span>avg end-to-end time</span></article>
        <article class='panel'><strong>{summary['avg_judge_score']:.1f}</strong><span>avg judge score</span></article>
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
    answer_regeneration_count = 0

    while True:
        attempt_counts[current_model] += 1
        query = build_search_query(question, current_model, attempt_counts[current_model], retry_reason)

        t0 = time.perf_counter()
        initial_candidates, base_stats = hybrid_retrieve(chunks, question, query, current_model)
        retrieval_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        flow.append(f"hybrid_retrieve_{current_model}")
        logs.append({
            "node": f"hybrid_retrieve_{current_model}",
            "message": f"{current_model.upper()} hybrid retrieval 후보 수집 완료",
            "payload": {"attempt": attempt_counts[current_model], "query": query, "stats": base_stats, "selected_chunk_ids": [item['chunk_id'] for item in initial_candidates], "elapsed_ms": retrieval_elapsed},
        })
        node_timings.append({"node": f"hybrid_retrieve_{current_model}", "elapsed_ms": retrieval_elapsed, "details": {"attempt": attempt_counts[current_model], "query": query, **base_stats}})

        t0 = time.perf_counter()
        reranked_chunks, rerank_stats = rerank_chunks(question, initial_candidates, current_model)
        rerank_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_chunks = reranked_chunks
        flow.append(f"rerank_{current_model}")
        rerank_lift = (reranked_chunks[0]["rerank_score"] - reranked_chunks[0]["base_score"]) if reranked_chunks else 0.0
        rerank_lifts.append(rerank_lift)
        logs.append({
            "node": f"rerank_{current_model}",
            "message": f"{current_model.upper()} reranker가 hybrid 후보를 재정렬함",
            "payload": {"stats": rerank_stats, "selected_chunk_ids": [item['chunk_id'] for item in reranked_chunks], "top_rerank_lift": rerank_lift, "elapsed_ms": rerank_elapsed},
        })
        node_timings.append({"node": f"rerank_{current_model}", "elapsed_ms": rerank_elapsed, "details": {**rerank_stats, "top_rerank_lift": rerank_lift}})
        search_attempts.append({
            "node": f"hybrid_retrieve_{current_model}+rerank_{current_model}",
            "elapsed_ms": round(retrieval_elapsed + rerank_elapsed, 2),
            "attempt": attempt_counts[current_model],
            "query": query,
            "initial_candidates": initial_candidates,
            "chunks": reranked_chunks,
        })
        explanation.append(
            f"hybrid retrieval이 후보를 모으고 rerank_{current_model}가 {', '.join(item['chunk_id'] for item in reranked_chunks[:4]) or '청크 없음'} 순으로 정렬했다."
        )

        grade_stats = {
            "top_score": reranked_chunks[0]["rerank_score"] if reranked_chunks else 0,
            "distinct_sections": len({item['section'] for item in reranked_chunks}),
        }
        t0 = time.perf_counter()
        action, quality, message = grade_search(question, reranked_chunks, grade_stats, current_model)
        grade_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_quality = quality
        flow.append(f"judge_retrieval → {action}")
        logs.append({"node": "judge_retrieval", "message": message, "payload": {"action": action, "quality": quality, "elapsed_ms": grade_elapsed}})
        node_timings.append({"node": "judge_retrieval", "elapsed_ms": grade_elapsed, "details": {"action": action, "quality": quality}})
        explanation.append(f"judge_retrieval이 '{quality['reason']}'로 판단해 {action}을 선택했다.")

        if action == "answer":
            break
        if action == "escalate_to_14b":
            restart_count += 1
            retry_reason = quality["reason"]
            current_model = "14b"
            flow.append("route_upgrade 8b→14b")
            logs.append({"node": "route_upgrade", "message": "retrieval judge가 14B 승격을 요청함", "payload": {"reason": retry_reason}})
            node_timings.append({"node": "route_upgrade", "elapsed_ms": 0.0, "details": {"from": "8b", "to": "14b", "reason": retry_reason}})
            explanation.append("retrieval judge가 8B 검색 근거가 얕다고 봐서 14B 경로로 올렸다.")
            continue
        break

    t0 = time.perf_counter()
    draft_answer = build_answer(question_item, current_model, final_chunks, final_quality, mode="draft")
    answer_elapsed = round((time.perf_counter() - t0) * 1000, 2)
    flow.append("answer_draft")
    logs.append({"node": "answer_draft", "message": "초안 답변 생성", "payload": {"final_model": current_model, "elapsed_ms": answer_elapsed, "answer_preview": textwrap.shorten(draft_answer, width=240, placeholder='…')}})
    node_timings.append({"node": "answer_draft", "elapsed_ms": answer_elapsed, "details": {"final_model": current_model}})

    t0 = time.perf_counter()
    answer_action, draft_evaluation, answer_message = judge_answer(question_item, draft_answer, final_chunks, final_quality)
    answer_judge_elapsed = round((time.perf_counter() - t0) * 1000, 2)
    flow.append(f"judge_answer → {answer_action}")
    logs.append({"node": "judge_answer", "message": answer_message, "payload": {"action": answer_action, "evaluation": draft_evaluation, "elapsed_ms": answer_judge_elapsed}})
    node_timings.append({"node": "judge_answer", "elapsed_ms": answer_judge_elapsed, "details": {"action": answer_action, "evaluation": draft_evaluation}})
    explanation.append(f"judge_answer가 초안 답변을 보고 {answer_action}을 결정했다.")

    final_answer = draft_answer
    final_evaluation = draft_evaluation
    if answer_action == "regenerate_once":
        answer_regeneration_count = 1
        t0 = time.perf_counter()
        final_answer = build_answer(question_item, current_model, final_chunks, final_quality, mode="revised")
        revise_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        flow.append("answer_revise")
        logs.append({"node": "answer_revise", "message": "answer judge 요청으로 답변을 한 번 더 정리함", "payload": {"elapsed_ms": revise_elapsed, "answer_preview": textwrap.shorten(final_answer, width=240, placeholder='…')}})
        node_timings.append({"node": "answer_revise", "elapsed_ms": revise_elapsed, "details": {"regeneration": 1}})

        t0 = time.perf_counter()
        final_evaluation = evaluate_run(question_item, final_answer, final_chunks, final_quality)
        final_judge_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        flow.append("judge_answer_final")
        logs.append({"node": "judge_answer_final", "message": "재생성 후 최종 채점 완료", "payload": {"evaluation": final_evaluation, "elapsed_ms": final_judge_elapsed}})
        node_timings.append({"node": "judge_answer_final", "elapsed_ms": final_judge_elapsed, "details": final_evaluation})
        explanation.append("초안 점수가 낮아 answer를 한 번 더 직답형으로 정리한 뒤 최종 judge를 다시 기록했다.")

    logs.append({"node": "judge", "message": "미리 정한 정답 기준으로 최종 응답 평가 완료", "payload": final_evaluation})
    explanation.append(f"마지막에는 {current_model.upper()} answer 단계 결과를 최종 judge에 저장했다.")
    explanation.append(final_evaluation["judge_comment"])

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
        "answer_regeneration_count": answer_regeneration_count,
        "final_model": current_model,
        "quality": final_quality,
        "expected_answer": final_evaluation["gold_answer"],
        "evaluation": final_evaluation,
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
        "round": "round4-ko",
        "doc_title": SOURCE_TITLE,
        "doc_url": doc_url,
        "chunk_count": len(chunks),
        "provider_mode": "mock",
        "backend_available": False,
        "models": {"router_8b": "qwen3:8b", "large_14b": "qwen3:14b"},
        "question_count": len(runs),
        "routes": {"8b": sum(1 for run in runs if run['route_decision'] == '8b'), "14b": sum(1 for run in runs if run['route_decision'] == '14b')},
        "total_restarts": sum(run['restart_count'] for run in runs),
        "answer_regenerations": sum(run['answer_regeneration_count'] for run in runs),
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
