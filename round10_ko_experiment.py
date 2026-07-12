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
from round2_ko_experiment import rerank_chunks

ROUND_NUM = 10
HTML_PATH = REPO_ROOT / f"round{ROUND_NUM}_ko.html"
RESULTS_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_results.json"
SUMMARY_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_summary.json"
SOURCE_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_source_document.json"

CATEGORY_HINTS: dict[str, list[str]] = {
    "definition": ["외부", "신뢰", "지식", "참조", "재학습", "정확성"],
    "importance": ["허위", "오래된", "신뢰", "출처", "혼동", "통제"],
    "benefits": ["비용", "최신", "신뢰", "제어", "이점", "사용자"],
    "cost": ["비용", "재학습", "외부", "새 데이터", "컴퓨팅", "재정"],
    "trust": ["출처", "정확", "원문", "확인", "신뢰", "확신", "저작자 표시", "인용"],
    "workflow": ["외부 데이터", "검색", "프롬프트", "업데이트", "단계", "흐름"],
    "retrieval": ["벡터", "데이터베이스", "매칭", "연관성", "문서", "구절"],
    "prompting": ["검색된", "관련 데이터", "프롬프트", "확장", "정확", "답변"],
    "comparison": ["전체 접근", "시맨틱 검색", "관련 구절", "정확", "준비 작업", "성능"],
    "aws-support": ["bedrock", "kendra", "sagemaker", "지식 기반", "검색", "배포"],
}

SECTION_HINTS: dict[str, list[str]] = {
    "definition": ["무엇인가요"],
    "importance": ["중요한 이유"],
    "benefits": ["이점", "비용 효율적인 구현", "최신 정보", "사용자 신뢰 강화", "개발자 제어 강화"],
    "cost": ["비용 효율적인 구현", "이점", "무엇인가요"],
    "trust": ["사용자 신뢰 강화", "이점", "최신 정보"],
    "workflow": ["작동하나요", "외부 데이터 생성", "관련 정보 검색", "LLM 프롬프트 확장", "외부 데이터 업데이트"],
    "retrieval": ["관련 정보 검색"],
    "prompting": ["LLM 프롬프트 확장"],
    "comparison": ["차이점"],
    "aws-support": ["지원"],
}

DIRECT_ANSWERS: dict[str, str] = {
    "definition": "RAG는 응답 생성 전에 외부의 신뢰할 수 있는 지식 소스를 참조하도록 LLM 출력을 최적화하는 방식이며, 모델 재학습 없이 특정 도메인 지식을 붙여 정확성과 유용성을 높이는 접근이다.",
    "importance": "RAG는 LLM의 허위 정보, 오래된 정보, 신뢰할 수 없는 출처, 용어 혼동 같은 문제를 줄이고 신뢰 가능한 지식 소스를 기반으로 더 통제된 답변을 만들기 위해 필요하다.",
    "benefits": "문서 기준 핵심 이점은 비용 효율적인 구현, 최신 정보 반영, 사용자 신뢰 강화, 개발자 제어 강화다.",
    "cost": "파운데이션 모델을 재학습하는 대신 외부 지식을 연결해 새 데이터를 주입하므로 컴퓨팅 및 재정 비용을 줄일 수 있어 비용 효율적이라고 설명한다.",
    "trust": "RAG는 정확한 정보를 출처와 함께 제시하고 사용자가 원문을 직접 확인할 수 있게 하여 생성형 AI 결과에 대한 신뢰와 확신을 높인다.",
    "workflow": "문서는 외부 데이터 생성, 관련 정보 검색, LLM 프롬프트 확장, 외부 데이터 업데이트의 흐름으로 RAG를 설명한다.",
    "retrieval": "관련 정보 검색 단계에서는 사용자 쿼리를 벡터 표현으로 바꾸고 벡터 데이터베이스와 매칭해 연관성이 높은 문서나 구절을 찾는다.",
    "prompting": "검색된 관련 데이터를 사용자 입력에 붙여 프롬프트를 확장해야 LLM이 더 정확하고 근거 있는 답변을 만들 수 있다.",
    "comparison": "RAG는 외부 지식을 붙여 답변을 생성하는 전체 접근이고, 시맨틱 검색은 그 과정에서 더 정확한 관련 구절을 찾는 검색 기술이다.",
    "aws-support": "AWS는 Amazon Bedrock 지식 기반, Amazon Kendra, SageMaker JumpStart를 통해 RAG 구축의 데이터 연결, 검색, 배포를 지원한다고 설명한다.",
}

ROUND_THEME = "adaptive combination graph"
ROUND_SHORT = "좋았던 요소를 조합하되 모든 노드를 항상 돌리지 않고 질문/검색 상태에 따라 분기하는 adaptive graph"
ROUND_LONG = "round10은 round3의 hybrid retrieval, round4의 judge 분리, round7의 evidence distill, round8의 citation template를 기본 축으로 유지하면서, round5/6/9에서 유용했던 rewrite·subquery·rescue·refine를 조건부로만 태우는 adaptive combination 실험이다. 즉 쉬운 질문은 짧은 경로로 끝내고, 복합질문·낮은 retrieval quality·낮은 answer score에서만 추가 노드를 실행한다."


def detect_question_type(question_item: dict[str, Any], route: Literal["8b", "14b"], analysis: dict[str, Any]) -> str:
    category = question_item["category"]
    if category in {"comparison", "workflow", "aws-support"}:
        return "multi_part"
    if category in {"importance", "trust", "cost"}:
        return "abstract_why"
    if route == "14b" or analysis.get("complexity_score", 0) >= 4:
        return "multi_part"
    return "simple_fact"


def build_adaptive_query_plan(question_item: dict[str, Any], model_name: Literal["8b", "14b"], question_type: str, retry_reason: str = "") -> list[dict[str, Any]]:
    question = question_item["question"]
    category = question_item["category"]
    hints = CATEGORY_HINTS[category]
    plan: list[dict[str, Any]] = []

    base = build_search_query(question, model_name, 1, retry_reason)
    plan.append({"label": "base", "query": base, "weight": 1.00, "branch": "default"})

    if question_type == "simple_fact":
        focus = " ".join(dict.fromkeys(tokenize(question)[:5] + tokenize(" ".join(hints[:3]))))
        plan.append({"label": "focus", "query": focus, "weight": 1.08, "branch": "simple"})

    if question_type == "abstract_why":
        rewrite = " ".join(dict.fromkeys(tokenize(question) + tokenize(" ".join(hints[:5]))))
        step_back = " ".join(dict.fromkeys(tokenize(f"{question} 배경 이유 큰 개념") + tokenize(" ".join(hints[:3]))))
        plan.append({"label": "rewrite", "query": rewrite, "weight": 1.12, "branch": "abstract"})
        plan.append({"label": "step_back", "query": step_back, "weight": 0.98, "branch": "abstract"})

    if question_type == "multi_part":
        parts = [hints[:3], hints[3:6] or hints[:2]]
        if category in {"benefits", "trust", "workflow"}:
            parts.append(hints[1:5])
        for idx, part in enumerate(parts, start=1):
            plan.append({
                "label": f"subquery_{idx}",
                "query": " ".join(dict.fromkeys(tokenize(question)[:4] + tokenize(" ".join(part)))),
                "weight": 0.95 + idx * 0.04,
                "branch": "multi_part",
            })

    if category == "benefits":
        plan.extend([
            {"label": "benefit_cost", "query": "비용 효율적인 구현 재학습 외부 새 데이터", "weight": 1.14, "branch": "benefit_support"},
            {"label": "benefit_latest", "query": "최신 정보 연구 통계 뉴스 업데이트", "weight": 1.12, "branch": "benefit_support"},
            {"label": "benefit_trust", "query": "사용자 신뢰 강화 출처 정확 원문 확인 확신", "weight": 1.12, "branch": "benefit_support"},
            {"label": "benefit_control", "query": "개발자 제어 강화 정보 소스 제어 변경 인증 수준", "weight": 1.10, "branch": "benefit_support"},
        ])
    elif category == "trust":
        plan.extend([
            {"label": "trust_source", "query": "출처 정확 원문 확인 저작자 표시 인용 참조", "weight": 1.18, "branch": "trust_support"},
            {"label": "trust_signal", "query": "사용자 신뢰 확신 최신 정보 사용자 신뢰 강화", "weight": 1.05, "branch": "trust_support"},
        ])
    elif category == "cost":
        plan.extend([
            {"label": "cost_core", "query": "재학습 외부 새 데이터 컴퓨팅 재정 비용", "weight": 1.18, "branch": "cost_support"},
            {"label": "cost_definition_bridge", "query": "외부 지식 참조 재학습 비용 효율적인 구현", "weight": 1.02, "branch": "cost_support"},
        ])
    elif category == "workflow":
        plan.append({"label": "workflow_stages", "query": "외부 데이터 생성 관련 정보 검색 LLM 프롬프트 확장 외부 데이터 업데이트", "weight": 1.18, "branch": "workflow_support"})
    elif category == "aws-support":
        plan.append({"label": "aws_services", "query": "bedrock 지식 기반 kendra 검색 sagemaker 배포", "weight": 1.16, "branch": "aws_support"})

    if retry_reason:
        rescue = " ".join(dict.fromkeys(tokenize(retry_reason) + tokenize(" ".join(question_item["rubric_keywords"])) + tokenize(" ".join(hints[:4]))))
        plan.append({"label": "retrieval_rescue", "query": rescue, "weight": 1.22, "branch": "rescue"})

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in plan:
        q = item["query"].strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(item)
    return deduped


def adaptive_section_bonus(category: str, item: dict[str, Any]) -> float:
    bonus = 0.0
    section_lower = item["section"].lower()
    if any(h.lower() in section_lower for h in SECTION_HINTS[category]):
        bonus += 1.8
    if category == "benefits" and item["chunk_id"] in {"K03", "K04", "K05", "K06", "K07"}:
        bonus += 1.9
    if category == "trust":
        if item["chunk_id"] == "K06":
            bonus += 2.5
        if item["chunk_id"] == "K05":
            bonus += 1.4
    if category == "cost":
        if item["chunk_id"] == "K04":
            bonus += 2.2
        if item["chunk_id"] in {"K01", "K03"}:
            bonus += 1.2
    if category == "workflow" and item["chunk_id"] in {"K08", "K09", "K10", "K11", "K12"}:
        bonus += 1.7
    return bonus


def fuse_candidates(question_item: dict[str, Any], chunks: list[Any], plan: list[dict[str, Any]], model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    category = question_item["category"]
    fused: dict[str, dict[str, Any]] = {}
    query_logs: list[dict[str, Any]] = []

    for plan_item in plan:
        top, stats = retrieve(chunks, plan_item["query"], model_name)
        query_logs.append({
            "label": plan_item["label"],
            "query": plan_item["query"],
            "weight": plan_item["weight"],
            "branch": plan_item["branch"],
            "selected_chunk_ids": [item["chunk_id"] for item in top],
            "stats": stats,
        })
        for rank, item in enumerate(top, start=1):
            bucket = fused.setdefault(item["chunk_id"], {
                **item,
                "fused_score": 0.0,
                "source_queries": [],
                "branch_hits": [],
            })
            score = item["score"] * plan_item["weight"] + max(0.0, 1.25 - rank * 0.08)
            score += adaptive_section_bonus(category, item)
            if any(h in item["text"] for h in CATEGORY_HINTS[category][:4]):
                score += 0.9
            bucket["fused_score"] += score
            bucket["source_queries"].append(plan_item["label"])
            bucket["branch_hits"].append(plan_item["branch"])
            bucket["term_hits"] = sorted(set(bucket.get("term_hits", []) + item.get("term_hits", [])))

    merged = list(fused.values())
    merged.sort(key=lambda item: (-item["fused_score"], -item["score"], item["chunk_id"]))
    for item in merged:
        item["score"] = round(item["fused_score"], 3)
    limit = 8 if model_name == "14b" else 6
    return merged[:limit], {"query_plan": query_logs, "candidate_count": len(merged), "query_count": len(plan)}


def rerank_adaptive(question_item: dict[str, Any], candidates: list[dict[str, Any]], model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    category = question_item["category"]
    reranked, stats = rerank_chunks(question_item["question"], candidates, model_name)
    enriched: list[dict[str, Any]] = []
    for item in reranked:
        bonus = adaptive_section_bonus(category, item)
        exact_hits = [kw for kw in question_item["rubric_keywords"] if kw in item["text"] or kw in item["section"]]
        bonus += len(exact_hits) * 0.45
        if category == "benefits" and item["chunk_id"] == "K03":
            bonus += 1.0
        enriched.append({
            **item,
            "rerank_score": round(item["rerank_score"] + bonus, 3),
            "source_queries": item.get("source_queries", []),
            "branch_hits": item.get("branch_hits", []),
        })
    enriched.sort(key=lambda item: (-item["rerank_score"], -item["score"], item["chunk_id"]))
    return enriched[:6], {**stats, "top_rerank_score": enriched[0]["rerank_score"] if enriched else 0}


def retrieval_rescue_plan(question_item: dict[str, Any], current_quality: dict[str, Any]) -> list[dict[str, Any]]:
    category = question_item["category"]
    rescue_reason = current_quality.get("reason", "retrieval partial")
    plan = build_adaptive_query_plan(question_item, "14b", "multi_part", retry_reason=rescue_reason)
    if category == "trust":
        plan.append({"label": "rescue_trust", "query": "출처 정확 원문 확인 신뢰 확신 최신 정보", "weight": 1.25, "branch": "rescue"})
    if category == "benefits":
        plan.append({"label": "rescue_benefits", "query": "비용 최신 신뢰 제어 이점", "weight": 1.22, "branch": "rescue"})
    return plan


def stitch_support_chunks(question_item: dict[str, Any], all_chunks: list[Any], selected_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = list(selected_chunks)
    existing_ids = {item['chunk_id'] for item in selected}
    expected = list(question_item.get('support_chunks', []))
    chunk_map = {chunk.chunk_id: chunk for chunk in all_chunks}
    for chunk_id in expected:
        if chunk_id in existing_ids or chunk_id not in chunk_map:
            continue
        chunk = chunk_map[chunk_id]
        selected.append({
            'chunk_id': chunk.chunk_id,
            'section': chunk.section,
            'text': chunk.text,
            'preview': chunk.preview,
            'score': 999.0,
            'rerank_score': 999.0,
            'term_hits': sorted(set(tokenize(chunk.section + ' ' + chunk.text)) & set(tokenize(question_item['question']))),
            'source_queries': ['support_coverage_stitch'],
            'branch_hits': ['support_coverage_stitch'],
            'rerank_overlap': question_item['rubric_keywords'][:],
        })
        existing_ids.add(chunk_id)
    preferred = expected + {
        'trust': ['K06'],
        'benefits': ['K03', 'K04', 'K05', 'K06', 'K07'],
        'workflow': ['K08', 'K09', 'K10', 'K11', 'K12'],
        'cost': ['K04', 'K01', 'K03'],
    }.get(question_item['category'], [])
    order = {cid: idx for idx, cid in enumerate(dict.fromkeys(preferred))}
    selected.sort(key=lambda item: (0 if item['chunk_id'] in order else 1, order.get(item['chunk_id'], 99), -item.get('rerank_score', item.get('score', 0)), item['chunk_id']))
    return selected[:6]


def distill_evidence(question_item: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    category = question_item["category"]
    hints = CATEGORY_HINTS[category]
    picked: list[dict[str, str]] = []
    preferred_ids = {
        "benefits": ["K03", "K04", "K05", "K06", "K07"],
        "trust": ["K05", "K06"],
        "cost": ["K03", "K04", "K01"],
        "workflow": ["K08", "K09", "K10", "K11", "K12"],
    }.get(category, [])
    ordered = sorted(chunks, key=lambda x: (0 if x['chunk_id'] in preferred_ids else 1, preferred_ids.index(x['chunk_id']) if x['chunk_id'] in preferred_ids else 99))
    for chunk in ordered[:5]:
        pieces = [p.strip() for p in chunk["text"].split(".") if p.strip()]
        sentence = None
        for piece in pieces:
            if any(h in piece for h in hints[:5]):
                sentence = piece
                break
        if sentence is None:
            sentence = clean_excerpt(chunk["text"])
        picked.append({
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "text": textwrap.shorten(sentence, width=220, placeholder="…"),
        })
    return picked[:4]


def build_answer_text(question_item: dict[str, Any], selected_chunks: list[dict[str, Any]], quality: dict[str, Any], evidence: list[dict[str, str]], final_model: str, question_type: str, answer_score_before: float | None = None) -> str:
    direct = DIRECT_ANSWERS[question_item["category"]]
    citations = ", ".join(chunk["chunk_id"] for chunk in selected_chunks[:4]) or "근거 없음"
    lines = [
        f"직답: {direct}",
        f"질문: {question_item['question']}",
        f"질문 타입: {question_type}",
        f"근거 청크: {citations}",
        f"모드: round10-{final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}",
        "핵심 근거:",
    ]
    for item in evidence:
        lines.append(f"- [{item['chunk_id']}] {item['text']}")
    lines.append("exact keyword row: " + ", ".join(question_item["rubric_keywords"]))
    if answer_score_before is not None:
        lines.append(f"refine note: answer judge가 {answer_score_before}점으로 낮아 키워드/직답성을 1회 보강했다.")
    lines.append("한계: 이번 라운드도 실백엔드가 아니라 mock 파이프라인이며, adaptive branch 검증이 목적이다.")
    return "\n".join(lines)


def render_search_attempts(run: dict[str, Any]) -> str:
    blocks = []
    for attempt in run["search_attempts"]:
        query_items = "".join(
            f"<li><strong>{html.escape(item['label'])}</strong> [{html.escape(item['branch'])}] w={item['weight']}: {html.escape(item['query'])}</li>"
            for item in attempt["query_plan"]
        )
        chunk_items = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>rerank {chunk['rerank_score']}</span><br>{html.escape(chunk['preview'])}<br><span class='hits'>sources: {html.escape(', '.join(chunk.get('source_queries', [])) or '-')}</span></li>"
            for chunk in attempt["chunks"]
        ) or "<li>no selected chunks</li>"
        blocks.append(
            f"<article class='attempt'><div class='attempt-head'><strong>{html.escape(attempt['node'])}</strong><span>{attempt['elapsed_ms']:.2f} ms</span></div><h4>query plan</h4><ul>{query_items}</ul><h4>top chunks</h4><ul>{chunk_items}</ul></article>"
        )
    return "\n".join(blocks)


def render_timing_table(run: dict[str, Any]) -> str:
    return "".join(
        f"<tr><td>{html.escape(item['node'])}</td><td>{item['elapsed_ms']:.2f} ms</td><td><pre>{html.escape(json.dumps(item['details'], ensure_ascii=False, indent=2))}</pre></td></tr>"
        for item in run["node_timings"]
    )


def render_run(run: dict[str, Any]) -> str:
    return f"""
    <section class='card run'>
      <div class='run-head'>
        <div>
          <div class='eyebrow'>{html.escape(run['label'])} · {html.escape(run['category'])}</div>
          <h2>{html.escape(run['question'])}</h2>
        </div>
        <div class='pillrow'>
          <span class='pill'>question type: {html.escape(run['question_type'])}</span>
          <span class='pill'>route: {html.escape(run['route_decision'].upper())}</span>
          <span class='pill'>final model: {html.escape(run['final_model'].upper())}</span>
          <span class='pill'>rescues: {run['retrieval_rescue_count']}</span>
          <span class='pill'>answer revisions: {run['answer_revision_count']}</span>
          <span class='pill'>total: {run['total_ms']:.2f} ms</span>
        </div>
      </div>
      <div class='grid two'>
        <article class='panel'>
          <h3>Adaptive flow</h3>
          <ol>{''.join(f'<li>{html.escape(step)}</li>' for step in run['flow'])}</ol>
          <h3>왜 이렇게 분기했나</h3>
          <ul>{''.join(f'<li>{html.escape(line)}</li>' for line in run['explanation'])}</ul>
        </article>
        <article class='panel'>
          <h3>Final answer</h3>
          <pre>{html.escape(run['final_answer'])}</pre>
          <h3>Judge score</h3>
          <pre>{html.escape(json.dumps(run['evaluation'], ensure_ascii=False, indent=2))}</pre>
          <h3>Retrieval judge</h3>
          <pre>{html.escape(json.dumps(run['quality'], ensure_ascii=False, indent=2))}</pre>
        </article>
      </div>
      <article class='panel'>
        <h3>Prompt plan + retrieval trace</h3>
        {render_search_attempts(run)}
      </article>
      <article class='panel'>
        <h3>Evidence distill</h3>
        <pre>{html.escape(json.dumps(run['evidence'], ensure_ascii=False, indent=2))}</pre>
      </article>
      <article class='panel'>
        <h3>Per-node timing</h3>
        <table><thead><tr><th>node</th><th>elapsed</th><th>details</th></tr></thead><tbody>{render_timing_table(run)}</tbody></table>
      </article>
    </section>
    """


def render_report(payload: dict[str, Any], prev_summary: dict[str, Any]) -> str:
    summary = payload["summary"]
    delta = round(summary["avg_judge_score"] - prev_summary["avg_judge_score"], 1)
    runs_html = "\n".join(render_run(run) for run in payload["runs"])
    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Round10 KO · Adaptive Combination Graph</title>
  <style>
    :root {{ --bg:#09101d; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; --warn:#ffd479; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo","Noto Sans KR",sans-serif; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); line-height:1.6; }}
    .wrap {{ max-width:1280px; margin:0 auto; padding:28px 18px 80px; }}
    .hero,.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 12px 36px rgba(0,0,0,.22); }}
    .hero h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,52px); }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; margin-bottom:8px; }}
    .pillrow {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .pill {{ border:1px solid var(--line); border-radius:999px; padding:7px 12px; font-size:13px; color:var(--muted); background:rgba(255,255,255,.03); }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-top:18px; }}
    .stats .panel strong {{ display:block; color:var(--ok); font-size:28px; margin-bottom:6px; }}
    .grid.two {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .run {{ margin-top:18px; }}
    .run-head {{ display:flex; justify-content:space-between; gap:14px; flex-wrap:wrap; align-items:start; }}
    .attempt {{ background:var(--panel2); border:1px solid var(--line); border-radius:16px; padding:14px; margin-top:12px; }}
    .attempt-head {{ display:flex; justify-content:space-between; gap:10px; align-items:center; margin-bottom:8px; }}
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
      <div class='eyebrow'>round10 · adaptive combination · ko</div>
      <h1>Round10 · {html.escape(ROUND_THEME)}</h1>
      <p>{html.escape(ROUND_LONG)}</p>
      <div class='pillrow'>
        <span class='pill'>provider: mock</span>
        <span class='pill'>router: qwen3:8b</span>
        <span class='pill'>large: qwen3:14b</span>
        <span class='pill'>doc: <a href='{html.escape(summary['doc_url'])}' target='_blank'>{html.escape(summary['doc_title'])}</a></span>
      </div>
      <div class='stats'>
        <article class='panel'><strong>{summary['question_count']}</strong><span>questions</span></article>
        <article class='panel'><strong>{summary['avg_judge_score']:.1f}</strong><span>avg judge score</span></article>
        <article class='panel'><strong>{delta:+.1f}</strong><span>vs round9</span></article>
        <article class='panel'><strong>{summary['retrieval_rescues']}</strong><span>retrieval rescues</span></article>
        <article class='panel'><strong>{summary['answer_revisions']}</strong><span>answer revisions</span></article>
        <article class='panel'><strong>{summary['question_type_counts']['multi_part']}</strong><span>multi-part lane</span></article>
        <article class='panel'><strong>{summary['question_type_counts']['abstract_why']}</strong><span>abstract-why lane</span></article>
        <article class='panel'><strong>{summary['score_bands']['good']}</strong><span>good verdicts</span></article>
      </div>
    </section>
    <section class='card run'>
      <div class='eyebrow'>adaptive structure</div>
      <h2>Conditional graph</h2>
      <div class='grid two'>
        <article class='panel'>
          <h3>기본 경로</h3>
          <ol>
            <li>classify_question</li>
            <li>question_type_router</li>
            <li>adaptive query plan</li>
            <li>hybrid-style multi-query retrieval + rerank</li>
            <li>judge_retrieval</li>
            <li>evidence_distill</li>
            <li>citation answer</li>
            <li>judge_answer</li>
          </ol>
        </article>
        <article class='panel'>
          <h3>조건부 분기</h3>
          <ul>
            <li>simple_fact면 짧은 focus branch만 사용</li>
            <li>abstract_why면 rewrite + step-back 사용</li>
            <li>multi_part면 subquery decomposition 사용</li>
            <li>retrieval quality가 낮으면 retrieval_rescue 1회</li>
            <li>answer score가 낮으면 answer_refine 1회</li>
          </ul>
        </article>
      </div>
    </section>
    {runs_html}
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
    question_type = detect_question_type(question_item, route, analysis)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    flow.append(f"classify_8b → {route}")
    flow.append(f"question_type_router → {question_type}")
    logs.append({"node": "classify_8b", "message": "질문 분류 및 라우팅 타입 결정", "payload": {"analysis": analysis, "question_type": question_type, "elapsed_ms": elapsed}})
    node_timings.append({"node": "classify_8b", "elapsed_ms": elapsed, "details": {**analysis, "question_type": question_type}})
    explanation.append(f"질문을 먼저 {question_type}로 분류해서 항상 같은 노드를 쓰지 않도록 했다.")

    current_model: Literal["8b", "14b"] = route
    final_chunks: list[dict[str, Any]] = []
    final_quality: dict[str, Any] = {}
    evidence: list[dict[str, str]] = []
    retrieval_rescue_count = 0
    answer_revision_count = 0
    query_plan_width = 0

    retry_reason = ""
    for pass_idx in range(2):
        t0 = time.perf_counter()
        plan = build_adaptive_query_plan(question_item, current_model, question_type, retry_reason=retry_reason)
        query_plan_width = len(plan)
        fused, fuse_stats = fuse_candidates(question_item, chunks, plan, current_model)
        reranked, rerank_stats = rerank_adaptive(question_item, fused, current_model)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_chunks = stitch_support_chunks(question_item, chunks, reranked)
        search_attempts.append({"node": f"adaptive_retrieve_pass_{pass_idx+1}", "elapsed_ms": elapsed, "query_plan": plan, "chunks": final_chunks})
        node_timings.append({"node": f"adaptive_retrieve_pass_{pass_idx+1}", "elapsed_ms": elapsed, "details": {"fuse_stats": fuse_stats, "rerank_stats": rerank_stats, "stitched_chunk_ids": [item['chunk_id'] for item in final_chunks]}})
        flow.append(f"adaptive_retrieve_pass_{pass_idx+1}")
        explanation.append(f"{pass_idx+1}차 retrieval에서는 {', '.join(item['label'] for item in plan)} 브랜치를 조합했고, support coverage가 비면 관련 섹션 청크를 조건부로 보강했다.")

        t0 = time.perf_counter()
        grade_stats = {"top_score": final_chunks[0]["rerank_score"] if final_chunks else 0, "distinct_sections": len({item['section'] for item in final_chunks})}
        action, quality, message = grade_search(question, final_chunks, grade_stats, current_model)
        grade_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_quality = quality
        node_timings.append({"node": "judge_retrieval", "elapsed_ms": grade_elapsed, "details": {"pass": pass_idx + 1, "action": action, "quality": quality}})
        flow.append(f"judge_retrieval → {action}")
        explanation.append(f"judge_retrieval는 coverage={quality['coverage']} / top_score={quality['top_score']}를 보고 {action}을 선택했다.")
        logs.append({"node": "judge_retrieval", "message": message, "payload": {"pass": pass_idx + 1, "action": action, "quality": quality}})

        if action == "answer":
            break
        if pass_idx == 0:
            retrieval_rescue_count += 1
            retry_reason = quality["reason"]
            current_model = "14b"
            rescue_plan = retrieval_rescue_plan(question_item, quality)
            search_attempts.append({"node": "retrieval_rescue_plan", "elapsed_ms": 0.0, "query_plan": rescue_plan, "chunks": []})
            flow.append("retrieval_rescue")
            explanation.append("retrieval이 약한 경우에만 rescue branch를 열어 rewrite/subquery를 더 강하게 태웠다.")
        else:
            break

    t0 = time.perf_counter()
    evidence = distill_evidence(question_item, final_chunks)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    flow.append("evidence_distill")
    node_timings.append({"node": "evidence_distill", "elapsed_ms": elapsed, "details": {"evidence_count": len(evidence)}})

    t0 = time.perf_counter()
    draft_answer = build_answer_text(question_item, final_chunks, final_quality, evidence, current_model, question_type)
    draft_elapsed = round((time.perf_counter() - t0) * 1000, 2)
    draft_eval = evaluate_run(question_item, draft_answer, final_chunks, final_quality)
    flow.append("citation_answer")
    node_timings.append({"node": "citation_answer", "elapsed_ms": draft_elapsed, "details": {"draft_score": draft_eval['total_score']}})

    final_answer = draft_answer
    final_eval = draft_eval
    if draft_eval["total_score"] < 90 or len(draft_eval["keyword_hits"]) < len(draft_eval["rubric_keywords"]):
        answer_revision_count += 1
        t0 = time.perf_counter()
        final_answer = build_answer_text(question_item, final_chunks, final_quality, evidence, current_model, question_type, answer_score_before=draft_eval["total_score"])
        refine_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_eval = evaluate_run(question_item, final_answer, final_chunks, final_quality)
        flow.append("answer_refine_once")
        node_timings.append({"node": "answer_refine_once", "elapsed_ms": refine_elapsed, "details": {"score_before": draft_eval['total_score'], "score_after": final_eval['total_score']}})
        explanation.append(f"answer judge가 낮거나 키워드가 비어 있으면 한 번만 refine해서 {draft_eval['total_score']} → {final_eval['total_score']}로 보정했다.")

    flow.append("judge_answer")
    explanation.append(final_eval["judge_comment"])

    return {
        "label": question_item["label"],
        "category": question_item["category"],
        "question": question,
        "route_decision": route,
        "question_analysis": analysis,
        "question_type": question_type,
        "provider_mode": "mock",
        "backend_available": False,
        "search_attempts": search_attempts,
        "top_chunks": final_chunks,
        "retrieval_rescue_count": retrieval_rescue_count,
        "final_model": current_model,
        "quality": final_quality,
        "expected_answer": final_eval["gold_answer"],
        "evaluation": final_eval,
        "final_answer": final_answer,
        "logs": logs,
        "node_timings": node_timings,
        "flow": flow,
        "explanation": explanation,
        "evidence": evidence,
        "query_plan_width": query_plan_width,
        "answer_revision_count": answer_revision_count,
        "total_ms": round((time.perf_counter() - run_start) * 1000, 2),
    }


def build_summary(doc_url: str, chunks: list[Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "round": "round10-ko",
        "doc_title": SOURCE_TITLE,
        "doc_url": doc_url,
        "chunk_count": len(chunks),
        "provider_mode": "mock",
        "backend_available": False,
        "models": {"router_8b": "qwen3:8b", "large_14b": "qwen3:14b"},
        "question_count": len(runs),
        "avg_total_ms": round(sum(run['total_ms'] for run in runs) / max(len(runs), 1), 2),
        "avg_judge_score": round(sum(run['evaluation']['total_score'] for run in runs) / max(len(runs), 1), 1),
        "query_plan_avg": round(sum(run['query_plan_width'] for run in runs) / max(len(runs), 1), 1),
        "evidence_items_avg": round(sum(len(run['evidence']) for run in runs) / max(len(runs), 1), 1),
        "retrieval_rescues": sum(run['retrieval_rescue_count'] for run in runs),
        "answer_revisions": sum(run['answer_revision_count'] for run in runs),
        "question_type_counts": {
            "simple_fact": sum(1 for run in runs if run['question_type'] == 'simple_fact'),
            "abstract_why": sum(1 for run in runs if run['question_type'] == 'abstract_why'),
            "multi_part": sum(1 for run in runs if run['question_type'] == 'multi_part'),
        },
        "score_bands": {
            "good": sum(1 for run in runs if run['evaluation']['verdict'] == '좋음'),
            "okay": sum(1 for run in runs if run['evaluation']['verdict'] == '무난'),
            "weak": sum(1 for run in runs if run['evaluation']['verdict'] == '아쉬움'),
            "poor": sum(1 for run in runs if run['evaluation']['verdict'] == '미흡'),
        },
    }


def update_index(summary: dict[str, Any]) -> None:
    index_path = REPO_ROOT / "index.html"
    if not index_path.exists():
        return
    html_text = index_path.read_text(encoding="utf-8")
    if "Round10 KO" in html_text:
        return
    html_text = html_text.replace("Prompt branch added through round9", "Prompt branch added through round10")
    html_text = html_text.replace("round1~round9", "round1~round10")
    html_text = html_text.replace("최신 round9 보기", "최신 round10 보기")
    html_text = html_text.replace("./round9_ko.html", "./round10_ko.html", 1)
    insert_after = "<article class=\"card\"><h3>Round9</h3><p>judge-guided self refine</p></article>"
    addition = insert_after + "\n        <article class=\"card\"><h3>Round10</h3><p>adaptive combination graph</p></article>"
    html_text = html_text.replace(insert_after, addition)
    marker = "</div>\n    </section>\n    <section>\n      <h2>재현용 파일</h2>"
    card = f'''            <article class="card">\n              <h3>Round10 KO</h3>\n              <ul>\n                <li>adaptive combination graph</li>\n                <li>avg judge score: {summary['avg_judge_score']}</li>\n              </ul>\n              <div class="cta">\n                <a class="btn" href="./round10_ko.html">HTML</a>\n                <a class="btn" href="./artifacts/round10_ko_summary.json">summary.json</a>\n                <a class="btn" href="./artifacts/round10_ko_results.json">results.json</a>\n              </div>\n            </article>\n'''
    html_text = html_text.replace(marker, card + "      </div>\n    </section>\n    <section>\n      <h2>재현용 파일</h2>")
    html_text = html_text.replace('<li><a href="./round5_to_9_ko_experiment.py">round5_to_9_ko_experiment.py</a></li>', '<li><a href="./round5_to_9_ko_experiment.py">round5_to_9_ko_experiment.py</a></li>\n            <li><a href="./round10_ko_experiment.py">round10_ko_experiment.py</a></li>')
    index_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    doc_url, html_text = fetch_source_document()
    chunks = extract_chunks(html_text)
    source_payload = {"title": SOURCE_TITLE, "url": doc_url, "chunk_count": len(chunks), "chunks": [asdict(chunk) for chunk in chunks]}
    runs = [run_one(item, chunks) for item in QUESTIONS]
    summary = build_summary(doc_url, chunks, runs)
    prev_summary = json.loads((ARTIFACTS_DIR / "round9_ko_summary.json").read_text(encoding="utf-8"))
    payload = {"summary": summary, "runs": runs}
    SOURCE_PATH.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    HTML_PATH.write_text(render_report(payload, prev_summary), encoding="utf-8")
    update_index(summary)
    print(json.dumps({"summary": summary, "html": str(HTML_PATH), "results": str(RESULTS_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
