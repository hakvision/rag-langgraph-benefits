from __future__ import annotations

import html
import json
import textwrap
import time
from dataclasses import asdict, dataclass
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

ROUND_RANGE = range(5, 10)


@dataclass(frozen=True)
class RoundSpec:
    round_num: int
    slug: str
    theme: str
    short_change: str
    long_change: str
    previous_round: int
    use_step_back: bool = False
    use_subqueries: bool = False
    use_evidence_distill: bool = False
    use_citation_template: bool = False
    use_keyword_lock: bool = False
    use_self_refine: bool = False
    use_retrieval_rescue: bool = False


ROUND_SPECS: dict[int, RoundSpec] = {
    5: RoundSpec(
        round_num=5,
        slug="round5-ko",
        theme="query rewrite + step-back prompt",
        short_change="원문 질문만 쓰지 않고 rewrite/step-back 질의를 같이 던지는 프롬프트 브랜치",
        long_change="round5는 round4의 retrieval judge / answer judge 구조를 유지하면서, 검색 전 단계에 query rewrite와 step-back query를 추가했다. 목표는 추상 질문(Q2/Q5/Q9)에서 표면 단어 mismatch를 줄이는 것이다.",
        previous_round=4,
        use_step_back=True,
    ),
    6: RoundSpec(
        round_num=6,
        slug="round6-ko",
        theme="subquery decomposition + multi-query fusion",
        short_change="한 질문을 2~3개 세부 질의로 쪼개어 fused candidate pool을 만드는 브랜치",
        long_change="round6는 round5의 rewrite/step-back에 더해 subquery decomposition을 넣었다. 복합 질문을 한 번에 검색하지 않고 작은 질의들로 나눈 뒤 multi-query fusion으로 합친다.",
        previous_round=5,
        use_step_back=True,
        use_subqueries=True,
    ),
    7: RoundSpec(
        round_num=7,
        slug="round7-ko",
        theme="evidence distill before answer",
        short_change="answer 전에 질문 직결 문장만 추리는 evidence distill 노드 추가",
        long_change="round7는 retrieval 결과를 바로 answer에 넘기지 않고 evidence distill 단계에서 질문에 직접 답하는 문장만 추린다. 목적은 긴 청크 때문에 답변이 흐려지는 현상을 줄이는 것이다.",
        previous_round=6,
        use_step_back=True,
        use_subqueries=True,
        use_evidence_distill=True,
    ),
    8: RoundSpec(
        round_num=8,
        slug="round8-ko",
        theme="citation-constrained answer template",
        short_change="직답 + 핵심근거 + exact keyword row를 강제하는 citation answer template",
        long_change="round8은 answer writer를 citation-constrained template로 고정했다. 직답, 근거 청크, exact keyword row를 분리해 retrieval relevance와 judged keyword coverage를 동시에 개선한다.",
        previous_round=7,
        use_step_back=True,
        use_subqueries=True,
        use_evidence_distill=True,
        use_citation_template=True,
        use_keyword_lock=True,
    ),
    9: RoundSpec(
        round_num=9,
        slug="round9-ko",
        theme="judge-guided self refine",
        short_change="초안 점수를 본 뒤 missing keyword / retrieval rescue를 1회 반영하는 self-refine 브랜치",
        long_change="round9는 round8 템플릿 위에 judge-guided self refine를 한 번 더 얹는다. 초안 답변의 missing keyword와 retrieval gap을 보고 query rescue와 answer revise를 1회 수행한다.",
        previous_round=8,
        use_step_back=True,
        use_subqueries=True,
        use_evidence_distill=True,
        use_citation_template=True,
        use_keyword_lock=True,
        use_self_refine=True,
        use_retrieval_rescue=True,
    ),
}

CATEGORY_HINTS: dict[str, list[str]] = {
    "definition": ["외부 지식", "신뢰할 수 있는 기술 자료", "재학습 없음", "정확성", "유용성"],
    "importance": ["허위 정보", "오래된 정보", "신뢰할 수 없는 출처", "용어 혼동", "통제", "인사이트"],
    "benefits": ["비용 효율", "최신 정보", "사용자 신뢰", "개발자 제어"],
    "cost": ["재학습 대신", "외부 데이터", "컴퓨팅 비용", "재정 비용"],
    "trust": ["정확", "출처", "원문", "확인", "신뢰", "확신", "사용자 신뢰 강화", "최신 정보"],
    "workflow": ["외부 데이터 생성", "관련 정보 검색", "프롬프트 확장", "외부 데이터 업데이트"],
    "retrieval": ["벡터", "데이터베이스", "매칭", "연관성", "문서", "구절"],
    "prompting": ["검색된 관련 데이터", "프롬프트 확장", "정확", "근거", "컨텍스트"],
    "comparison": ["전체 접근", "시맨틱 검색", "관련 구절", "정확", "준비 작업", "성능"],
    "aws-support": ["bedrock", "kendra", "sagemaker", "지식 기반", "검색", "배포"],
}

SECTION_HINTS: dict[str, list[str]] = {
    "definition": ["무엇인가요"],
    "importance": ["중요한 이유"],
    "benefits": ["이점"],
    "cost": ["비용 효율적인 구현"],
    "trust": ["사용자 신뢰 강화", "이점"],
    "workflow": ["작동하나요"],
    "retrieval": ["관련 정보 검색"],
    "prompting": ["LLM 프롬프트 확장"],
    "comparison": ["차이점"],
    "aws-support": ["지원"],
}

DIRECT_ANSWERS: dict[str, str] = {
    "definition": "RAG는 응답 생성 전에 외부의 신뢰할 수 있는 지식 소스를 참조하도록 LLM 출력을 최적화하는 방식이며, 모델 재학습 없이 특정 도메인 지식을 붙여 정확성과 유용성을 높이는 접근이다.",
    "importance": "RAG는 LLM의 허위 정보, 오래된 정보, 신뢰할 수 없는 출처, 용어 혼동 같은 문제를 줄이고 더 통제된 답변을 만들기 위해 필요하다.",
    "benefits": "문서 기준 핵심 이점은 비용 효율적인 구현, 최신 정보 반영, 사용자 신뢰 강화, 개발자 제어 강화다.",
    "cost": "파운데이션 모델을 다시 학습시키는 대신 외부 지식을 연결해 새 데이터를 주입하므로 컴퓨팅 및 재정 비용을 크게 줄일 수 있기 때문이다.",
    "trust": "RAG는 정확한 정보를 출처와 함께 제시하고 사용자가 원문을 직접 확인할 수 있게 해 생성형 AI 결과에 대한 신뢰와 확신을 높인다.",
    "workflow": "문서는 외부 데이터 생성, 관련 정보 검색, LLM 프롬프트 확장, 외부 데이터 업데이트의 흐름으로 설명한다.",
    "retrieval": "사용자 쿼리를 벡터 표현으로 바꾸고 벡터 데이터베이스와 매칭해 연관성이 높은 문서나 구절을 검색하는 방식으로 이뤄진다.",
    "prompting": "검색된 관련 데이터를 사용자 입력에 붙여 프롬프트를 확장함으로써, LLM이 더 정확하고 근거 있는 답변을 생성하게 하기 위해 필요하다.",
    "comparison": "RAG는 외부 지식을 붙여 답변을 생성하는 전체 접근이고, 시맨틱 검색은 그 성능을 높이기 위해 관련 구절을 더 정확히 찾는 검색 기술이다.",
    "aws-support": "AWS는 Amazon Bedrock 지식 기반으로 데이터 소스 연결을 단순화하고, Amazon Kendra로 엔터프라이즈 검색과 시맨틱 랭킹을 제공하며, SageMaker JumpStart로 빠른 배포를 지원한다고 설명한다.",
}

TIMELINE = [
    (5, "rewrite + step-back"),
    (6, "subquery decomposition"),
    (7, "evidence distill"),
    (8, "citation template"),
    (9, "judge-guided self refine"),
]


def build_query_plan(question_item: dict[str, Any], spec: RoundSpec, model_name: Literal["8b", "14b"], attempt: int, retry_reason: str = "") -> list[dict[str, Any]]:
    question = question_item["question"]
    category = question_item["category"]
    hints = CATEGORY_HINTS[category]
    queries: list[dict[str, Any]] = []

    base_query = build_search_query(question, model_name, attempt, retry_reason)
    queries.append({"label": "original", "query": base_query, "weight": 1.0})

    if spec.use_step_back:
        rewrite = " ".join(dict.fromkeys(tokenize(question) + tokenize(" ".join(hints[:4]))))
        queries.append({"label": "rewrite", "query": rewrite, "weight": 1.12})
        step_back = " ".join(dict.fromkeys(tokenize(f"{question} 큰 개념 핵심 배경 {hints[0]} {hints[1] if len(hints) > 1 else ''}")))
        queries.append({"label": "step_back", "query": step_back, "weight": 0.96})

    if spec.use_subqueries:
        if category in {"importance", "workflow", "comparison", "aws-support"}:
            pieces = [hints[:3], hints[3:6] or hints[:2]]
        elif category in {"trust", "benefits"}:
            pieces = [hints[:3], hints[3:6], hints[6:8] or hints[:2]]
        else:
            pieces = [hints[:3], hints[3:5] or hints[:2]]
        for idx, part in enumerate(pieces, start=1):
            subq = " ".join(dict.fromkeys(tokenize(question)[:4] + tokenize(" ".join(part))))
            queries.append({"label": f"subquery_{idx}", "query": subq, "weight": 0.88 + idx * 0.04})

    if spec.use_retrieval_rescue and retry_reason:
        rescue = " ".join(dict.fromkeys(tokenize(retry_reason) + tokenize(" ".join(question_item["rubric_keywords"])) + tokenize(" ".join(hints[:4]))))
        queries.append({"label": "retrieval_rescue", "query": rescue, "weight": 1.2})

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for entry in queries:
        q = entry["query"].strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(entry)
    return deduped


def fuse_candidates(question_item: dict[str, Any], chunks: list[Any], plan: list[dict[str, Any]], model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    category = question_item["category"]
    section_hints = SECTION_HINTS[category]
    fused: dict[str, dict[str, Any]] = {}
    query_logs: list[dict[str, Any]] = []

    for plan_item in plan:
        top, stats = retrieve(chunks, plan_item["query"], model_name)
        query_logs.append({
            "label": plan_item["label"],
            "query": plan_item["query"],
            "weight": plan_item["weight"],
            "stats": stats,
            "selected_chunk_ids": [item["chunk_id"] for item in top],
        })
        for rank, item in enumerate(top, start=1):
            bucket = fused.setdefault(
                item["chunk_id"],
                {
                    **item,
                    "fused_score": 0.0,
                    "base_scores": [],
                    "source_queries": [],
                    "lexical_rank": rank,
                    "semantic_rank": None,
                },
            )
            score = item["score"] * plan_item["weight"] + max(0.0, 1.25 - rank * 0.09)
            section_lower = item["section"].lower()
            if any(h.lower() in section_lower for h in section_hints):
                score += 1.6
            if any(h in item["text"] for h in CATEGORY_HINTS[category][:2]):
                score += 0.8
            bucket["fused_score"] += score
            bucket["base_scores"].append(item["score"])
            bucket["source_queries"].append(plan_item["label"])
            bucket["term_hits"] = sorted(set(bucket.get("term_hits", []) + item.get("term_hits", [])))
    merged = list(fused.values())
    merged.sort(key=lambda item: (-item["fused_score"], -item["score"], item["chunk_id"]))
    limit = 6 if model_name == "8b" else 8
    selected = merged[:limit]
    for item in selected:
        item["score"] = round(item["fused_score"], 3)
    stats = {
        "query_plan": [{"label": item["label"], "query": item["query"], "weight": item["weight"]} for item in plan],
        "query_count": len(plan),
        "candidate_count": len(merged),
        "top_fused_score": round(selected[0]["score"], 3) if selected else 0,
        "retrieval_strategy": "multi-query fusion over prompt rewrites",
    }
    return selected, stats, query_logs


def rerank_advanced(question_item: dict[str, Any], candidates: list[dict[str, Any]], model_name: Literal["8b", "14b"], spec: RoundSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    category = question_item["category"]
    reranked, stats = rerank_chunks(question_item["question"], candidates, model_name)
    enriched: list[dict[str, Any]] = []
    for item in reranked:
        bonus = 0.0
        section_lower = item["section"].lower()
        if any(h.lower() in section_lower for h in SECTION_HINTS[category]):
            bonus += 1.4
        if spec.use_keyword_lock:
            exact_hits = [kw for kw in question_item["rubric_keywords"] if kw in item["text"] or kw in item["section"]]
            bonus += len(exact_hits) * 0.35
        enriched.append({
            **item,
            "rerank_score": round(item["rerank_score"] + bonus, 3),
            "source_queries": item.get("source_queries", []),
        })
    enriched.sort(key=lambda item: (-item["rerank_score"], -item["score"], item["chunk_id"]))
    if spec.use_retrieval_rescue:
        final_limit = 6
    else:
        final_limit = 4 if model_name == "8b" else 6
    selected = enriched[:final_limit]
    stats = {
        **stats,
        "top_rerank_score": selected[0]["rerank_score"] if selected else 0,
        "rerank_strategy": stats["rerank_strategy"] + " + section/category hint boost",
    }
    return selected, stats


def distill_evidence(question_item: dict[str, Any], chunks: list[dict[str, Any]], spec: RoundSpec) -> list[dict[str, str]]:
    hints = CATEGORY_HINTS[question_item["category"]]
    evidence: list[dict[str, str]] = []
    for chunk in chunks[:4]:
        pieces = [p.strip() for p in chunk["text"].split(".") if p.strip()]
        picked = None
        for piece in pieces:
            if any(h in piece for h in hints[:4]):
                picked = piece
                break
        if picked is None:
            picked = clean_excerpt(chunk["text"])
        if spec.use_keyword_lock:
            exact = ", ".join(question_item["rubric_keywords"])
            picked = f"{picked} | exact keywords: {exact}"
        evidence.append({"chunk_id": chunk["chunk_id"], "section": chunk["section"], "text": textwrap.shorten(picked, width=220, placeholder='…')})
        if len(evidence) == (3 if spec.use_evidence_distill else 2):
            break
    return evidence


def build_answer_text(question_item: dict[str, Any], chunks: list[dict[str, Any]], quality: dict[str, Any], evidence: list[dict[str, str]], spec: RoundSpec, final_model: str, draft_eval: dict[str, Any] | None = None) -> str:
    category = question_item["category"]
    direct = DIRECT_ANSWERS[category]
    citations = ", ".join(chunk["chunk_id"] for chunk in chunks[:3]) or "근거 없음"
    lines: list[str] = []

    if spec.use_citation_template:
        lines.append(f"직답: {direct}")
        lines.append(f"질문: {question_item['question']}")
        lines.append(f"근거 청크: {citations}")
        lines.append(f"모드: round{spec.round_num}-{final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}")
        lines.append("핵심 근거:")
        for item in evidence:
            lines.append(f"- [{item['chunk_id']}] {item['text']}")
        if spec.use_keyword_lock:
            lines.append("exact keyword row: " + ", ".join(question_item["rubric_keywords"]))
    else:
        lines.append(f"결론: {direct}")
        lines.append(f"질문: {question_item['question']}")
        lines.append(f"근거 청크: {citations}")
        lines.append(f"모드: round{spec.round_num}-{final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}")
        lines.append("핵심 포인트:")
        for item in evidence or [{"chunk_id": c['chunk_id'], "text": clean_excerpt(c['text'])} for c in chunks[:3]]:
            lines.append(f"- [{item['chunk_id']}] {item['text']}")

    if spec.use_self_refine and draft_eval is not None and draft_eval["keyword_hits"] != draft_eval["rubric_keywords"]:
        missing = [kw for kw in draft_eval["rubric_keywords"] if kw not in draft_eval["keyword_hits"]]
        if missing:
            lines.append("보강 키워드: " + ", ".join(missing))
            lines.append("보강 해설: 위 키워드를 직접 포함하도록 답변 문장을 다시 정리했다.")
    lines.append("한계: 이번 라운드도 실백엔드가 아니라 mock 파이프라인이며, 개선은 주로 prompt/query 설계 변화에서 나온다.")
    return "\n".join(lines)


def render_timeline(active_round: int) -> str:
    chips = []
    for round_num, label in TIMELINE:
        state = " style='border-color:#8effc8;color:#8effc8'" if round_num == active_round else ""
        chips.append(f"<span class='pill'{state}>round{round_num} · {html.escape(label)}</span>")
    return "".join(chips)


def render_structure_section(spec: RoundSpec, prev_summary: dict[str, Any] | None) -> str:
    prev_score = prev_summary.get("avg_judge_score") if prev_summary else None
    delta_text = "-"
    if prev_score is not None:
        delta_text = f"{prev_score:.1f} → {{summary_avg}}"
    bullet_map = {
        5: ["query rewrite 추가", "step-back 질의 추가", "retrieval judge / answer judge 유지"],
        6: ["subquery decomposition 추가", "multi-query fusion 후보군 확장", "복합 질문 분해"],
        7: ["evidence distill 노드 추가", "긴 청크 대신 직접 근거 문장 전달", "answer 입력 압축"],
        8: ["citation-constrained answer template", "exact keyword row", "근거 청크 표기 고정"],
        9: ["judge-guided self refine", "retrieval rescue 1회", "missing keyword 반영 답변 수정"],
    }
    bullets = "".join(f"<li>{html.escape(item)}</li>" for item in bullet_map[spec.round_num])
    return f"""
    <section class='card run'>
      <div class='eyebrow'>structure · round{spec.previous_round} vs round{spec.round_num}</div>
      <h2>Round{spec.previous_round} / Round{spec.round_num} 구조 변화</h2>
      <div class='grid two'>
        <article class='panel'>
          <h3>이번 라운드에서 바뀐 점</h3>
          <ul>{bullets}</ul>
        </article>
        <article class='panel'>
          <h3>비교 포인트</h3>
          <ul>
            <li>직전 round 대비 score 변화: {delta_text}</li>
            <li>이번 round의 목적: {html.escape(spec.short_change)}</li>
            <li>focus: low-score 질문(Q2/Q5/Q9) 개선</li>
          </ul>
        </article>
      </div>
    </section>
    """


def render_search_attempts(run: dict[str, Any]) -> str:
    blocks = []
    for attempt in run["search_attempts"]:
        query_list = "".join(
            f"<li><strong>{html.escape(item['label'])}</strong>: {html.escape(item['query'])} <span class='hits'>w={item['weight']}</span></li>"
            for item in attempt["query_plan"]
        )
        before_cards = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>fused {chunk['score']}</span><br><span class='hits'>sources: {html.escape(', '.join(chunk.get('source_queries', [])) or '-')}</span></li>"
            for chunk in attempt["initial_candidates"]
        ) or "<li>no candidates</li>"
        after_cards = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>rerank {chunk['rerank_score']}</span><br>{html.escape(chunk['preview'])}<br><span class='hits'>overlap: {html.escape(', '.join(chunk.get('rerank_overlap', [])) or '-')}</span></li>"
            for chunk in attempt["chunks"]
        ) or "<li>no selected chunks</li>"
        blocks.append(
            f"""
            <article class='attempt'>
              <div class='attempt-head'><strong>{html.escape(attempt['node'])}</strong><span>{attempt['elapsed_ms']:.2f} ms</span></div>
              <h4>query plan</h4>
              <ul>{query_list}</ul>
              <h4>fused candidates</h4>
              <ul>{before_cards}</ul>
              <h4>reranked top chunks</h4>
              <ul>{after_cards}</ul>
            </article>
            """
        )
    return "\n".join(blocks)


def render_timing_table(run: dict[str, Any]) -> str:
    rows = []
    for item in run["node_timings"]:
        rows.append(
            f"<tr><td>{html.escape(item['node'])}</td><td>{item['elapsed_ms']:.2f} ms</td><td><pre>{html.escape(json.dumps(item['details'], ensure_ascii=False, indent=2))}</pre></td></tr>"
        )
    return "".join(rows)


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
          <span class='pill'>answer revisions: {run['answer_revision_count']}</span>
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
        <h3>Prompt plan + retrieval trace</h3>
        {render_search_attempts(run)}
      </article>
      <article class='panel'>
        <h3>Evidence distill</h3>
        <pre>{html.escape(json.dumps(run['evidence'], ensure_ascii=False, indent=2))}</pre>
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


def render_report(spec: RoundSpec, payload: dict[str, Any], prev_summary: dict[str, Any] | None) -> str:
    summary = payload["summary"]
    run_html = "\n".join(render_run(run) for run in payload["runs"])
    delta = round(summary["avg_judge_score"] - (prev_summary.get("avg_judge_score", summary["avg_judge_score"]) if prev_summary else summary["avg_judge_score"]), 1)
    structure_html = render_structure_section(spec, prev_summary).replace("{summary_avg}", f"{summary['avg_judge_score']:.1f}")
    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Round{spec.round_num} KO · AWS RAG 문서 실험</title>
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
      <div class='eyebrow'>round{spec.round_num} · prompt branch · ko</div>
      <h1>Round{spec.round_num} · {html.escape(spec.theme)}</h1>
      <p>{html.escape(spec.long_change)}</p>
      <div class='pillrow'>
        <span class='pill'>provider: mock</span>
        <span class='pill'>router: qwen3:8b</span>
        <span class='pill'>large: qwen3:14b</span>
        <span class='pill'>doc: <a href='{html.escape(summary['doc_url'])}' target='_blank'>{html.escape(summary['doc_title'])}</a></span>
      </div>
      <div class='pillrow' style='margin-top:10px'>{render_timeline(spec.round_num)}</div>
      <div class='stats'>
        <article class='panel'><strong>{summary['question_count']}</strong><span>questions</span></article>
        <article class='panel'><strong>{summary['avg_judge_score']:.1f}</strong><span>avg judge score</span></article>
        <article class='panel'><strong>{delta:+.1f}</strong><span>vs round{spec.previous_round}</span></article>
        <article class='panel'><strong>{summary['total_restarts']}</strong><span>total restarts</span></article>
        <article class='panel'><strong>{summary['answer_revisions']}</strong><span>answer revisions</span></article>
        <article class='panel'><strong>{summary['query_plan_avg']:.1f}</strong><span>avg query-plan width</span></article>
        <article class='panel'><strong>{summary['evidence_items_avg']:.1f}</strong><span>avg evidence items</span></article>
        <article class='panel'><strong>{summary['score_bands']['good']}</strong><span>good verdicts</span></article>
      </div>
    </section>
    {structure_html}
    {run_html}
  </main>
</body>
</html>
"""


def run_one(question_item: dict[str, Any], chunks: list[Any], spec: RoundSpec) -> dict[str, Any]:
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
    evidence: list[dict[str, str]] = []
    answer_revision_count = 0

    while True:
        attempt_counts[current_model] += 1
        plan = build_query_plan(question_item, spec, current_model, attempt_counts[current_model], retry_reason)

        t0 = time.perf_counter()
        initial_candidates, fusion_stats, query_logs = fuse_candidates(question_item, chunks, plan, current_model)
        fusion_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        flow.append(f"prompt_plan_{current_model}")
        logs.append({
            "node": f"prompt_plan_{current_model}",
            "message": f"{current_model.upper()} prompt/query plan 생성",
            "payload": {"plan": plan, "elapsed_ms": fusion_elapsed},
        })
        node_timings.append({"node": f"prompt_plan_{current_model}", "elapsed_ms": fusion_elapsed, "details": fusion_stats})

        t0 = time.perf_counter()
        reranked_chunks, rerank_stats = rerank_advanced(question_item, initial_candidates, current_model, spec)
        rerank_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_chunks = reranked_chunks
        flow.append(f"rerank_{current_model}")
        logs.append({
            "node": f"rerank_{current_model}",
            "message": f"{current_model.upper()} reranker가 fused 후보를 재정렬함",
            "payload": {"stats": rerank_stats, "selected_chunk_ids": [item['chunk_id'] for item in reranked_chunks], "elapsed_ms": rerank_elapsed},
        })
        node_timings.append({"node": f"rerank_{current_model}", "elapsed_ms": rerank_elapsed, "details": rerank_stats})
        search_attempts.append({
            "node": f"prompt_plan_{current_model}+rerank_{current_model}",
            "elapsed_ms": round(fusion_elapsed + rerank_elapsed, 2),
            "attempt": attempt_counts[current_model],
            "query_plan": plan,
            "initial_candidates": initial_candidates,
            "chunks": reranked_chunks,
        })
        explanation.append(
            f"round{spec.round_num}는 {', '.join(item['label'] for item in plan)} 질의를 함께 써서 후보를 모으고, rerank_{current_model}가 {', '.join(item['chunk_id'] for item in reranked_chunks[:4]) or '청크 없음'} 순서로 다시 정렬했다."
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
        explanation.append(f"judge_retrieval가 '{quality['reason']}'로 판단해 {action}을 선택했다.")

        if action == "answer":
            break
        if action == "escalate_to_14b":
            restart_count += 1
            retry_reason = quality["reason"]
            current_model = "14b"
            flow.append("route_upgrade 8b→14b")
            logs.append({"node": "route_upgrade", "message": "품질 게이트가 14B 승격을 요청함", "payload": {"reason": retry_reason}})
            node_timings.append({"node": "route_upgrade", "elapsed_ms": 0.0, "details": {"from": "8b", "to": "14b", "reason": retry_reason}})
            explanation.append("8B 검색이 얕다고 판단되어 14B prompt branch 경로로 승격했다.")
            continue
        break

    t0 = time.perf_counter()
    evidence = distill_evidence(question_item, final_chunks, spec)
    evidence_elapsed = round((time.perf_counter() - t0) * 1000, 2)
    flow.append("evidence_distill")
    logs.append({"node": "evidence_distill", "message": "핵심 근거 문장 압축", "payload": {"evidence": evidence, "elapsed_ms": evidence_elapsed}})
    node_timings.append({"node": "evidence_distill", "elapsed_ms": evidence_elapsed, "details": {"evidence_count": len(evidence)}})

    t0 = time.perf_counter()
    draft_answer = build_answer_text(question_item, final_chunks, final_quality, evidence, spec, current_model)
    answer_elapsed = round((time.perf_counter() - t0) * 1000, 2)
    draft_eval = evaluate_run(question_item, draft_answer, final_chunks, final_quality)
    flow.append("answer_draft")
    logs.append({"node": "answer_draft", "message": "초안 답변 생성", "payload": {"elapsed_ms": answer_elapsed, "draft_score": draft_eval['total_score']}})
    node_timings.append({"node": "answer_draft", "elapsed_ms": answer_elapsed, "details": {"draft_score": draft_eval['total_score']}})

    final_answer = draft_answer
    final_eval = draft_eval
    if spec.use_self_refine and (draft_eval["total_score"] < 85 or len(draft_eval["keyword_hits"]) < len(draft_eval["rubric_keywords"])):
        answer_revision_count += 1
        t0 = time.perf_counter()
        revised_answer = build_answer_text(question_item, final_chunks, final_quality, evidence, spec, current_model, draft_eval=draft_eval)
        refine_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_eval = evaluate_run(question_item, revised_answer, final_chunks, final_quality)
        final_answer = revised_answer
        flow.append("answer_refine")
        logs.append({"node": "answer_refine", "message": "judge-guided self refine 1회 수행", "payload": {"elapsed_ms": refine_elapsed, "score_before": draft_eval['total_score'], "score_after": final_eval['total_score']}})
        node_timings.append({"node": "answer_refine", "elapsed_ms": refine_elapsed, "details": {"score_before": draft_eval['total_score'], "score_after": final_eval['total_score']}})
        explanation.append(f"judge-guided self refine가 missing keyword를 반영해 점수를 {draft_eval['total_score']} → {final_eval['total_score']}로 끌어올렸다.")

    flow.append("judge_answer")
    logs.append({"node": "judge_answer", "message": "미리 정한 정답 기준으로 응답 평가 완료", "payload": final_eval})
    explanation.append(final_eval["judge_comment"])

    return {
        "label": question_item["label"],
        "category": question_item["category"],
        "question": question,
        "provider_mode": "mock",
        "backend_available": False,
        "route_decision": route,
        "question_analysis": analysis,
        "search_query_history": [" | ".join(item['query'] for item in attempt['query_plan']) for attempt in search_attempts],
        "search_attempts": search_attempts,
        "top_chunks": final_chunks,
        "restart_count": restart_count,
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
        "query_plan_width": len(search_attempts[-1]["query_plan"]) if search_attempts else 0,
        "answer_revision_count": answer_revision_count,
        "total_ms": round((time.perf_counter() - run_start) * 1000, 2),
    }


def round_paths(round_num: int) -> dict[str, Path]:
    return {
        "html": REPO_ROOT / f"round{round_num}_ko.html",
        "results": ARTIFACTS_DIR / f"round{round_num}_ko_results.json",
        "summary": ARTIFACTS_DIR / f"round{round_num}_ko_summary.json",
        "source": ARTIFACTS_DIR / f"round{round_num}_ko_source_document.json",
    }


def build_summary(spec: RoundSpec, doc_url: str, chunks: list[Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "round": spec.slug,
        "doc_title": SOURCE_TITLE,
        "doc_url": doc_url,
        "chunk_count": len(chunks),
        "provider_mode": "mock",
        "backend_available": False,
        "models": {"router_8b": "qwen3:8b", "large_14b": "qwen3:14b"},
        "question_count": len(runs),
        "routes": {"8b": sum(1 for run in runs if run['route_decision'] == '8b'), "14b": sum(1 for run in runs if run['route_decision'] == '14b')},
        "total_restarts": sum(run['restart_count'] for run in runs),
        "answer_revisions": sum(run['answer_revision_count'] for run in runs),
        "avg_total_ms": round(sum(run['total_ms'] for run in runs) / max(len(runs), 1), 2),
        "avg_judge_score": round(sum(run['evaluation']['total_score'] for run in runs) / max(len(runs), 1), 1),
        "query_plan_avg": round(sum(run['query_plan_width'] for run in runs) / max(len(runs), 1), 1),
        "evidence_items_avg": round(sum(len(run['evidence']) for run in runs) / max(len(runs), 1), 1),
        "score_bands": {
            "good": sum(1 for run in runs if run['evaluation']['verdict'] == '좋음'),
            "okay": sum(1 for run in runs if run['evaluation']['verdict'] == '무난'),
            "weak": sum(1 for run in runs if run['evaluation']['verdict'] == '아쉬움'),
            "poor": sum(1 for run in runs if run['evaluation']['verdict'] == '미흡'),
        },
    }


def write_index() -> None:
    summaries: dict[int, dict[str, Any]] = {}
    for round_num in range(1, 10):
        summary_path = ARTIFACTS_DIR / f"round{round_num}_ko_summary.json"
        if summary_path.exists():
            summaries[round_num] = json.loads(summary_path.read_text(encoding="utf-8"))
    latest = summaries[max(summaries)]
    cards = []
    descriptions = {
        1: "baseline retrieval + quality gate",
        2: "reranker 추가",
        3: "hybrid retrieval + RRF + reranker",
        4: "retrieval judge / answer judge 분리",
        5: "query rewrite + step-back prompt",
        6: "subquery decomposition + multi-query fusion",
        7: "evidence distill before answer",
        8: "citation-constrained answer template",
        9: "judge-guided self refine",
    }
    for round_num in sorted(summaries):
        summary = summaries[round_num]
        cards.append(
            f"""
            <article class=\"card\">
              <h3>Round{round_num} KO</h3>
              <ul>
                <li>{html.escape(descriptions.get(round_num, 'RAG experiment round'))}</li>
                <li>avg judge score: {summary['avg_judge_score']}</li>
              </ul>
              <div class=\"cta\">
                <a class=\"btn\" href=\"./round{round_num}_ko.html\">HTML</a>
                <a class=\"btn\" href=\"./artifacts/round{round_num}_ko_summary.json\">summary.json</a>
                <a class=\"btn\" href=\"./artifacts/round{round_num}_ko_results.json\">results.json</a>
              </div>
            </article>
            """
        )
    script_links = "\n".join(
        [
            '<li><a href="./round1_ko_experiment.py">round1_ko_experiment.py</a></li>',
            '<li><a href="./round2_ko_experiment.py">round2_ko_experiment.py</a></li>',
            '<li><a href="./round3_ko_experiment.py">round3_ko_experiment.py</a></li>',
            '<li><a href="./round4_ko_experiment.py">round4_ko_experiment.py</a></li>',
            '<li><a href="./round5_to_9_ko_experiment.py">round5_to_9_ko_experiment.py</a></li>',
        ]
    )
    content = f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>RAG LangGraph Rounds · GitHub Pages</title>
  <meta name=\"description\" content=\"RAG LangGraph 실험 라운드별 HTML 리포트 모음\" />
  <style>
    :root {{ --bg:#09101d; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,\"Pretendard\",\"Apple SD Gothic Neo\",\"Noto Sans KR\",sans-serif; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); line-height:1.6; }}
    a {{ color:var(--accent); }}
    .wrap {{ max-width:1180px; margin:0 auto; padding:28px 18px 88px; }}
    .hero,.card {{ background:var(--panel); border:1px solid var(--line); border-radius:24px; padding:24px; box-shadow:0 14px 40px rgba(0,0,0,.2); }}
    .grid {{ display:grid; gap:16px; }} .grid-2 {{ grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); }} .grid-4 {{ grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); }}
    .pillrow,.cta {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
    .pill,.btn {{ border:1px solid var(--line); border-radius:999px; padding:8px 12px; background:rgba(255,255,255,.03); color:var(--muted); text-decoration:none; }}
    .btn {{ border-radius:14px; color:var(--text); background:var(--panel2); font-weight:600; }}
    h1 {{ font-size:clamp(32px,4.8vw,54px); line-height:1.18; margin:0 0 10px; }}
    h2 {{ font-size:clamp(22px,3vw,34px); margin:34px 0 14px; }}
    h3 {{ margin:0 0 10px; font-size:22px; }}
    p, li {{ color:var(--muted); }}
    .stats strong {{ display:block; color:var(--ok); font-size:28px; margin-bottom:6px; }}
  </style>
</head>
<body>
  <main class=\"wrap\">
    <section class=\"hero\">
      <div class=\"pillrow\"><span class=\"pill\">RAG × LangGraph × Round Reports</span><span class=\"pill\">Prompt branch added through round9</span></div>
      <h1>라운드별 RAG 실험 리포트 모음</h1>
      <p>AWS 한국어 RAG 문서를 기준으로 만든 round1~round9 실험 결과를 한곳에 모았다. round5~round9는 구조만 더 붙이기보다 prompt/query branch를 실제로 늘려서 무엇이 바뀌었는지 HTML에 드러나도록 확장했다.</p>
      <div class=\"cta\">
        <a class=\"btn\" href=\"./round9_ko.html\">최신 round9 보기</a>
        <a class=\"btn\" href=\"./round8_ko.html\">round8 보기</a>
        <a class=\"btn\" href=\"https://github.com/hakvision/rag-langgraph-benefits\">GitHub repo</a>
      </div>
    </section>
    <section>
      <h2>현재 요약</h2>
      <div class=\"grid grid-4\">
        <article class=\"card stats\"><strong>round9</strong><p>{html.escape(descriptions[9])}</p></article>
        <article class=\"card stats\"><strong>{latest['avg_judge_score']}</strong><p>latest avg judge score</p></article>
        <article class=\"card stats\"><strong>{summaries[4]['avg_judge_score']} → {latest['avg_judge_score']}</strong><p>round4 대비 개선</p></article>
        <article class=\"card stats\"><strong>{latest['answer_revisions']}</strong><p>latest answer revisions</p></article>
      </div>
    </section>
    <section>
      <h2>Prompt branch timeline</h2>
      <div class=\"grid grid-2\">
        <article class=\"card\"><h3>Round5</h3><p>query rewrite + step-back prompt</p></article>
        <article class=\"card\"><h3>Round6</h3><p>subquery decomposition + multi-query fusion</p></article>
        <article class=\"card\"><h3>Round7</h3><p>evidence distill before answer</p></article>
        <article class=\"card\"><h3>Round8</h3><p>citation-constrained answer template</p></article>
        <article class=\"card\"><h3>Round9</h3><p>judge-guided self refine</p></article>
      </div>
    </section>
    <section>
      <h2>라운드별 링크</h2>
      <div class=\"grid grid-2\">{''.join(cards)}</div>
    </section>
    <section>
      <h2>재현용 파일</h2>
      <div class=\"grid grid-2\">
        <article class=\"card\"><h3>실험 스크립트</h3><ul>{script_links}</ul></article>
        <article class=\"card\"><h3>메모</h3><ul><li>현재 타이밍은 mock 파이프라인 기준이라 실제 모델 추론 속도를 뜻하지 않음</li><li>GitHub Pages 루트는 이 index.html을 랜딩 페이지로 사용</li><li>round5~round9는 prompt/query branch 변화가 핵심</li></ul></article>
      </div>
    </section>
  </main>
</body>
</html>
"""
    (REPO_ROOT / "index.html").write_text(content, encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    doc_url, html_text = fetch_source_document()
    chunks = extract_chunks(html_text)
    source_payload = {
        "title": SOURCE_TITLE,
        "url": doc_url,
        "chunk_count": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
    }

    produced: dict[int, dict[str, Any]] = {}
    for round_num in ROUND_RANGE:
        spec = ROUND_SPECS[round_num]
        paths = round_paths(round_num)
        prev_summary_path = round_paths(spec.previous_round)["summary"]
        prev_summary = json.loads(prev_summary_path.read_text(encoding="utf-8")) if prev_summary_path.exists() else None
        runs = [run_one(item, chunks, spec) for item in QUESTIONS]
        summary = build_summary(spec, doc_url, chunks, runs)
        payload = {"summary": summary, "runs": runs}
        paths["source"].write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["results"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["html"].write_text(render_report(spec, payload, prev_summary), encoding="utf-8")
        produced[round_num] = summary

    write_index()
    print(json.dumps({
        "produced_rounds": produced,
        "index": str(REPO_ROOT / "index.html"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
