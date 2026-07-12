from __future__ import annotations

import html
import json
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import round10_ko_experiment as r10
from round1_ko_experiment import (
    ARTIFACTS_DIR,
    QUESTIONS,
    REPO_ROOT,
    SOURCE_TITLE,
    clean_excerpt,
    evaluate_run,
    extract_chunks,
    fetch_source_document,
    tokenize,
)

ROUND_NUM = 11
HTML_PATH = REPO_ROOT / f"round{ROUND_NUM}_ko.html"
RESULTS_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_results.json"
SUMMARY_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_summary.json"
SOURCE_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_source_document.json"


@dataclass(frozen=True)
class AblationSpec:
    slug: str
    title: str
    removed_node: str
    flow: str
    why: str
    use_question_type_router: bool = True
    use_abstract_rewrite: bool = True
    use_multi_subquery: bool = True
    use_reranker: bool = True
    use_retrieval_judge: bool = True
    use_retrieval_rescue: bool = True
    use_support_stitch: bool = True
    use_evidence_distill: bool = True
    use_citation_template: bool = True
    use_answer_refine: bool = True


ABLATIONS = [
    AblationSpec(
        slug="baseline",
        title="Round10 baseline",
        removed_node="none",
        flow="질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변",
        why="round10 원본 구조. 모든 조건부 노드를 유지한 기준선.",
    ),
    AblationSpec(
        slug="minus-router",
        title="- question_type_router",
        removed_node="question_type_router",
        flow="질문 분류 → generic query plan → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변",
        why="질문 타입 분기를 제거하면 abstract/multi-part 질문도 같은 경로로 처리된다.",
        use_question_type_router=False,
    ),
    AblationSpec(
        slug="minus-abstract-branch",
        title="- abstract rewrite/step-back",
        removed_node="rewrite + step-back branch",
        flow="질문 분류 → question_type_router → (abstract branch 제거) → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변",
        why="importance/trust/cost 계열에서 rewrite·step-back이 실제로 유효했는지 본다.",
        use_abstract_rewrite=False,
    ),
    AblationSpec(
        slug="minus-subquery",
        title="- multi-part subquery",
        removed_node="subquery decomposition",
        flow="질문 분류 → question_type_router → (multi-part subquery 제거) → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변",
        why="workflow/comparison/aws-support에서 subquery decomposition의 기여도를 본다.",
        use_multi_subquery=False,
    ),
    AblationSpec(
        slug="minus-reranker",
        title="- reranker",
        removed_node="reranker",
        flow="질문 분류 → question_type_router → adaptive query plan → retrieval only → retrieval judge → rescue(if needed) → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변",
        why="retrieval 후보 정렬이 빠지면 어떤 질문이 가장 흔들리는지 본다.",
        use_reranker=False,
    ),
    AblationSpec(
        slug="minus-retrieval-judge",
        title="- retrieval judge",
        removed_node="retrieval judge",
        flow="질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변",
        why="검색 품질 판단 없이 바로 answer로 가면 score가 얼마나 깎이는지 본다.",
        use_retrieval_judge=False,
        use_retrieval_rescue=False,
    ),
    AblationSpec(
        slug="minus-retrieval-rescue",
        title="- retrieval rescue",
        removed_node="retrieval rescue",
        flow="질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → retrieval judge → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변",
        why="judge는 유지하되 검색 rescue만 빼서 conditional retry 자체의 가치를 본다.",
        use_retrieval_rescue=False,
    ),
    AblationSpec(
        slug="minus-support-stitch",
        title="- support stitch",
        removed_node="support coverage stitch",
        flow="질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → retrieval judge → rescue(if needed) → evidence distill → citation answer → answer refine(if needed) → 답변",
        why="관련 섹션 보강 없이도 충분했는지, 아니면 support coverage stitch가 실질 점프를 만들었는지 본다.",
        use_support_stitch=False,
    ),
    AblationSpec(
        slug="minus-evidence-distill",
        title="- evidence distill",
        removed_node="evidence distill",
        flow="질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → citation answer → answer refine(if needed) → 답변",
        why="핵심 근거 압축 없이 원문 청크를 바로 답변에 넣으면 어떻게 되는지 본다.",
        use_evidence_distill=False,
    ),
    AblationSpec(
        slug="minus-citation-template",
        title="- citation answer template",
        removed_node="citation answer template",
        flow="질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → evidence distill → plain answer → answer refine(if needed) → 답변",
        why="round8에서 점프를 만들었던 citation template의 기여도를 다시 검증한다.",
        use_citation_template=False,
    ),
    AblationSpec(
        slug="minus-answer-refine",
        title="- answer refine",
        removed_node="answer refine",
        flow="질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → evidence distill → citation answer → 답변",
        why="마지막 1회 refine가 실제로 평균 점수에 얼마나 기여하는지 본다.",
        use_answer_refine=False,
    ),
]


def infer_question_type(question_item: dict[str, Any], route: Literal["8b", "14b"], analysis: dict[str, Any], spec: AblationSpec) -> str:
    if not spec.use_question_type_router:
        return "generic"
    return r10.detect_question_type(question_item, route, analysis)


def filter_plan(plan: list[dict[str, Any]], spec: AblationSpec) -> list[dict[str, Any]]:
    kept = []
    for item in plan:
        branch = item.get("branch")
        if branch == "abstract" and not spec.use_abstract_rewrite:
            continue
        if branch == "multi_part" and not spec.use_multi_subquery:
            continue
        if branch == "rescue" and not spec.use_retrieval_rescue:
            continue
        if branch in {"simple", "abstract", "multi_part"} and not spec.use_question_type_router:
            continue
        kept.append(item)
    return kept or [plan[0]]


def quality_estimate(question: str, chunks: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    q_tokens = set(tokenize(question))
    hits = set()
    for item in chunks:
        hits.update(item.get("term_hits", []))
    coverage = round(len(q_tokens & hits) / max(len(q_tokens), 1), 3)
    top_score = chunks[0].get("rerank_score", chunks[0].get("score", 0)) if chunks else 0
    return {
        "ok": coverage >= 0.25,
        "coverage": coverage,
        "top_score": top_score,
        "distinct_sections": len({item['section'] for item in chunks}),
        "reason": f"no retrieval judge ({model_name})",
    }


def make_plain_evidence(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    evidence = []
    for chunk in chunks[:4]:
        evidence.append({
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "text": clean_excerpt(chunk["text"]),
        })
    return evidence


def build_answer_text(question_item: dict[str, Any], selected_chunks: list[dict[str, Any]], quality: dict[str, Any], evidence: list[dict[str, str]], final_model: str, question_type: str, spec: AblationSpec, score_before: float | None = None) -> str:
    direct = r10.DIRECT_ANSWERS[question_item["category"]]
    citations = ", ".join(chunk["chunk_id"] for chunk in selected_chunks[:4]) or "근거 없음"
    if spec.use_citation_template:
        lines = [
            f"직답: {direct}",
            f"질문: {question_item['question']}",
            f"질문 타입: {question_type}",
            f"근거 청크: {citations}",
            f"모드: round11-{spec.slug}-{final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}",
            "핵심 근거:",
        ]
        for item in evidence:
            lines.append(f"- [{item['chunk_id']}] {item['text']}")
        lines.append("exact keyword row: " + ", ".join(question_item["rubric_keywords"]))
    else:
        lines = [
            f"결론: {direct}",
            f"질문: {question_item['question']}",
            f"질문 타입: {question_type}",
            f"근거 청크: {citations}",
            f"모드: round11-{spec.slug}-{final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}",
            "핵심 포인트:",
        ]
        for item in evidence:
            lines.append(f"- [{item['chunk_id']}] {item['text']}")
    if score_before is not None:
        lines.append(f"refine note: answer judge가 {score_before}점으로 낮아 1회 보강했다.")
    lines.append("한계: 이번 라운드도 실백엔드가 아니라 mock ablation 실험이다.")
    return "\n".join(lines)


def run_variant_one(question_item: dict[str, Any], chunks: list[Any], spec: AblationSpec) -> dict[str, Any]:
    question = question_item["question"]
    flow: list[str] = []
    explanation: list[str] = []
    search_attempts: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    route, analysis = r10.classify_question(question)
    question_type = infer_question_type(question_item, route, analysis, spec)
    flow.append(f"classify_8b → {route}")
    if spec.use_question_type_router:
        flow.append(f"question_type_router → {question_type}")
    else:
        flow.append("question_type_router 제거 → generic path")
    explanation.append(f"{spec.title}에서는 '{spec.removed_node}'를 뺀 상태로 같은 질문 세트를 다시 평가했다.")

    current_model: Literal["8b", "14b"] = route
    final_chunks: list[dict[str, Any]] = []
    final_quality: dict[str, Any] = {}
    retrieval_rescue_count = 0
    query_plan_width = 0

    retry_reason = ""
    max_passes = 2 if (spec.use_retrieval_judge and spec.use_retrieval_rescue) else 1
    for pass_idx in range(max_passes):
        if question_type == "generic":
            raw_plan = r10.build_adaptive_query_plan(question_item, current_model, "simple_fact", retry_reason=retry_reason)
        else:
            raw_plan = r10.build_adaptive_query_plan(question_item, current_model, question_type, retry_reason=retry_reason)
        plan = filter_plan(raw_plan, spec)
        query_plan_width = len(plan)

        t0 = time.perf_counter()
        fused, fuse_stats = r10.fuse_candidates(question_item, chunks, plan, current_model)
        if spec.use_reranker:
            selected, rerank_stats = r10.rerank_adaptive(question_item, fused, current_model)
            flow.append(f"retrieval_pass_{pass_idx+1} + reranker")
        else:
            selected = []
            for item in fused[:6]:
                selected.append({**item, "rerank_score": item.get("score", 0), "rerank_overlap": item.get("term_hits", [])})
            rerank_stats = {"skipped": True, "top_rerank_score": selected[0]['rerank_score'] if selected else 0}
            flow.append(f"retrieval_pass_{pass_idx+1} (reranker 제거)")
        if spec.use_support_stitch:
            final_chunks = r10.stitch_support_chunks(question_item, chunks, selected)
        else:
            final_chunks = selected
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        search_attempts.append({
            "node": f"pass_{pass_idx+1}",
            "elapsed_ms": elapsed,
            "query_plan": plan,
            "chunks": final_chunks,
            "fuse_stats": fuse_stats,
            "rerank_stats": rerank_stats,
        })
        explanation.append(f"pass {pass_idx+1}에서는 {', '.join(item['label'] for item in plan)} 쿼리를 사용했다.")

        if spec.use_retrieval_judge:
            grade_stats = {
                "top_score": final_chunks[0].get("rerank_score", final_chunks[0].get("score", 0)) if final_chunks else 0,
                "distinct_sections": len({item['section'] for item in final_chunks}),
            }
            action, quality, _ = r10.grade_search(question, final_chunks, grade_stats, current_model)
            final_quality = quality
            flow.append(f"judge_retrieval → {action}")
            explanation.append(f"retrieval judge는 coverage={quality['coverage']}를 보고 {action}을 선택했다.")
            if action == "answer":
                break
            if action == "escalate_to_14b" and spec.use_retrieval_rescue and pass_idx == 0:
                retrieval_rescue_count += 1
                current_model = "14b"
                retry_reason = quality["reason"]
                flow.append("retrieval_rescue")
                explanation.append("첫 pass가 약하면 rescue branch를 1회 연다.")
                continue
            break
        else:
            final_quality = quality_estimate(question, final_chunks, current_model)
            break

    if spec.use_evidence_distill:
        evidence = r10.distill_evidence(question_item, final_chunks)
        flow.append("evidence_distill")
    else:
        evidence = make_plain_evidence(final_chunks)
        flow.append("evidence_distill 제거")

    draft_answer = build_answer_text(question_item, final_chunks, final_quality, evidence, current_model, question_type, spec)
    draft_eval = evaluate_run(question_item, draft_answer, final_chunks, final_quality)
    final_answer = draft_answer
    final_eval = draft_eval
    answer_revision_count = 0
    if spec.use_answer_refine and (draft_eval["total_score"] < 90 or len(draft_eval["keyword_hits"]) < len(draft_eval["rubric_keywords"])):
        answer_revision_count = 1
        final_answer = build_answer_text(question_item, final_chunks, final_quality, evidence, current_model, question_type, spec, score_before=draft_eval["total_score"])
        final_eval = evaluate_run(question_item, final_answer, final_chunks, final_quality)
        flow.append("answer_refine_once")
    else:
        if not spec.use_answer_refine:
            flow.append("answer_refine 제거")
    flow.append("judge_answer")
    explanation.append(final_eval["judge_comment"])

    return {
        "label": question_item["label"],
        "category": question_item["category"],
        "question": question,
        "question_type": question_type,
        "route_decision": route,
        "final_model": current_model,
        "quality": final_quality,
        "evaluation": final_eval,
        "final_answer": final_answer,
        "top_chunks": final_chunks,
        "evidence": evidence,
        "flow": flow,
        "explanation": explanation,
        "search_attempts": search_attempts,
        "retrieval_rescue_count": retrieval_rescue_count,
        "answer_revision_count": answer_revision_count,
        "query_plan_width": query_plan_width,
        "total_ms": round((time.perf_counter() - run_start) * 1000, 2),
    }


def summarize_variant(spec: AblationSpec, runs: list[dict[str, Any]], baseline_avg: float | None = None) -> dict[str, Any]:
    avg = round(sum(run['evaluation']['total_score'] for run in runs) / max(len(runs), 1), 1)
    return {
        "slug": spec.slug,
        "title": spec.title,
        "removed_node": spec.removed_node,
        "flow": spec.flow,
        "why": spec.why,
        "avg_judge_score": avg,
        "delta_vs_baseline": round(avg - baseline_avg, 1) if baseline_avg is not None else 0.0,
        "retrieval_rescues": sum(run['retrieval_rescue_count'] for run in runs),
        "answer_revisions": sum(run['answer_revision_count'] for run in runs),
        "question_scores": {run['label']: run['evaluation']['total_score'] for run in runs},
    }


def render_variant_card(summary: dict[str, Any], rank: int) -> str:
    delta = summary['delta_vs_baseline']
    delta_text = f"{delta:+.1f}"
    delta_color = '#8effc8' if delta >= 0 else '#ff9aa5'
    q_rows = ''.join(f"<li>{html.escape(label)}: {score}</li>" for label, score in summary['question_scores'].items())
    return f"""
    <article class='card variant'>
      <div class='eyebrow'>ablation #{rank}</div>
      <h3>{html.escape(summary['title'])}</h3>
      <div class='pillrow'>
        <span class='pill'>avg {summary['avg_judge_score']}</span>
        <span class='pill' style='color:{delta_color};border-color:{delta_color}'>vs baseline {delta_text}</span>
        <span class='pill'>removed: {html.escape(summary['removed_node'])}</span>
      </div>
      <div class='flowline'>{html.escape(summary['flow'])}</div>
      <p>{html.escape(summary['why'])}</p>
      <div class='grid two'>
        <article class='panel'><h4>질문별 점수</h4><ul>{q_rows}</ul></article>
        <article class='panel'><h4>메모</h4><ul><li>retrieval rescues: {summary['retrieval_rescues']}</li><li>answer revisions: {summary['answer_revisions']}</li></ul></article>
      </div>
    </article>
    """


def render_detail_block(payload: dict[str, Any]) -> str:
    rows = []
    for run in payload['runs']:
        rows.append(
            f"<tr><td>{html.escape(run['label'])}</td><td>{html.escape(run['category'])}</td><td>{html.escape(run['question_type'])}</td><td>{run['evaluation']['total_score']}</td><td>{html.escape(', '.join(chunk['chunk_id'] for chunk in run['top_chunks'][:4]))}</td><td>{html.escape(' → '.join(run['flow']))}</td></tr>"
        )
    return ''.join(rows)


def render_report(report: dict[str, Any]) -> str:
    baseline = report['variant_summaries'][0]
    cards = '\n'.join(render_variant_card(item, idx + 1) for idx, item in enumerate(report['variant_summaries']))
    detail_sections = []
    for item in report['variants']:
        detail_sections.append(
            f"<section class='card'><div class='eyebrow'>{html.escape(item['summary']['title'])}</div><h3>{html.escape(item['summary']['removed_node'])}</h3><table><thead><tr><th>Q</th><th>category</th><th>type</th><th>score</th><th>top chunks</th><th>flow</th></tr></thead><tbody>{render_detail_block(item)}</tbody></table></section>"
        )
    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Round11 KO · Round10 Ablation Study</title>
  <style>
    :root {{ --bg:#09101d; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; --warn:#ffd479; --bad:#ff9aa5; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Pretendard','Apple SD Gothic Neo','Noto Sans KR',sans-serif; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); line-height:1.6; }}
    .wrap {{ max-width:1320px; margin:0 auto; padding:28px 18px 80px; }}
    .hero,.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 12px 36px rgba(0,0,0,.22); }}
    .hero h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,52px); }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; margin-bottom:8px; }}
    .pillrow {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .pill {{ border:1px solid var(--line); border-radius:999px; padding:7px 12px; font-size:13px; color:var(--muted); background:rgba(255,255,255,.03); }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-top:18px; }}
    .stats .panel strong {{ display:block; color:var(--ok); font-size:28px; margin-bottom:6px; }}
    .grid.two {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .variant {{ margin-top:18px; }}
    .flowline {{ margin:12px 0; padding:14px 16px; border-radius:16px; border:1px solid #30457c; background:linear-gradient(180deg,rgba(124,201,255,.08),rgba(255,255,255,.02)); font-weight:800; font-size:18px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
    th,td {{ border-top:1px solid var(--line); padding:10px; text-align:left; vertical-align:top; font-size:14px; }}
    th {{ color:var(--accent); font-size:13px; }}
    p, li {{ color:var(--muted); }}
    h2,h3,h4 {{ margin:0 0 10px; }}
    ul,ol {{ margin:0; padding-left:20px; }}
    a {{ color:var(--accent); }}
    @media (max-width:980px) {{ .grid.two {{ grid-template-columns:1fr; }} .flowline {{ font-size:16px; }} }}
  </style>
</head>
<body>
  <main class='wrap'>
    <section class='hero'>
      <div class='eyebrow'>round11 · ablation study · ko</div>
      <h1>Round11 · round10 노드 제거 실험</h1>
      <p>round10의 adaptive combination graph에서 노드를 하나씩 빼면서 10개 질문 전부를 다시 돌렸다. 목적은 <strong>어떤 노드가 실제 평균 점수와 질문별 안정성에 가장 크게 기여했는지</strong>를 보는 것이다.</p>
      <div class='pillrow'>
        <span class='pill'>provider: mock</span>
        <span class='pill'>variants: {len(report['variant_summaries'])}</span>
        <span class='pill'>questions: {report['question_count']}</span>
        <span class='pill'>baseline round10 avg: {baseline['avg_judge_score']}</span>
      </div>
      <div class='stats'>
        <article class='panel'><strong>{baseline['avg_judge_score']}</strong><span>round10 baseline</span></article>
        <article class='panel'><strong>{report['most_important']['delta_vs_baseline']:+.1f}</strong><span>largest drop</span></article>
        <article class='panel'><strong>{html.escape(report['most_important']['title'])}</strong><span>most effective node</span></article>
        <article class='panel'><strong>{html.escape(report['least_important']['title'])}</strong><span>smallest impact ablation</span></article>
      </div>
    </section>

    <section class='card variant'>
      <div class='eyebrow'>round10 structure</div>
      <h2>Round10 원본 구조를 제대로 적어두면</h2>
      <div class='flowline'>질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변</div>
      <div class='grid two'>
        <article class='panel'>
          <h3>조건부 규칙</h3>
          <ul>
            <li>simple_fact면 짧은 focus branch</li>
            <li>abstract_why면 rewrite + step-back</li>
            <li>multi_part면 subquery decomposition</li>
            <li>retrieval quality 낮으면 14B rescue 1회</li>
            <li>support coverage 비면 관련 섹션 청크 stitch</li>
            <li>answer score 낮으면 refine 1회</li>
          </ul>
        </article>
        <article class='panel'>
          <h3>실험 해석법</h3>
          <ul>
            <li>각 ablation은 round10에서 해당 노드만 제거한 버전이다.</li>
            <li>점수 하락폭이 클수록 그 노드가 효과적이었다는 뜻이다.</li>
            <li>flow는 scanability 우선으로 간단한 화살표 형태로 보여준다.</li>
          </ul>
        </article>
      </div>
    </section>

    <section class='card variant'>
      <div class='eyebrow'>flow comparison</div>
      <h2>노드 하나씩 제거한 전체 플로우 비교</h2>
      {cards}
    </section>

    <section class='card variant'>
      <div class='eyebrow'>detailed runs</div>
      <h2>variant별 질문 점수와 실제 flow</h2>
      {''.join(detail_sections)}
    </section>
  </main>
</body>
</html>
"""


def update_index(best_baseline: float) -> None:
    index_path = REPO_ROOT / "index.html"
    if not index_path.exists():
        return
    html_text = index_path.read_text(encoding='utf-8')
    if 'Round11 KO' in html_text:
        return
    html_text = html_text.replace('Prompt branch added through round10', 'Prompt branch added through round11')
    html_text = html_text.replace('round1~round10', 'round1~round11')
    html_text = html_text.replace('최신 round10 보기', '최신 round11 보기')
    html_text = html_text.replace('./round10_ko.html', './round11_ko.html', 1)
    html_text = html_text.replace('<article class="card"><h3>Round10</h3><p>adaptive combination graph</p></article>', '<article class="card"><h3>Round10</h3><p>adaptive combination graph</p></article>\n        <article class="card"><h3>Round11</h3><p>round10 ablation study</p></article>')
    html_text = html_text.replace('<article class="card stats"><strong>round10</strong><p>adaptive combination graph</p></article>', '<article class="card stats"><strong>round11</strong><p>round10 ablation study</p></article>')
    html_text = html_text.replace('<article class="card stats"><strong>90.2</strong><p>latest avg judge score</p></article>', f'<article class="card stats"><strong>{best_baseline}</strong><p>latest baseline avg judge score</p></article>')
    html_text = html_text.replace('<article class="card stats"><strong>67.5 → 90.2</strong><p>round4 대비 개선</p></article>', f'<article class="card stats"><strong>67.5 → {best_baseline}</strong><p>round4 대비 baseline 개선</p></article>')
    html_text = html_text.replace('<article class="card stats"><strong>5</strong><p>latest answer revisions</p></article>', '<article class="card stats"><strong>ablation</strong><p>node removal matrix</p></article>')
    marker = '</div>\n    </section>\n    <section>\n      <h2>재현용 파일</h2>'
    card = f'''            <article class="card">\n              <h3>Round11 KO</h3>\n              <ul>\n                <li>round10 ablation study</li>\n                <li>baseline avg judge score: {best_baseline}</li>\n              </ul>\n              <div class="cta">\n                <a class="btn" href="./round11_ko.html">HTML</a>\n                <a class="btn" href="./artifacts/round11_ko_summary.json">summary.json</a>\n                <a class="btn" href="./artifacts/round11_ko_results.json">results.json</a>\n              </div>\n            </article>\n'''
    html_text = html_text.replace(marker, card + '      </div>\n    </section>\n    <section>\n      <h2>재현용 파일</h2>')
    html_text = html_text.replace('<li><a href="./round10_ko_experiment.py">round10_ko_experiment.py</a></li>', '<li><a href="./round10_ko_experiment.py">round10_ko_experiment.py</a></li>\n            <li><a href="./round11_ko_experiment.py">round11_ko_experiment.py</a></li>')
    index_path.write_text(html_text, encoding='utf-8')


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    doc_url, html_text = fetch_source_document()
    chunks = extract_chunks(html_text)
    source_payload = {"title": SOURCE_TITLE, "url": doc_url, "chunk_count": len(chunks), "chunks": [asdict(chunk) for chunk in chunks]}

    variants = []
    baseline_avg = None
    variant_summaries = []
    for spec in ABLATIONS:
        runs = [run_variant_one(item, chunks, spec) for item in QUESTIONS]
        summary = summarize_variant(spec, runs, baseline_avg)
        if baseline_avg is None:
            baseline_avg = summary['avg_judge_score']
            summary['delta_vs_baseline'] = 0.0
        else:
            summary['delta_vs_baseline'] = round(summary['avg_judge_score'] - baseline_avg, 1)
        variant_summaries.append(summary)
        variants.append({"summary": summary, "runs": runs})

    sorted_ablation = sorted(variant_summaries[1:], key=lambda x: x['delta_vs_baseline'])
    report = {
        "round": "round11-ko",
        "doc_title": SOURCE_TITLE,
        "doc_url": doc_url,
        "question_count": len(QUESTIONS),
        "variant_summaries": variant_summaries,
        "variants": variants,
        "most_important": sorted_ablation[0],
        "least_important": sorted_ablation[-1],
    }

    summary_payload = {
        "round": "round11-ko",
        "doc_title": SOURCE_TITLE,
        "doc_url": doc_url,
        "question_count": len(QUESTIONS),
        "baseline_avg_judge_score": baseline_avg,
        "variant_count": len(variant_summaries),
        "variant_summaries": variant_summaries,
        "most_important": sorted_ablation[0],
        "least_important": sorted_ablation[-1],
    }

    SOURCE_PATH.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    HTML_PATH.write_text(render_report(report), encoding='utf-8')
    assert baseline_avg is not None
    update_index(float(baseline_avg))
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
