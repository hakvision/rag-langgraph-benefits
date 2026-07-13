from __future__ import annotations

import html
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import round13_ko_pdf_experiment as base

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
HTML_PATH = REPO_ROOT / "round14_langgraph_visualized.html"
RESULTS_PATH = ARTIFACTS_DIR / "round14_langgraph_visualized_results.json"
SUMMARY_PATH = ARTIFACTS_DIR / "round14_langgraph_visualized_summary.json"
SOURCE_PATH = ARTIFACTS_DIR / "round14_langgraph_visualized_source_document.json"
QUESTIONS_PATH = ARTIFACTS_DIR / "round14_langgraph_visualized_questions.json"
GRAPH_PATH = ARTIFACTS_DIR / "round14_langgraph_visualized_graph.json"


class RoundState(TypedDict, total=False):
    question_item: dict[str, Any]
    question: str
    chunks: list[base.Chunk]
    route_decision: str
    current_model: str
    question_analysis: dict[str, Any]
    question_type: str
    query_plan: list[dict[str, Any]]
    query_plan_width: int
    final_chunks: list[dict[str, Any]]
    final_quality: dict[str, Any]
    retrieval_action: str
    retrieval_message: str
    retrieval_rescue_count: int
    retry_reason: str
    evidence: list[dict[str, str]]
    draft_answer: str
    final_answer: str
    evaluation: dict[str, Any]
    should_refine: bool
    search_attempts: list[dict[str, Any]]
    flow: list[str]
    trace: list[dict[str, Any]]
    explanation: list[str]
    answer_revision_count: int
    provider_mode: str
    backend_available: bool
    expected_answer: str


FALLBACK_MERMAID = """
flowchart TD
    start([question]) --> classify[classify_question]
    classify --> qtype[detect_question_type]
    qtype --> plan1[build_adaptive_query_plan pass1]
    plan1 --> retrieve1[fuse_candidates + rerank_chunks + stitch_support_chunks]
    retrieve1 --> judge1{grade_search}
    judge1 -->|answer| evidence[distill_evidence]
    judge1 -->|retry| rescue[retrieval_rescue]
    rescue --> plan2[build_adaptive_query_plan pass2]
    plan2 --> retrieve2[fuse_candidates + rerank_chunks + stitch_support_chunks]
    retrieve2 --> judge2[grade_search]
    judge2 --> evidence
    evidence --> answer[build_answer_text]
    answer --> judge_answer[evaluate_run]
    judge_answer --> refine{score < 92?}
    refine -->|yes| answer_refine[build_answer_text refine]
    refine -->|no| finalize[finalize]
    answer_refine --> finalize
    finalize --> end([end])
""".strip()


def tracing_enabled() -> bool:
    tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    api_key = bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"))
    return tracing_flag and api_key


def append_trace(state: RoundState, node: str, detail: str, payload: dict[str, Any] | None = None, elapsed_ms: float | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    flow = list(state.get("flow", []))
    flow.append(node)
    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": node,
            "detail": detail,
            "elapsed_ms": round(elapsed_ms or 0.0, 2),
            "payload": payload or {},
        }
    )
    return flow, trace


def base_state(question_item: dict[str, Any], chunks: list[base.Chunk]) -> RoundState:
    return {
        "question_item": question_item,
        "question": question_item["question"],
        "chunks": chunks,
        "search_attempts": [],
        "flow": [],
        "trace": [],
        "explanation": [],
        "retrieval_rescue_count": 0,
        "answer_revision_count": 0,
        "retry_reason": "",
        "provider_mode": "mock",
        "backend_available": False,
    }


def classify_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    route, analysis = base.classify_question(state["question"])
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "classify_8b",
        f"복잡도 {analysis['complexity_score']}로 {route} lane 선택",
        {"route": route, "analysis": analysis},
        elapsed,
    )
    return {
        "route_decision": route,
        "current_model": route,
        "question_analysis": analysis,
        "flow": flow,
        "trace": trace,
        "explanation": list(state.get("explanation", [])) + [f"질문 복잡도를 보고 {route} 경로로 시작했다."],
    }


def question_type_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    question_type = base.detect_question_type(state["question_item"], state["route_decision"], state["question_analysis"])
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "question_type_router",
        f"질문 타입을 {question_type}로 판정",
        {"question_type": question_type},
        elapsed,
    )
    return {
        "question_type": question_type,
        "flow": flow,
        "trace": trace,
        "explanation": list(state.get("explanation", [])) + [f"question_type_router가 {question_type} branch를 골랐다."],
    }


def plan_pass_1_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    plan = base.build_adaptive_query_plan(state["question_item"], state["current_model"], state["question_type"])
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "adaptive_query_plan_pass_1",
        f"{len(plan)}개 query branch 생성",
        {"labels": [item["label"] for item in plan]},
        elapsed,
    )
    return {
        "query_plan": plan,
        "query_plan_width": len(plan),
        "flow": flow,
        "trace": trace,
    }


def retrieve_pass_1_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    fused, fuse_stats = base.fuse_candidates(state["question_item"], state["chunks"], state["query_plan"], state["current_model"])
    reranked, rerank_stats = base.rerank_chunks(state["question_item"], fused, state["current_model"])
    final_chunks = base.stitch_support_chunks(state["question_item"], state["chunks"], reranked)
    elapsed = (time.perf_counter() - t0) * 1000
    payload = {
        "query_labels": [item["label"] for item in state["query_plan"]],
        "chunk_ids": [item["chunk_id"] for item in final_chunks],
        "fuse_stats": fuse_stats,
        "rerank_stats": rerank_stats,
    }
    flow, trace = append_trace(
        state,
        "adaptive_retrieve_pass_1",
        f"{len(final_chunks)}개 top chunk 확보",
        payload,
        elapsed,
    )
    attempts = list(state.get("search_attempts", []))
    attempts.append(
        {
            "node": "adaptive_retrieve_pass_1",
            "elapsed_ms": round(elapsed, 2),
            "query_plan": state["query_plan"],
            "chunks": final_chunks,
            "fuse_stats": fuse_stats,
            "rerank_stats": rerank_stats,
        }
    )
    return {
        "final_chunks": final_chunks,
        "search_attempts": attempts,
        "flow": flow,
        "trace": trace,
    }


def judge_pass_1_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    grade_stats = {
        "top_score": state["final_chunks"][0].get("rerank_score", state["final_chunks"][0].get("score", 0)) if state.get("final_chunks") else 0,
        "distinct_sections": len({item["section"] for item in state.get("final_chunks", [])}),
    }
    action, quality, message = base.grade_search(state["question"], state.get("final_chunks", []), grade_stats, state["current_model"])
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "judge_retrieval_pass_1",
        message,
        {"action": action, "quality": quality},
        elapsed,
    )
    return {
        "retrieval_action": action,
        "retrieval_message": message,
        "final_quality": quality,
        "flow": flow,
        "trace": trace,
        "explanation": list(state.get("explanation", [])) + [message],
    }


def rescue_transition_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    retry_reason = state["final_quality"]["reason"]
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "retrieval_rescue",
        "retrieval 품질이 낮아 14B lane으로 승격",
        {"retry_reason": retry_reason},
        elapsed,
    )
    attempts = list(state.get("search_attempts", []))
    attempts.append({"node": "retrieval_rescue", "elapsed_ms": round(elapsed, 2), "reason": retry_reason, "chunks": []})
    return {
        "current_model": "14b",
        "retry_reason": retry_reason,
        "retrieval_rescue_count": int(state.get("retrieval_rescue_count", 0)) + 1,
        "search_attempts": attempts,
        "flow": flow,
        "trace": trace,
        "explanation": list(state.get("explanation", [])) + ["첫 번째 retrieval이 약해서 rescue branch를 열었다."],
    }


def plan_pass_2_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    plan = base.build_adaptive_query_plan(state["question_item"], state["current_model"], state["question_type"], retry_reason=state.get("retry_reason", ""))
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "adaptive_query_plan_pass_2",
        f"rescue reason 반영해 {len(plan)}개 query branch 생성",
        {"labels": [item["label"] for item in plan], "retry_reason": state.get("retry_reason", "")},
        elapsed,
    )
    return {
        "query_plan": plan,
        "query_plan_width": len(plan),
        "flow": flow,
        "trace": trace,
    }


def retrieve_pass_2_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    fused, fuse_stats = base.fuse_candidates(state["question_item"], state["chunks"], state["query_plan"], state["current_model"])
    reranked, rerank_stats = base.rerank_chunks(state["question_item"], fused, state["current_model"])
    final_chunks = base.stitch_support_chunks(state["question_item"], state["chunks"], reranked)
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "adaptive_retrieve_pass_2",
        f"rescue 후 {len(final_chunks)}개 top chunk 확보",
        {
            "query_labels": [item["label"] for item in state["query_plan"]],
            "chunk_ids": [item["chunk_id"] for item in final_chunks],
            "fuse_stats": fuse_stats,
            "rerank_stats": rerank_stats,
        },
        elapsed,
    )
    attempts = list(state.get("search_attempts", []))
    attempts.append(
        {
            "node": "adaptive_retrieve_pass_2",
            "elapsed_ms": round(elapsed, 2),
            "query_plan": state["query_plan"],
            "chunks": final_chunks,
            "fuse_stats": fuse_stats,
            "rerank_stats": rerank_stats,
        }
    )
    return {
        "final_chunks": final_chunks,
        "search_attempts": attempts,
        "flow": flow,
        "trace": trace,
    }


def judge_pass_2_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    grade_stats = {
        "top_score": state["final_chunks"][0].get("rerank_score", state["final_chunks"][0].get("score", 0)) if state.get("final_chunks") else 0,
        "distinct_sections": len({item["section"] for item in state.get("final_chunks", [])}),
    }
    action, quality, message = base.grade_search(state["question"], state.get("final_chunks", []), grade_stats, state["current_model"])
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "judge_retrieval_pass_2",
        message,
        {"action": action, "quality": quality},
        elapsed,
    )
    return {
        "retrieval_action": action,
        "retrieval_message": message,
        "final_quality": quality,
        "flow": flow,
        "trace": trace,
        "explanation": list(state.get("explanation", [])) + [message],
    }


def evidence_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    evidence = base.distill_evidence(state["question_item"], state.get("final_chunks", []))
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "evidence_distill",
        f"{len(evidence)}개 evidence row 추출",
        {"chunk_ids": [item["chunk_id"] for item in evidence]},
        elapsed,
    )
    return {"evidence": evidence, "flow": flow, "trace": trace}


def answer_draft_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    draft_answer = base.build_answer_text(
        state["question_item"],
        state.get("final_chunks", []),
        state.get("final_quality", {}),
        state.get("evidence", []),
        state.get("current_model", "8b"),
        state["question_type"],
    )
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "citation_answer",
        "citation-constrained draft answer 생성",
        {"preview": draft_answer.splitlines()[0] if draft_answer else ""},
        elapsed,
    )
    return {"draft_answer": draft_answer, "final_answer": draft_answer, "flow": flow, "trace": trace}


def answer_judge_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    evaluation = base.evaluate_run(state["question_item"], state["draft_answer"], state.get("final_chunks", []), state.get("final_quality", {}))
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "judge_answer",
        f"judge score {evaluation['total_score']}",
        {"total_score": evaluation["total_score"], "verdict": evaluation["verdict"]},
        elapsed,
    )
    return {
        "evaluation": evaluation,
        "expected_answer": evaluation["gold_answer"],
        "flow": flow,
        "trace": trace,
        "explanation": list(state.get("explanation", [])) + [evaluation["judge_comment"]],
    }


def refine_gate_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    should_refine = state["evaluation"]["total_score"] < 92
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "answer_refine_gate",
        "answer_refine_once 실행 여부 판정",
        {"should_refine": should_refine, "score": state["evaluation"]["total_score"]},
        elapsed,
    )
    return {"should_refine": should_refine, "flow": flow, "trace": trace}


def answer_refine_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    refined_answer = base.build_answer_text(
        state["question_item"],
        state.get("final_chunks", []),
        state.get("final_quality", {}),
        state.get("evidence", []),
        state.get("current_model", "8b"),
        state["question_type"],
        answer_score_before=state["evaluation"]["total_score"],
    )
    evaluation = base.evaluate_run(state["question_item"], refined_answer, state.get("final_chunks", []), state.get("final_quality", {}))
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "answer_refine_once",
        f"answer score {state['evaluation']['total_score']} → {evaluation['total_score']}",
        {"score_before": state["evaluation"]["total_score"], "score_after": evaluation["total_score"]},
        elapsed,
    )
    return {
        "final_answer": refined_answer,
        "evaluation": evaluation,
        "answer_revision_count": int(state.get("answer_revision_count", 0)) + 1,
        "flow": flow,
        "trace": trace,
        "explanation": list(state.get("explanation", [])) + [f"answer refine로 {state['evaluation']['total_score']}점에서 {evaluation['total_score']}점으로 보정했다."],
    }


def finalize_node(state: RoundState) -> dict[str, Any]:
    t0 = time.perf_counter()
    elapsed = (time.perf_counter() - t0) * 1000
    flow, trace = append_trace(
        state,
        "finalize",
        "질문별 LangGraph run 종료",
        {
            "final_model": state.get("current_model"),
            "score": state.get("evaluation", {}).get("total_score"),
            "rescue_count": state.get("retrieval_rescue_count", 0),
            "answer_revision_count": state.get("answer_revision_count", 0),
        },
        elapsed,
    )
    return {"flow": flow, "trace": trace}


def route_after_judge_pass_1(state: RoundState) -> str:
    return "evidence_distill" if state.get("retrieval_action") == "answer" else "retrieval_rescue"


def route_after_refine_gate(state: RoundState) -> str:
    return "answer_refine_once" if state.get("should_refine") else "finalize"


def build_app() -> Any:
    graph = StateGraph(RoundState)
    graph.add_node("classify_8b", classify_node)
    graph.add_node("question_type_router", question_type_node)
    graph.add_node("adaptive_query_plan_pass_1", plan_pass_1_node)
    graph.add_node("adaptive_retrieve_pass_1", retrieve_pass_1_node)
    graph.add_node("judge_retrieval_pass_1", judge_pass_1_node)
    graph.add_node("retrieval_rescue", rescue_transition_node)
    graph.add_node("adaptive_query_plan_pass_2", plan_pass_2_node)
    graph.add_node("adaptive_retrieve_pass_2", retrieve_pass_2_node)
    graph.add_node("judge_retrieval_pass_2", judge_pass_2_node)
    graph.add_node("evidence_distill", evidence_node)
    graph.add_node("citation_answer", answer_draft_node)
    graph.add_node("judge_answer", answer_judge_node)
    graph.add_node("answer_refine_gate", refine_gate_node)
    graph.add_node("answer_refine_once", answer_refine_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "classify_8b")
    graph.add_edge("classify_8b", "question_type_router")
    graph.add_edge("question_type_router", "adaptive_query_plan_pass_1")
    graph.add_edge("adaptive_query_plan_pass_1", "adaptive_retrieve_pass_1")
    graph.add_edge("adaptive_retrieve_pass_1", "judge_retrieval_pass_1")
    graph.add_conditional_edges(
        "judge_retrieval_pass_1",
        route_after_judge_pass_1,
        {"evidence_distill": "evidence_distill", "retrieval_rescue": "retrieval_rescue"},
    )
    graph.add_edge("retrieval_rescue", "adaptive_query_plan_pass_2")
    graph.add_edge("adaptive_query_plan_pass_2", "adaptive_retrieve_pass_2")
    graph.add_edge("adaptive_retrieve_pass_2", "judge_retrieval_pass_2")
    graph.add_edge("judge_retrieval_pass_2", "evidence_distill")
    graph.add_edge("evidence_distill", "citation_answer")
    graph.add_edge("citation_answer", "judge_answer")
    graph.add_edge("judge_answer", "answer_refine_gate")
    graph.add_conditional_edges(
        "answer_refine_gate",
        route_after_refine_gate,
        {"answer_refine_once": "answer_refine_once", "finalize": "finalize"},
    )
    graph.add_edge("answer_refine_once", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_one(app: Any, question_item: dict[str, Any], chunks: list[base.Chunk]) -> dict[str, Any]:
    state = base_state(question_item, chunks)
    config = {
        "run_name": f"{question_item['label']}-{question_item['category']}",
        "tags": ["round14", "langgraph", "ko", "pdf", "mock"],
        "metadata": {"question_label": question_item["label"], "category": question_item["category"], "tracing_enabled": tracing_enabled()},
    }
    final_state = app.invoke(state, config=config)
    return {
        "label": question_item["label"],
        "category": question_item["category"],
        "question": question_item["question"],
        "route_decision": final_state["route_decision"],
        "question_analysis": final_state["question_analysis"],
        "question_type": final_state["question_type"],
        "provider_mode": final_state["provider_mode"],
        "backend_available": final_state["backend_available"],
        "graph_runtime": True,
        "langsmith_tracing_enabled": tracing_enabled(),
        "search_attempts": final_state.get("search_attempts", []),
        "top_chunks": final_state.get("final_chunks", []),
        "retrieval_rescue_count": final_state.get("retrieval_rescue_count", 0),
        "final_model": final_state.get("current_model"),
        "quality": final_state.get("final_quality", {}),
        "expected_answer": final_state.get("expected_answer"),
        "evaluation": final_state.get("evaluation", {}),
        "final_answer": final_state.get("final_answer", final_state.get("draft_answer", "")),
        "flow": final_state.get("flow", []),
        "trace": final_state.get("trace", []),
        "explanation": final_state.get("explanation", []),
        "evidence": final_state.get("evidence", []),
        "query_plan_width": final_state.get("query_plan_width", 0),
        "answer_revision_count": final_state.get("answer_revision_count", 0),
    }


def build_summary(chunks: list[base.Chunk], runs: list[dict[str, Any]], mermaid: str) -> dict[str, Any]:
    return {
        "round": "round14-langgraph-visualized",
        "doc_title": base.PDF_TITLE,
        "doc_url": base.PDF_URL,
        "pdf_path": str(base.PDF_PATH),
        "chunk_count": len(chunks),
        "provider_mode": "mock",
        "backend_available": False,
        "langgraph_runtime": True,
        "langsmith_tracing_enabled": tracing_enabled(),
        "langsmith_env_present": bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")),
        "models": {"router_8b": "qwen3:8b", "large_14b": "qwen3:14b"},
        "question_count": len(runs),
        "avg_judge_score": round(sum(run["evaluation"]["total_score"] for run in runs) / max(len(runs), 1), 1),
        "retrieval_rescues": sum(run["retrieval_rescue_count"] for run in runs),
        "answer_revisions": sum(run["answer_revision_count"] for run in runs),
        "question_type_counts": {
            "simple_fact": sum(1 for run in runs if run["question_type"] == "simple_fact"),
            "abstract_why": sum(1 for run in runs if run["question_type"] == "abstract_why"),
            "multi_part": sum(1 for run in runs if run["question_type"] == "multi_part"),
        },
        "score_bands": {
            "good": sum(1 for run in runs if run["evaluation"]["verdict"] == "좋음"),
            "okay": sum(1 for run in runs if run["evaluation"]["verdict"] == "무난"),
            "weak": sum(1 for run in runs if run["evaluation"]["verdict"] == "아쉬움"),
            "poor": sum(1 for run in runs if run["evaluation"]["verdict"] == "미흡"),
        },
        "question_scores": {run["label"]: run["evaluation"]["total_score"] for run in runs},
        "graph": {"mermaid": mermaid, "node_count": 15},
    }


def render_trace(trace: list[dict[str, Any]]) -> str:
    items = []
    for item in trace:
        payload = json.dumps(item.get("payload", {}), ensure_ascii=False, indent=2)
        items.append(
            f"""
            <article class='trace-row'>
              <div class='trace-head'>
                <strong>{html.escape(item['node'])}</strong>
                <span>{item['elapsed_ms']:.2f} ms</span>
              </div>
              <p>{html.escape(item['detail'])}</p>
              <pre>{html.escape(payload)}</pre>
            </article>
            """
        )
    return "\n".join(items)


def render_search_attempts(run: dict[str, Any]) -> str:
    cards = []
    for attempt in run["search_attempts"]:
        chunk_lines = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>{chunk.get('rerank_score', chunk.get('score', 0))}</span></li>"
            for chunk in attempt.get("chunks", [])
        )
        plan_lines = "".join(
            f"<li><strong>{html.escape(item['label'])}</strong> · {html.escape(item['branch'])}<br><code>{html.escape(item['query'])}</code></li>"
            for item in attempt.get("query_plan", [])
            if isinstance(item, dict) and "query" in item
        )
        cards.append(
            f"""
            <article class='panel'>
              <h4>{html.escape(attempt['node'])}</h4>
              <p>elapsed: {attempt.get('elapsed_ms', 0)} ms</p>
              <div class='grid two'>
                <div><h5>query plan</h5><ul>{plan_lines or '<li>없음</li>'}</ul></div>
                <div><h5>top chunks</h5><ul>{chunk_lines or '<li>없음</li>'}</ul></div>
              </div>
            </article>
            """
        )
    return "\n".join(cards)


def render_report(summary: dict[str, Any], runs: list[dict[str, Any]], mermaid: str) -> str:
    run_cards = []
    for run in runs:
        evidence_lines = "".join(
            f"<li><strong>{html.escape(item['chunk_id'])}</strong> · {html.escape(item['text'])}</li>"
            for item in run.get("evidence", [])
        )
        chunk_lines = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>{chunk.get('rerank_score', chunk.get('score', 0))}</span><br>{html.escape(base.clean_excerpt(chunk['text']))}</li>"
            for chunk in run.get("top_chunks", [])
        )
        run_cards.append(
            f"""
            <section class='card run'>
              <div class='eyebrow'>{html.escape(run['label'])} · {html.escape(run['category'])}</div>
              <h2>{html.escape(run['question'])}</h2>
              <div class='stats mini'>
                <article class='panel'><strong>{run['evaluation']['total_score']:.1f}</strong><span>judge score</span></article>
                <article class='panel'><strong>{html.escape(run['final_model'].upper())}</strong><span>final lane</span></article>
                <article class='panel'><strong>{run['retrieval_rescue_count']}</strong><span>retrieval rescues</span></article>
                <article class='panel'><strong>{run['answer_revision_count']}</strong><span>answer revisions</span></article>
              </div>
              <div class='grid two'>
                <article class='panel'>
                  <h3>Expected answer</h3>
                  <p>{html.escape(run['expected_answer'])}</p>
                  <h3>Final answer</h3>
                  <pre>{html.escape(run['final_answer'])}</pre>
                </article>
                <article class='panel'>
                  <h3>Judge summary</h3>
                  <ul>
                    <li>verdict: <strong>{html.escape(run['evaluation']['verdict'])}</strong></li>
                    <li>keyword hits: {html.escape(', '.join(run['evaluation']['keyword_hits']) or '없음')}</li>
                    <li>support hits: {html.escape(', '.join(run['evaluation']['support_chunks_hit']) or '없음')}</li>
                    <li>quality score: {run['evaluation']['score_breakdown']['quality_score']}</li>
                  </ul>
                  <p>{html.escape(run['evaluation']['judge_comment'])}</p>
                  <h3>Evidence</h3>
                  <ul>{evidence_lines}</ul>
                </article>
              </div>
              <article class='panel'>
                <h3>Trace timeline</h3>
                <div class='trace-list'>{render_trace(run['trace'])}</div>
              </article>
              <article class='panel'>
                <h3>Retrieval attempts</h3>
                {render_search_attempts(run)}
              </article>
              <article class='panel'>
                <h3>Top chunks</h3>
                <ul>{chunk_lines}</ul>
              </article>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Round14 LangGraph Visualized · KO</title>
  <meta name='description' content='round13 PDF branch를 실제 LangGraph StateGraph로 감싼 시각화 리포트' />
  <style>
    :root {{ --bg:#09101d; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; --warn:#ffd479; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Pretendard','Apple SD Gothic Neo','Noto Sans KR',sans-serif; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); line-height:1.65; }}
    .wrap {{ max-width:1320px; margin:0 auto; padding:28px 18px 88px; }}
    .hero,.card,.panel,pre,.trace-row {{ background:var(--panel); border:1px solid var(--line); border-radius:24px; box-shadow:0 14px 40px rgba(0,0,0,.2); }}
    .hero,.card,.panel {{ padding:22px; }}
    pre {{ padding:16px; overflow:auto; white-space:pre-wrap; color:#d9e6ff; }}
    .hero h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,52px); line-height:1.18; }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; margin-bottom:8px; }}
    .pillrow,.cta,.stats {{ display:flex; flex-wrap:wrap; gap:10px; }}
    .pill,.btn {{ border:1px solid var(--line); border-radius:999px; padding:8px 12px; background:rgba(255,255,255,.03); color:var(--muted); text-decoration:none; }}
    .btn {{ border-radius:14px; color:var(--text); background:var(--panel2); font-weight:700; }}
    .grid {{ display:grid; gap:16px; }}
    .grid.two {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .grid.three {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
    .stats strong {{ display:block; color:var(--ok); font-size:30px; margin-bottom:6px; }}
    .mini article {{ min-width:180px; flex:1; }}
    p, li, span {{ color:var(--muted); }}
    .mermaid-shell {{ padding:10px; border-radius:18px; background:rgba(255,255,255,.02); border:1px solid var(--line); overflow:auto; }}
    .mermaid {{ min-width:980px; }}
    .trace-list {{ display:grid; gap:10px; }}
    .trace-row {{ padding:14px; }}
    .trace-head {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:8px; }}
    .score {{ color:var(--ok); }}
    h2,h3,h4,h5 {{ margin:0 0 10px; }}
    a {{ color:var(--accent); }}
    @media (max-width: 980px) {{ .grid.two,.grid.three {{ grid-template-columns:1fr; }} .mermaid {{ min-width:760px; }} }}
  </style>
  <script type='module'>
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
  </script>
</head>
<body>
  <main class='wrap'>
    <section class='hero'>
      <div class='eyebrow'>round14 · langgraph runtime wrapper · ko</div>
      <h1>Round14 · round13 PDF branch를 실제 StateGraph로 감싸서 실행</h1>
      <p>이번 페이지는 정적 구조도 수준을 넘어서, <strong>round13_ko_pdf_experiment.py</strong>의 핵심 단계들을 실제 <strong>LangGraph StateGraph</strong> 노드로 분해해 질문별로 <code>app.invoke()</code> 한 결과를 보여준다. 다만 현재 환경에는 LangSmith API key가 없어서 <strong>원격 LangSmith trace 업로드는 비활성</strong> 상태이고, 대신 로컬 trace timeline을 HTML과 JSON에 남겼다.</p>
      <div class='pillrow'>
        <span class='pill'>question count: {summary['question_count']}</span>
        <span class='pill'>avg judge score: {summary['avg_judge_score']}</span>
        <span class='pill'>retrieval rescues: {summary['retrieval_rescues']}</span>
        <span class='pill'>answer revisions: {summary['answer_revisions']}</span>
        <span class='pill'>langsmith tracing enabled: {str(summary['langsmith_tracing_enabled']).lower()}</span>
      </div>
      <div class='pillrow' style='margin-top:10px;'>
        <a class='btn' href='./round13_ko_pdf.html'>round13 PDF 리포트</a>
        <a class='btn' href='./artifacts/round14_langgraph_visualized_results.json'>results.json</a>
        <a class='btn' href='./artifacts/round14_langgraph_visualized_summary.json'>summary.json</a>
        <a class='btn' href='./round14_langgraph_visualized.py'>runtime script</a>
      </div>
    </section>

    <section class='grid three' style='margin-top:18px;'>
      <article class='card stats'><strong>{summary['question_type_counts']['simple_fact']}</strong><p>simple_fact</p></article>
      <article class='card stats'><strong>{summary['question_type_counts']['abstract_why']}</strong><p>abstract_why</p></article>
      <article class='card stats'><strong>{summary['question_type_counts']['multi_part']}</strong><p>multi_part</p></article>
    </section>

    <section class='card' style='margin-top:18px;'>
      <div class='eyebrow'>graph runtime</div>
      <h2>Compiled graph</h2>
      <p>아래 Mermaid는 실제로 compile한 StateGraph 기준 구조다. 이 구조 위에서 각 질문을 실행했고, 결과 trace는 아래 질문별 섹션에서 확인할 수 있다.</p>
      <div class='mermaid-shell'>
        <div class='mermaid'>
{html.escape(mermaid)}
        </div>
      </div>
    </section>

    {''.join(run_cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    base.ensure_pdf()
    full_text = base.extract_pdf_text()
    chunks = base.build_chunks(full_text)
    app = build_app()
    try:
        mermaid = app.get_graph().draw_mermaid()
    except Exception:
        mermaid = FALLBACK_MERMAID

    runs = [run_one(app, item, chunks) for item in base.QUESTIONS]
    summary = build_summary(chunks, runs, mermaid)
    source_payload = {
        "title": base.PDF_TITLE,
        "url": base.PDF_URL,
        "pdf_path": str(base.PDF_PATH),
        "chunk_count": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
        "raw_text_path": str(base.TEXT_PATH),
    }
    graph_payload = {
        "mermaid": mermaid,
        "langgraph_runtime": True,
        "langsmith_tracing_enabled": tracing_enabled(),
        "node_names": [
            "classify_8b",
            "question_type_router",
            "adaptive_query_plan_pass_1",
            "adaptive_retrieve_pass_1",
            "judge_retrieval_pass_1",
            "retrieval_rescue",
            "adaptive_query_plan_pass_2",
            "adaptive_retrieve_pass_2",
            "judge_retrieval_pass_2",
            "evidence_distill",
            "citation_answer",
            "judge_answer",
            "answer_refine_gate",
            "answer_refine_once",
            "finalize",
        ],
    }
    payload = {"summary": summary, "runs": runs}

    SOURCE_PATH.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    QUESTIONS_PATH.write_text(json.dumps(base.QUESTIONS, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    GRAPH_PATH.write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    HTML_PATH.write_text(render_report(summary, runs, mermaid), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": summary,
                "html": str(HTML_PATH),
                "results": str(RESULTS_PATH),
                "questions": str(QUESTIONS_PATH),
                "source": str(SOURCE_PATH),
                "graph": str(GRAPH_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
