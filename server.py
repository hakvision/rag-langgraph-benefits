from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from build_demo import (
    SOURCE_TITLE,
    SOURCE_URL,
    build_search_query,
    classify_question,
    clean_excerpt,
    extract_chunks,
    fetch_source_document,
    grade_search,
    retrieve,
)

REPO_ROOT = Path(__file__).resolve().parent


class BackendSettings(BaseModel):
    provider_mode: Literal["mock", "openai-compatible"] = "mock"
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str | None = None
    model_8b: str = "qwen3:8b"
    model_14b: str = "qwen3:14b"
    temperature: float = 0.1
    top_k_8b: int = 4
    top_k_14b: int = 6


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    settings: BackendSettings = Field(default_factory=BackendSettings)


class AppState(dict):
    question: str
    route_decision: Literal["8b", "14b"]
    question_analysis: dict
    search_query_history: list[str]
    top_chunks: list[dict]
    attempts_8b: int
    attempts_14b: int
    restart_count: int
    retry_reason: str
    final_model: str
    final_answer: str
    quality: dict
    logs: list[dict]


def append_log(state: AppState, node: str, message: str, **payload: Any) -> None:
    state.setdefault("logs", []).append({"node": node, "message": message, "payload": payload})


@dataclass
class DocStore:
    title: str
    url: str
    chunks: list


def build_ui_html() -> str:
    return """<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>로컬 8B/14B 라우팅 RAG UI</title>
  <style>
    :root {
      --bg:#0b1020; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; --warn:#ffd479;
    }
    * { box-sizing:border-box; }
    body { margin:0; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo","Noto Sans KR",sans-serif; }
    .wrap { max-width:1280px; margin:0 auto; padding:24px 16px 80px; }
    h1,h2,h3 { margin:0 0 12px; line-height:1.2; }
    h1 { font-size:clamp(28px,4vw,48px); }
    .hero,.card { background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 12px 36px rgba(0,0,0,.22); }
    .hero p,.sub,.muted { color:var(--muted); }
    .grid { display:grid; gap:16px; }
    .layout { grid-template-columns: 380px 1fr; align-items:start; }
    .pillrow { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
    .pill { border:1px solid var(--line); border-radius:999px; padding:7px 12px; font-size:13px; color:var(--muted); background:rgba(255,255,255,.03); }
    label { display:block; font-size:13px; color:var(--muted); margin:0 0 6px; }
    input, select, textarea, button { width:100%; border-radius:14px; border:1px solid var(--line); background:#09101d; color:var(--text); padding:12px 14px; font:inherit; }
    textarea { min-height:130px; resize:vertical; }
    .formgrid { display:grid; gap:12px; }
    .inline2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .btnrow { display:flex; gap:10px; }
    button { cursor:pointer; background:linear-gradient(180deg,#1b2b52,#152444); font-weight:700; }
    button.secondary { background:#0c1530; }
    .status { font-size:13px; color:var(--accent); margin-top:10px; min-height:20px; }
    pre { margin:0; white-space:pre-wrap; word-break:break-word; }
    .answer { background:#0a0f1d; border:1px solid var(--line); border-radius:18px; padding:16px; min-height:110px; }
    .section-title { margin:28px 0 10px; font-size:24px; }
    .runmeta { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 14px; }
    .loglist { display:grid; gap:12px; }
    .logitem { background:var(--panel2); border:1px solid var(--line); border-radius:18px; padding:14px; }
    .logitem h4 { margin:0 0 8px; font-size:16px; color:var(--accent); }
    .small { font-size:13px; color:var(--muted); }
    .chunks { display:grid; gap:10px; }
    .chunk { background:#0a0f1d; border:1px solid var(--line); border-radius:16px; padding:12px; }
    .score { color:var(--warn); font-size:12px; }
    a { color:var(--accent); }
    @media (max-width: 980px) { .layout { grid-template-columns:1fr; } .inline2 { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <div class=\"pill\" style=\"display:inline-block;width:auto\">LOCAL BACKEND UI</div>
      <h1>집 PC에서 직접 질문 넣고, 8B/14B 라우팅 RAG 결과를 보는 화면</h1>
      <p class=\"sub\">여기서는 네가 브라우저에 질문을 넣으면, 이 로컬 서버가 <strong>8B router → 8B/14B 검색 → 품질 게이트 → 재시작</strong> 흐름으로 실행하고 로그까지 반환한다. 즉 <strong>네가 입력한 대로</strong> 바로 결과가 나온다.</p>
      <div class=\"pillrow\">
        <span class=\"pill\">8B router</span>
        <span class=\"pill\">8B query rewrite</span>
        <span class=\"pill\">14B fallback</span>
        <span class=\"pill\">RAG chunks</span>
        <span class=\"pill\">full logs</span>
      </div>
    </section>

    <section class=\"grid layout\" style=\"margin-top:18px\">
      <aside class=\"card\">
        <h2>실행 설정</h2>
        <div class=\"formgrid\">
          <div>
            <label for=\"provider_mode\">모드</label>
            <select id=\"provider_mode\">
              <option value=\"mock\">mock demo (테스트용)</option>
              <option value=\"openai-compatible\">openai-compatible local backend</option>
            </select>
          </div>
          <div>
            <label for=\"base_url\">Base URL</label>
            <input id=\"base_url\" value=\"http://127.0.0.1:11434/v1\" />
          </div>
          <div>
            <label for=\"api_key\">API key (없으면 비워둬도 됨)</label>
            <input id=\"api_key\" value=\"\" placeholder=\"ollama면 보통 비워둠\" />
          </div>
          <div class=\"inline2\">
            <div>
              <label for=\"model_8b\">8B model</label>
              <input id=\"model_8b\" value=\"qwen3:8b\" />
            </div>
            <div>
              <label for=\"model_14b\">14B model</label>
              <input id=\"model_14b\" value=\"qwen3:14b\" />
            </div>
          </div>
          <div>
            <label for=\"question\">질문</label>
            <textarea id=\"question\">What happens if the search results look weird or low quality?</textarea>
          </div>
          <div class=\"btnrow\">
            <button id=\"runBtn\">질문 실행</button>
            <button id=\"saveBtn\" class=\"secondary\" type=\"button\">설정 저장</button>
          </div>
          <div id=\"status\" class=\"status\">준비됨</div>
        </div>
      </aside>

      <main class=\"grid\">
        <section class=\"card\">
          <h2>최종 답변</h2>
          <div id=\"runmeta\" class=\"runmeta\"></div>
          <div class=\"answer\"><pre id=\"answer\">아직 실행 안 함</pre></div>
        </section>

        <section class=\"card\">
          <h2>검색 청크</h2>
          <div id=\"chunks\" class=\"chunks\"></div>
        </section>

        <section class=\"card\">
          <h2>상세 로그</h2>
          <div id=\"logs\" class=\"loglist\"></div>
        </section>

        <section class=\"card\">
          <h2>문서 정보</h2>
          <pre id=\"docinfo\">loading…</pre>
        </section>
      </main>
    </section>
  </div>
<script>
const $ = (id) => document.getElementById(id);
const fields = ["provider_mode","base_url","api_key","model_8b","model_14b","question"];
function collectSettings() {
  return {
    provider_mode: $("provider_mode").value,
    base_url: $("base_url").value.trim(),
    api_key: $("api_key").value.trim() || null,
    model_8b: $("model_8b").value.trim(),
    model_14b: $("model_14b").value.trim()
  };
}
function saveSettings() {
  const data = { ...collectSettings(), question: $("question").value };
  localStorage.setItem("rag_ui_settings", JSON.stringify(data));
  $("status").textContent = "설정 저장됨";
}
function loadSettings() {
  const raw = localStorage.getItem("rag_ui_settings");
  if (!raw) return;
  try {
    const data = JSON.parse(raw);
    for (const key of fields) {
      if (data[key] !== undefined && $(key)) $(key).value = data[key];
    }
  } catch (e) {}
}
function renderMeta(result) {
  $("runmeta").innerHTML = [
    `initial route: ${result.route_decision.toUpperCase()}`,
    `final model: ${result.final_model.toUpperCase()}`,
    `restarts: ${result.restart_count}`,
    `doc source: <a href="${result.doc_source}" target="_blank">langgraph agentic rag</a>`
  ].map(x => `<span class="pill">${x}</span>`).join("");
}
function renderChunks(chunks) {
  $("chunks").innerHTML = chunks.map(chunk => `
    <div class="chunk">
      <div><strong>${chunk.chunk_id}</strong> · ${chunk.section}</div>
      <div class="score">score ${chunk.score}</div>
      <div class="small">${chunk.preview}</div>
    </div>
  `).join("") || '<div class="small">청크 없음</div>';
}
function renderLogs(logs) {
  $("logs").innerHTML = logs.map(log => `
    <div class="logitem">
      <h4>${log.node}</h4>
      <div class="small" style="margin-bottom:8px">${log.message}</div>
      <pre>${JSON.stringify(log.payload, null, 2)}</pre>
    </div>
  `).join("");
}
async function loadDocInfo() {
  const res = await fetch('/api/documents');
  const data = await res.json();
  $("docinfo").textContent = JSON.stringify(data, null, 2);
}
async function runQuery() {
  saveSettings();
  $("status").textContent = "실행 중…";
  $("answer").textContent = "실행 중…";
  try {
    const payload = {
      question: $("question").value,
      settings: collectSettings()
    };
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'request failed');
    renderMeta(data);
    $("answer").textContent = data.final_answer;
    renderChunks(data.top_chunks || []);
    renderLogs(data.logs || []);
    $("status").textContent = '완료';
  } catch (err) {
    $("status").textContent = '실패: ' + err.message;
    $("answer").textContent = '실패: ' + err.message;
  }
}
$("runBtn").addEventListener('click', runQuery);
$("saveBtn").addEventListener('click', saveSettings);
loadSettings();
loadDocInfo();
</script>
</body>
</html>"""


def make_initial_state(question: str) -> AppState:
    return AppState(
        question=question,
        route_decision="8b",
        question_analysis={},
        search_query_history=[],
        top_chunks=[],
        attempts_8b=0,
        attempts_14b=0,
        restart_count=0,
        retry_reason="",
        final_model="8b",
        final_answer="",
        quality={},
        logs=[],
    )


app = FastAPI(title="Local Routed RAG UI")
DOC_URL, DOC_HTML = fetch_source_document()
DOC_CHUNKS = extract_chunks(DOC_HTML)
DOC_STORE = DocStore(title=SOURCE_TITLE, url=DOC_URL, chunks=DOC_CHUNKS)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return build_ui_html()


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    return {
        "title": DOC_STORE.title,
        "url": DOC_STORE.url,
        "chunk_count": len(DOC_STORE.chunks),
        "sample_sections": [chunk.section for chunk in DOC_STORE.chunks[:6]],
        "note": "이 문서를 RAG 청크로 사용 중",
    }


async def call_openai_compatible(base_url: str, api_key: str | None, model: str, system: str, user: str, temperature: float = 0.1) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]


async def analyze_question(question: str, settings: BackendSettings) -> tuple[str, dict]:
    fallback_route, fallback_profile = classify_question(question)
    if settings.provider_mode == "mock":
        return fallback_route, {**fallback_profile, "source": "heuristic-mock"}

    system = (
        "You are an 8B router for a RAG workflow. Return strict JSON with keys route, question_type, reason, keywords. "
        "route must be either '8b' or '14b'. Use 14b for multi-step comparison, failure diagnosis, or when grading/rewrite logic matters."
    )
    user = f"Question: {question}"
    try:
        content = await call_openai_compatible(settings.base_url, settings.api_key, settings.model_8b, system, user, settings.temperature)
        match = re.search(r"\{.*\}", content, re.S)
        payload = json.loads(match.group(0) if match else content)
        route = payload.get("route", fallback_route)
        if route not in {"8b", "14b"}:
            route = fallback_route
        payload["source"] = "8b-model"
        payload.setdefault("fallback_profile", fallback_profile)
        return route, payload
    except Exception as exc:
        fallback_profile["source"] = f"heuristic-fallback:{type(exc).__name__}"
        return fallback_route, fallback_profile


async def model_rewrite_query(question: str, model_name: str, settings: BackendSettings, retry_reason: str, heuristic_query: str) -> str:
    if settings.provider_mode == "mock":
        return heuristic_query
    system = (
        "You rewrite user questions into compact retrieval queries for a LangGraph RAG system. "
        "Return only the rewritten query, no markdown. Keep it under 20 words."
    )
    user = (
        f"Original question: {question}\n"
        f"Current retry reason: {retry_reason or 'none'}\n"
        f"Heuristic query candidate: {heuristic_query}\n"
        "Produce a stronger retrieval query."
    )
    model = settings.model_8b if model_name == "8b" else settings.model_14b
    try:
        content = await call_openai_compatible(settings.base_url, settings.api_key, model, system, user, settings.temperature)
        return content.strip().replace("\n", " ")[:240]
    except Exception:
        return heuristic_query


async def model_answer(question: str, model_name: str, settings: BackendSettings, retrieved: list[dict], quality: dict) -> str:
    context = "\n\n".join(
        f"[{item['chunk_id']}] {item['section']}\n{clean_excerpt(item['text'])}" for item in retrieved[:4]
    )
    if settings.provider_mode == "mock":
        lines = [
            f"Route: {model_name.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}",
            "문서 근거 요약:",
        ]
        for item in retrieved[:3]:
            lines.append(f"- [{item['chunk_id']}] {item['section']}: {clean_excerpt(item['text'])}")
        return "\n".join(lines)

    system = (
        "Answer in Korean. You are a grounded RAG answerer. Use only the provided context. "
        "Start with one short summary sentence, then 2-4 bullet points with chunk ids. If evidence is weak, say so explicitly."
    )
    user = f"Question:\n{question}\n\nQuality:\n{json.dumps(quality, ensure_ascii=False)}\n\nContext:\n{context}"
    model = settings.model_8b if model_name == "8b" else settings.model_14b
    try:
        return await call_openai_compatible(settings.base_url, settings.api_key, model, system, user, settings.temperature)
    except Exception as exc:
        fallback = [f"모델 응답 실패로 로컬 요약으로 대체함 ({type(exc).__name__})."]
        for item in retrieved[:3]:
            fallback.append(f"- [{item['chunk_id']}] {item['section']}: {clean_excerpt(item['text'])}")
        return "\n".join(fallback)


async def run_search_node(state: AppState, settings: BackendSettings, model_name: Literal["8b", "14b"]) -> None:
    attempt_key = "attempts_8b" if model_name == "8b" else "attempts_14b"
    state[attempt_key] += 1
    heuristic_query = build_search_query(state["question"], model_name, state[attempt_key], state.get("retry_reason", ""))
    final_query = await model_rewrite_query(state["question"], model_name, settings, state.get("retry_reason", ""), heuristic_query)
    state["search_query_history"].append(f"{model_name.upper()}#{state[attempt_key]}: {final_query}")
    retrieved, stats = retrieve(DOC_STORE.chunks, final_query, model_name)
    limit = settings.top_k_8b if model_name == "8b" else settings.top_k_14b
    state["top_chunks"] = retrieved[:limit]
    state["final_model"] = model_name
    append_log(
        state,
        f"search_{model_name}",
        f"{model_name.upper()} search completed.",
        attempt=state[attempt_key],
        heuristic_query=heuristic_query,
        final_query=final_query,
        top_chunks=state["top_chunks"][:4],
        stats=stats,
    )


def should_force_14b_retry(state: AppState) -> bool:
    question_lower = state["question"].lower()
    return state["final_model"] == "14b" and state["attempts_14b"] == 1 and any(
        term in question_lower for term in ["low quality", "weird", "compare", "retry", "rewrite", "grading"]
    )


async def execute_rag(question: str, settings: BackendSettings) -> AppState:
    state = make_initial_state(question)
    route, analysis = await analyze_question(question, settings)
    state["route_decision"] = route
    state["question_analysis"] = analysis
    state["final_model"] = route
    append_log(state, "classify_8b", f"8B router classified the question for {route.upper()} search.", analysis=analysis)

    current_model: Literal["8b", "14b"] = route  # type: ignore[assignment]
    while True:
        await run_search_node(state, settings, current_model)
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

        state["quality"] = quality
        append_log(state, "grade_results", message, action=action, quality=quality)

        if action == "answer":
            break
        if action == "retry_same_model":
            state["restart_count"] += 1
            state["retry_reason"] = quality.get("reason", "")
            if current_model == "8b" and state["attempts_8b"] >= 2:
                current_model = "14b"
                append_log(state, "route_upgrade", "8B retries exhausted; escalating to 14B.", reason=state["retry_reason"])
            elif current_model == "14b" and state["attempts_14b"] >= 2:
                break
            continue
        if action == "escalate_to_14b":
            state["restart_count"] += 1
            state["retry_reason"] = quality.get("reason", "")
            current_model = "14b"
            append_log(state, "route_upgrade", "Quality gate escalated retrieval to 14B.", reason=state["retry_reason"])
            continue
        break

    state["final_answer"] = await model_answer(state["question"], state["final_model"], settings, state["top_chunks"], state["quality"])
    append_log(state, "answer", "Generated final answer.", final_model=state["final_model"], answer_preview=textwrap.shorten(state["final_answer"], width=260, placeholder="…"))
    return state


@app.post("/api/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    try:
        result = await execute_rag(request.question, request.settings)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"백엔드 호출 실패: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"서버 실행 실패: {exc}") from exc

    return {
        "question": result["question"],
        "route_decision": result["route_decision"],
        "question_analysis": result["question_analysis"],
        "search_query_history": result["search_query_history"],
        "top_chunks": result["top_chunks"],
        "restart_count": result["restart_count"],
        "final_model": result["final_model"],
        "quality": result["quality"],
        "final_answer": result["final_answer"],
        "logs": result["logs"],
        "doc_source": DOC_STORE.url,
        "doc_title": DOC_STORE.title,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=os.getenv("RAG_UI_HOST", "127.0.0.1"), port=int(os.getenv("RAG_UI_PORT", "8000")), reload=False)
