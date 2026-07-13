from __future__ import annotations

import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUT_HTML = REPO_ROOT / "round14_langgraph_static_graph.html"
OUT_JSON = ARTIFACTS_DIR / "round14_langgraph_static_graph.json"
ROUND13_SUMMARY = ARTIFACTS_DIR / "round13_ko_pdf_summary.json"


def load_round13_summary() -> dict:
    return json.loads(ROUND13_SUMMARY.read_text(encoding="utf-8"))


def build_spec(summary: dict) -> dict:
    nodes = [
        {
            "id": "start",
            "label": "질문 입력",
            "kind": "io",
            "function": "run_one(question_item, chunks)",
            "why": "질문별로 독립 실행 trace를 만들기 위한 시작점이다.",
        },
        {
            "id": "classify",
            "label": "classify_8b",
            "kind": "compute",
            "function": "classify_question(question)",
            "why": "질문 복잡도를 계산해 8B/14B 경로의 첫 차선을 정한다.",
        },
        {
            "id": "qtype",
            "label": "question_type_router",
            "kind": "router",
            "function": "detect_question_type(question_item, route, analysis)",
            "why": "simple_fact / abstract_why / multi_part에 따라 query expansion 폭을 바꾼다.",
        },
        {
            "id": "plan1",
            "label": "adaptive_query_plan\n(pass 1)",
            "kind": "compute",
            "function": "build_adaptive_query_plan(..., retry_reason='')",
            "why": "첫 retrieval용 query 묶음을 만든다.",
        },
        {
            "id": "retrieve1",
            "label": "adaptive_retrieve_pass_1",
            "kind": "compute",
            "function": "fuse_candidates() + rerank_chunks() + stitch_support_chunks()",
            "why": "후보 수집, 재정렬, support chunk 보강을 한 번에 수행하는 핵심 검색 패스다.",
        },
        {
            "id": "judge1",
            "label": "judge_retrieval",
            "kind": "judge",
            "function": "grade_search(question, final_chunks, stats, current_model)",
            "why": "현재 검색 품질로 바로 답변 가능한지, rescue가 필요한지 결정한다.",
        },
        {
            "id": "rescue",
            "label": "retrieval_rescue",
            "kind": "router",
            "function": "retry_reason = quality['reason']; current_model = '14b'",
            "why": "품질이 낮을 때만 rescue branch를 열고 14B 차선으로 승격한다.",
        },
        {
            "id": "plan2",
            "label": "adaptive_query_plan\n(pass 2)",
            "kind": "compute",
            "function": "build_adaptive_query_plan(..., retry_reason=quality.reason)",
            "why": "실패 이유를 반영해 두 번째 query plan을 더 강하게 만든다.",
        },
        {
            "id": "retrieve2",
            "label": "adaptive_retrieve_pass_2",
            "kind": "compute",
            "function": "fuse_candidates() + rerank_chunks() + stitch_support_chunks()",
            "why": "rescue 이후의 보강 retrieval pass다.",
        },
        {
            "id": "judge2",
            "label": "judge_retrieval\n(pass 2)",
            "kind": "judge",
            "function": "grade_search(question, final_chunks, stats, current_model)",
            "why": "두 번째 pass 이후에는 답변으로 가거나 그냥 종료한다.",
        },
        {
            "id": "evidence",
            "label": "evidence_distill",
            "kind": "compute",
            "function": "distill_evidence(question_item, final_chunks)",
            "why": "긴 청크를 답변용 근거 문장으로 압축한다.",
        },
        {
            "id": "answer",
            "label": "citation_answer",
            "kind": "compute",
            "function": "build_answer_text(...)",
            "why": "직답 + 근거 + exact keyword row 형식의 답변을 만든다.",
        },
        {
            "id": "judge_answer",
            "label": "judge_answer",
            "kind": "judge",
            "function": "evaluate_run(question_item, final_answer, top_chunks, quality)",
            "why": "gold/rubric 기준으로 점수를 계산하고 refine 여부를 결정한다.",
        },
        {
            "id": "refine_gate",
            "label": "score < 92?",
            "kind": "judge",
            "function": "if draft_eval['total_score'] < 92",
            "why": "낮은 점수일 때만 1회 answer refine을 허용한다.",
        },
        {
            "id": "refine",
            "label": "answer_refine_once",
            "kind": "compute",
            "function": "build_answer_text(..., answer_score_before=draft_eval['total_score'])",
            "why": "무한 self-refine 대신 1회만 보정한다.",
        },
        {
            "id": "final",
            "label": "최종 score / report row",
            "kind": "io",
            "function": "run payload + build_summary() + render_report()",
            "why": "질문별 trace와 점수를 JSON/HTML로 저장하기 위한 종료 지점이다.",
        },
    ]

    edges = [
        {"from": "start", "to": "classify", "label": "invoke"},
        {"from": "classify", "to": "qtype", "label": "route + analysis"},
        {"from": "qtype", "to": "plan1", "label": "simple_fact / abstract_why / multi_part"},
        {"from": "plan1", "to": "retrieve1", "label": "query plan"},
        {"from": "retrieve1", "to": "judge1", "label": "chunks + stats"},
        {"from": "judge1", "to": "evidence", "label": "answer"},
        {"from": "judge1", "to": "rescue", "label": "retry"},
        {"from": "rescue", "to": "plan2", "label": "retry_reason + 14b"},
        {"from": "plan2", "to": "retrieve2", "label": "rescued query plan"},
        {"from": "retrieve2", "to": "judge2", "label": "chunks + stats"},
        {"from": "judge2", "to": "evidence", "label": "answer / stop retry"},
        {"from": "evidence", "to": "answer", "label": "evidence rows"},
        {"from": "answer", "to": "judge_answer", "label": "draft answer"},
        {"from": "judge_answer", "to": "refine_gate", "label": "draft score"},
        {"from": "refine_gate", "to": "refine", "label": "yes"},
        {"from": "refine_gate", "to": "final", "label": "no"},
        {"from": "refine", "to": "final", "label": "re-evaluate"},
    ]

    mermaid = """
flowchart TD
    start([질문 입력]) --> classify[classify_8b\n복잡도 기반 8B/14B route]
    classify --> qtype{question_type_router\nsimple_fact / abstract_why / multi_part}
    qtype --> plan1[adaptive_query_plan\npass 1]
    plan1 --> retrieve1[adaptive_retrieve_pass_1\nfuse + rerank + support stitch]
    retrieve1 --> judge1{judge_retrieval}
    judge1 -->|answer| evidence[evidence_distill]
    judge1 -->|retry| rescue[retrieval_rescue\n14B로 승격]
    rescue --> plan2[adaptive_query_plan\npass 2 with retry_reason]
    plan2 --> retrieve2[adaptive_retrieve_pass_2\nfuse + rerank + support stitch]
    retrieve2 --> judge2{judge_retrieval\npass 2}
    judge2 -->|answer or stop| evidence
    evidence --> answer[citation_answer]
    answer --> judge_answer[judge_answer\nevaluate_run]
    judge_answer --> refine_gate{score < 92?}
    refine_gate -->|yes| refine[answer_refine_once]
    refine_gate -->|no| final([최종 score / report row])
    refine --> final

    qtype -.-> simple[simple_fact: focus query 강화]
    qtype -.-> abstract[abstract_why: rewrite + step-back]
    qtype -.-> multi[multi_part: subquery decomposition]

    classDef io fill:#173153,stroke:#7cc9ff,color:#eef3ff,stroke-width:2px;
    classDef compute fill:#1b2142,stroke:#9db3ff,color:#eef3ff;
    classDef judge fill:#3a2446,stroke:#ffb2d8,color:#fff0fb;
    classDef router fill:#213a2e,stroke:#8effc8,color:#eefcf5;

    class start,final io;
    class classify,plan1,retrieve1,plan2,retrieve2,evidence,answer,refine compute;
    class judge1,judge2,judge_answer,refine_gate judge;
    class qtype,rescue router;
""".strip()

    flowline = (
        "질문 입력 → classify_8b → question_type_router → adaptive_query_plan(pass1) → "
        "adaptive_retrieve_pass_1 → judge_retrieval → [필요시 retrieval_rescue + pass2] → "
        "evidence_distill → citation_answer → judge_answer → [필요시 answer_refine_once] → 최종 score"
    )

    return {
        "title": "Round14 LangGraph Static Graph",
        "source": {
            "base_script": "round13_ko_pdf_experiment.py",
            "reference_script": "round10_ko_experiment.py",
            "summary_json": "artifacts/round13_ko_pdf_summary.json",
        },
        "notes": {
            "mode": "static-graph",
            "langgraph_runtime": False,
            "provider_mode": summary.get("provider_mode"),
            "backend_available": summary.get("backend_available"),
            "purpose": "LangSmith tracing 전 단계에서 최종 RAG의 분기 구조를 눈으로 검증하기 위한 정적 그래프",
        },
        "summary_snapshot": {
            "question_count": summary.get("question_count"),
            "avg_judge_score": summary.get("avg_judge_score"),
            "retrieval_rescues": summary.get("retrieval_rescues"),
            "answer_revisions": summary.get("answer_revisions"),
            "question_type_counts": summary.get("question_type_counts"),
        },
        "nodes": nodes,
        "edges": edges,
        "flowline": flowline,
        "mermaid": mermaid,
    }


def render_node_cards(nodes: list[dict]) -> str:
    cards = []
    for node in nodes:
        cards.append(
            f"""
            <article class='card node-card kind-{html.escape(node['kind'])}'>
              <div class='eyebrow'>{html.escape(node['kind'])}</div>
              <h3>{html.escape(node['label'])}</h3>
              <p><strong>코드 위치</strong><br><code>{html.escape(node['function'])}</code></p>
              <p><strong>왜 필요한가</strong><br>{html.escape(node['why'])}</p>
            </article>
            """
        )
    return "\n".join(cards)


def render_edge_rows(edges: list[dict]) -> str:
    rows = []
    for edge in edges:
        rows.append(
            f"<tr><td><code>{html.escape(edge['from'])}</code></td><td><code>{html.escape(edge['to'])}</code></td><td>{html.escape(edge['label'])}</td></tr>"
        )
    return "\n".join(rows)


def render_html(spec: dict) -> str:
    snapshot = spec["summary_snapshot"]
    notes = spec["notes"]
    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Round14 · Static LangGraph View</title>
  <meta name='description' content='round13 KO PDF 파이프라인을 LangGraph 스타일 정적 그래프로 시각화한 페이지' />
  <style>
    :root {{ --bg:#09101d; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; --warn:#ffd479; --pink:#ffb2d8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Pretendard','Apple SD Gothic Neo','Noto Sans KR',sans-serif; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); line-height:1.65; }}
    .wrap {{ max-width:1320px; margin:0 auto; padding:28px 18px 88px; }}
    .hero,.card,.panel,pre {{ background:var(--panel); border:1px solid var(--line); border-radius:24px; box-shadow:0 14px 40px rgba(0,0,0,.2); }}
    .hero,.card,.panel {{ padding:22px; }}
    .hero h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,52px); line-height:1.18; }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; margin-bottom:8px; }}
    .pillrow,.cta {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .pill,.btn {{ border:1px solid var(--line); border-radius:999px; padding:8px 12px; background:rgba(255,255,255,.03); color:var(--muted); text-decoration:none; }}
    .btn {{ border-radius:14px; color:var(--text); background:var(--panel2); font-weight:700; }}
    .grid {{ display:grid; gap:16px; }}
    .grid.two {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .grid.three {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
    .flowline {{ margin-top:14px; padding:14px 16px; border-radius:16px; border:1px solid #30457c; background:linear-gradient(180deg,rgba(124,201,255,.08),rgba(255,255,255,.02)); font-size:18px; font-weight:800; }}
    .panel p, .card p, li {{ color:var(--muted); }}
    .stats strong {{ display:block; color:var(--ok); font-size:30px; margin-bottom:6px; }}
    .mermaid-shell {{ padding:10px; border-radius:18px; background:rgba(255,255,255,.02); border:1px solid var(--line); overflow:auto; }}
    .mermaid {{ min-width:980px; }}
    pre {{ padding:18px; overflow:auto; white-space:pre-wrap; color:#d9e6ff; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ border-top:1px solid var(--line); padding:10px; text-align:left; vertical-align:top; font-size:14px; }}
    th {{ color:var(--accent); font-size:13px; }}
    .kind-judge {{ border-color:#5b2f62; }}
    .kind-router {{ border-color:#2f6b51; }}
    .kind-io {{ border-color:#2f517a; }}
    .kind-compute {{ border-color:#394a86; }}
    .warn {{ color:var(--warn); }}
    a {{ color:var(--accent); }}
    @media (max-width: 980px) {{ .grid.two,.grid.three {{ grid-template-columns:1fr; }} .flowline {{ font-size:16px; }} .mermaid {{ min-width:760px; }} }}
  </style>
  <script type='module'>
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
  </script>
</head>
<body>
  <main class='wrap'>
    <section class='hero'>
      <div class='eyebrow'>round14 · static langgraph view · ko</div>
      <h1>Round14 · 최종 RAG를 LangGraph 스타일 정적 그래프로 먼저 시각화</h1>
      <p>이 페이지는 <strong>round13_ko_pdf_experiment.py</strong>의 실제 함수 흐름을 기준으로 만든 <strong>정적 graph view</strong>다. 아직 LangSmith trace를 붙인 실행 런타임은 아니고, 먼저 “어디서 분기하고 어디서 점수를 매기는지”를 한눈에 보이게 만드는 A안 산출물이다.</p>
      <div class='pillrow'>
        <span class='pill'>question count: {snapshot['question_count']}</span>
        <span class='pill'>avg judge score: {snapshot['avg_judge_score']}</span>
        <span class='pill'>retrieval rescues: {snapshot['retrieval_rescues']}</span>
        <span class='pill'>answer revisions: {snapshot['answer_revisions']}</span>
        <span class='pill'>provider mode: {html.escape(str(notes['provider_mode']))}</span>
      </div>
      <div class='pillrow' style='margin-top:10px;'>
        <a class='btn' href='./round13_ko_pdf.html'>round13 PDF 리포트</a>
        <a class='btn' href='./artifacts/round14_langgraph_static_graph.json'>graph spec JSON</a>
        <a class='btn' href='./round14_langgraph_static_graph.py'>generator script</a>
        <a class='btn' href='./round13_ko_pdf_experiment.py'>round13 코드</a>
      </div>
      <div class='flowline'>{html.escape(spec['flowline'])}</div>
    </section>

    <section class='grid three' style='margin-top:18px;'>
      <article class='card stats'><strong>{snapshot['question_type_counts']['simple_fact']}</strong><p>simple_fact questions</p></article>
      <article class='card stats'><strong>{snapshot['question_type_counts']['abstract_why']}</strong><p>abstract_why questions</p></article>
      <article class='card stats'><strong>{snapshot['question_type_counts']['multi_part']}</strong><p>multi_part questions</p></article>
    </section>

    <section class='card' style='margin-top:18px;'>
      <div class='eyebrow'>graph</div>
      <h2>Mermaid graph</h2>
      <p>아래 그래프는 <span class='warn'>실행 trace</span>가 아니라 <strong>현재 코드 구조를 LangGraph 식으로 옮겨 적은 정적 다이어그램</strong>이다. 즉, 다음 단계에서 실제 <code>StateGraph</code>로 포팅할 때의 설계 청사진 역할을 한다.</p>
      <div class='mermaid-shell'>
        <div class='mermaid'>
{html.escape(spec['mermaid'])}
        </div>
      </div>
    </section>

    <section class='grid two' style='margin-top:18px;'>
      <article class='panel'>
        <div class='eyebrow'>what this proves</div>
        <h2>A안에서 지금 확인한 것</h2>
        <ul>
          <li>최종 RAG가 실제로 어디서 branch 되는지 정리했다.</li>
          <li><code>judge_retrieval</code>와 <code>judge_answer</code>가 별개 gate라는 점을 시각적으로 분리했다.</li>
          <li>retrieval rescue와 answer refine가 <strong>항상 실행되는 노드가 아니라 조건부 노드</strong>라는 점을 드러냈다.</li>
          <li>질문 유형에 따라 <code>adaptive_query_plan</code> 내부 폭이 달라진다는 사실을 별도 branch 힌트로 표현했다.</li>
        </ul>
      </article>
      <article class='panel'>
        <div class='eyebrow'>what this does not prove</div>
        <h2>아직 아닌 것</h2>
        <ul>
          <li>실제 LangGraph runtime이나 LangSmith trace는 아직 연결하지 않았다.</li>
          <li>노드별 입력/출력 state serialization은 아직 정의하지 않았다.</li>
          <li>현재 페이지는 mock provider mode 기준 구조도이며, 실백엔드 추론 latency를 나타내지 않는다.</li>
          <li>다음 단계 B안에서 <code>StateGraph</code> 래퍼를 만들어야 질문별 live trace를 볼 수 있다.</li>
        </ul>
      </article>
    </section>

    <section class='card' style='margin-top:18px;'>
      <div class='eyebrow'>node catalog</div>
      <h2>노드별 설명</h2>
      <div class='grid three'>
        {render_node_cards(spec['nodes'])}
      </div>
    </section>

    <section class='grid two' style='margin-top:18px;'>
      <article class='panel'>
        <div class='eyebrow'>edge list</div>
        <h2>Edge / transition 목록</h2>
        <table>
          <thead><tr><th>from</th><th>to</th><th>label</th></tr></thead>
          <tbody>
            {render_edge_rows(spec['edges'])}
          </tbody>
        </table>
      </article>
      <article class='panel'>
        <div class='eyebrow'>mermaid source</div>
        <h2>원본 Mermaid 텍스트</h2>
        <pre><code>{html.escape(spec['mermaid'])}</code></pre>
      </article>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_round13_summary()
    spec = build_spec(summary)
    OUT_JSON.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_HTML.write_text(render_html(spec), encoding="utf-8")
    print(OUT_HTML.resolve())
    print(OUT_JSON.resolve())


if __name__ == "__main__":
    main()
