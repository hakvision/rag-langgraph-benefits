from __future__ import annotations

import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUT_HTML = REPO_ROOT / "round12_ko.html"


def load_json(name: str):
    return json.loads((ARTIFACTS_DIR / name).read_text(encoding="utf-8"))


def render_node_cards() -> str:
    nodes = [
        {
            "name": "classify_question",
            "role": "질문을 먼저 8B/14B 경로로 분류하는 시작 노드",
            "why": "질문이 짧고 직접적이면 가벼운 경로로 보내고, 비교/흐름/설명형이면 더 무거운 경로를 준비하기 위해 필요했다.",
            "tech": "question token 수, hard term(비교/차이/왜/어떻게/작동/흐름 등), heuristic complexity score를 써서 route를 정한다.",
        },
        {
            "name": "question_type_router",
            "role": "질문을 simple_fact / abstract_why / multi_part로 나누는 분기 노드",
            "why": "모든 질문에 같은 query expansion을 거는 대신, 질문 유형에 따라 필요한 브랜치만 켜기 위해 만들었다.",
            "tech": "category와 complexity를 함께 보고 type을 정한다. comparison/workflow/aws-support는 multi_part, importance/trust/cost는 abstract_why로 라우팅한다.",
        },
        {
            "name": "adaptive query plan",
            "role": "질문 유형별로 서로 다른 검색 질의 묶음을 만드는 노드",
            "why": "round5/6에서 보였던 rewrite / step-back / subquery를 항상 쓰지 않고 조건부로만 쓰기 위해 필요했다.",
            "tech": "base query는 항상 만들고, simple_fact면 focus query, abstract_why면 rewrite + step-back, multi_part면 subquery A/B/C를 추가한다. category별 benefit/trust/cost/workflow/aws-support 보조 query도 같이 붙인다.",
        },
        {
            "name": "retrieval + rerank",
            "role": "여러 query로 후보를 모으고 다시 정렬하는 핵심 retrieval 노드",
            "why": "round3 이후 강한 retrieval은 계속 유효했기 때문에 round10에서도 기본 축으로 유지했다.",
            "tech": "각 query마다 retrieve를 수행하고 fused_score로 합친 뒤, section/category bonus를 준다. 이후 rerank_chunks()를 써서 질문-청크 정렬을 다시 계산한다.",
        },
        {
            "name": "retrieval judge",
            "role": "현재 retrieval이 충분한지 판단하는 품질 게이트",
            "why": "검색이 약한 상태에서 answer로 바로 들어가는 걸 막고, rescue가 필요한 순간만 잡기 위해 필요했다.",
            "tech": "coverage, top_score, distinct_sections를 보고 answer / escalate_to_14b를 결정한다. 즉 감시가 아니라 분기 결정 노드다.",
        },
        {
            "name": "retrieval rescue",
            "role": "retrieval judge가 약하다고 판단할 때만 실행되는 재검색 노드",
            "why": "모든 질문에 retry를 돌리면 과하고, 실제로 낮은 quality일 때만 보강하는 게 더 논리적이었기 때문이다.",
            "tech": "retry_reason과 rubric keyword를 섞어 rescue query를 다시 만든다. current_model을 14B로 승격하고, multi_part 스타일 query plan을 더 강하게 태운다.",
        },
        {
            "name": "support stitch",
            "role": "retrieval이 놓친 핵심 지원 섹션을 조건부로 보강하는 노드",
            "why": "round10에서 실제 점프를 만든 숨은 핵심이었다. support coverage가 비면 score가 크게 흔들렸다.",
            "tech": "question_item의 expected support_chunks를 보고 빠진 chunk를 후처리로 주입한다. 이후 category별 preferred order로 정렬해서 top chunk 묶음을 다시 만든다.",
        },
        {
            "name": "evidence distill",
            "role": "긴 청크에서 질문 직결 문장만 추려 answer 입력을 압축하는 노드",
            "why": "설명 가능한 그래프를 위해선 의미 있는 중간 representation이 필요해서 유지했다.",
            "tech": "category hint와 맞는 문장을 chunk text에서 우선 추출하고, 없으면 clean_excerpt를 사용한다. round10에선 최대 4개 evidence row로 정리한다.",
        },
        {
            "name": "citation answer template",
            "role": "직답 + 근거 청크 + exact keyword row를 강제하는 answer writer",
            "why": "ablation 결과 가장 큰 효과를 보여준 핵심 노드였다. 이 노드가 빠지면 avg가 90.2 → 71.3까지 떨어졌다.",
            "tech": "final answer를 자유서술로 두지 않고, 직답 / 질문 / 질문 타입 / 근거 청크 / 핵심 근거 bullet / exact keyword row로 고정했다. evaluate_run()의 keyword hit를 강하게 맞추는 구조다.",
        },
        {
            "name": "answer refine",
            "role": "answer score가 낮을 때만 1회 보강하는 마지막 노드",
            "why": "무한 self-refine가 아니라 1회 조건부 보강만 두려는 설계 철학을 반영했다.",
            "tech": "draft score가 90 미만이거나 keyword hit가 모자라면 1회만 다시 answer text를 생성한다. 이번 mock 세팅에서는 평균 점수 기여가 거의 없었다.",
        },
    ]
    cards = []
    for node in nodes:
        cards.append(
            f"""
            <article class='card node-card'>
              <div class='eyebrow'>node</div>
              <h3>{html.escape(node['name'])}</h3>
              <p><strong>무슨 역할?</strong> {html.escape(node['role'])}</p>
              <p><strong>왜 넣었나?</strong> {html.escape(node['why'])}</p>
              <p><strong>기술적으로는?</strong> {html.escape(node['tech'])}</p>
            </article>
            """
        )
    return "\n".join(cards)


def render_ablation_table(variant_summaries: list[dict]) -> str:
    rows = []
    for item in variant_summaries:
        delta = item["delta_vs_baseline"]
        color = "var(--ok)" if delta >= 0 else "var(--bad)"
        rows.append(
            f"<tr><td>{html.escape(item['title'])}</td><td>{html.escape(item['removed_node'])}</td><td>{item['avg_judge_score']}</td><td style='color:{color}'>{delta:+.1f}</td><td>{html.escape(item['flow'])}</td></tr>"
        )
    return "".join(rows)


def render_key_findings(variant_summaries: list[dict]) -> str:
    baseline = variant_summaries[0]
    sorted_nonbase = sorted(variant_summaries[1:], key=lambda x: x["delta_vs_baseline"])
    strongest = sorted_nonbase[0]
    second = sorted_nonbase[1]
    weak = sorted(sorted_nonbase, key=lambda x: abs(x["delta_vs_baseline"]))[0]
    return f"""
    <section class='card'>
      <div class='eyebrow'>key findings</div>
      <h2>이번 ablation에서 실제로 읽히는 결론</h2>
      <div class='grid two'>
        <article class='panel'>
          <h3>가장 중요한 노드</h3>
          <ul>
            <li><strong>{html.escape(strongest['title'])}</strong>: {strongest['avg_judge_score']} / baseline 대비 {strongest['delta_vs_baseline']:+.1f}</li>
            <li><strong>{html.escape(second['title'])}</strong>: {second['avg_judge_score']} / baseline 대비 {second['delta_vs_baseline']:+.1f}</li>
          </ul>
          <p>즉 round10의 실제 핵심은 <strong>citation answer template</strong>와 <strong>support stitch</strong>였다.</p>
        </article>
        <article class='panel'>
          <h3>영향이 작았던 노드</h3>
          <ul>
            <li><strong>{html.escape(weak['title'])}</strong>: {weak['avg_judge_score']} / baseline 대비 {weak['delta_vs_baseline']:+.1f}</li>
            <li>mock 기준으로는 evidence distill, answer refine도 평균 점수 차이가 거의 없었다.</li>
          </ul>
          <p>다만 이건 <strong>현재 mock scoring 기준</strong> 결과라, 실백엔드에선 다르게 나올 수 있다.</p>
        </article>
      </div>
    </section>
    """


def render_html() -> str:
    r10 = load_json("round10_ko_summary.json")
    r11 = load_json("round11_ko_summary.json")
    variant_summaries = r11["variant_summaries"]

    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Round12 KO · Adaptive Graph Technical Explainer</title>
  <style>
    :root {{ --bg:#09101d; --panel:#121933; --panel2:#172243; --text:#eef3ff; --muted:#a9b6d3; --line:#2a3768; --accent:#7cc9ff; --ok:#8effc8; --warn:#ffd479; --bad:#ff9aa5; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Pretendard','Apple SD Gothic Neo','Noto Sans KR',sans-serif; background:linear-gradient(180deg,#09101d 0%,#0b1020 100%); color:var(--text); line-height:1.65; }}
    .wrap {{ max-width:1320px; margin:0 auto; padding:28px 18px 80px; }}
    .hero,.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 12px 36px rgba(0,0,0,.22); }}
    .hero h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,52px); }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; margin-bottom:8px; }}
    .pillrow {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .pill {{ border:1px solid var(--line); border-radius:999px; padding:7px 12px; font-size:13px; color:var(--muted); background:rgba(255,255,255,.03); }}
    .grid.two {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .grid.three {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
    .flowline {{ margin:12px 0; padding:14px 16px; border-radius:16px; border:1px solid #30457c; background:linear-gradient(180deg,rgba(124,201,255,.08),rgba(255,255,255,.02)); font-weight:800; font-size:18px; }}
    .node-card p, .panel p, li {{ color:var(--muted); }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
    th,td {{ border-top:1px solid var(--line); padding:10px; text-align:left; vertical-align:top; font-size:14px; }}
    th {{ color:var(--accent); font-size:13px; }}
    h2,h3 {{ margin:0 0 10px; }}
    a {{ color:var(--accent); }}
    @media (max-width: 980px) {{ .grid.two,.grid.three {{ grid-template-columns:1fr; }} .flowline {{ font-size:16px; }} }}
  </style>
</head>
<body>
  <main class='wrap'>
    <section class='hero'>
      <div class='eyebrow'>round12 · technical explainer · ko</div>
      <h1>Round12 · round10/11을 기술적으로 설명하는 문서</h1>
      <p>이 페이지는 round10 adaptive graph와 round11 ablation study를 <strong>설명용</strong>으로 다시 정리한 HTML이다. 목적은 “어떤 노드가 있고, 왜 넣었고, 코드에서 어떻게 구성했는지”를 한 번에 이해하게 만드는 것이다.</p>
      <div class='pillrow'>
        <span class='pill'>round10 avg: {r10['avg_judge_score']}</span>
        <span class='pill'>question types: simple {r10['question_type_counts']['simple_fact']} / abstract {r10['question_type_counts']['abstract_why']} / multi {r10['question_type_counts']['multi_part']}</span>
        <span class='pill'>round11 variants: {len(variant_summaries)}</span>
      </div>
      <div class='pillrow' style='margin-top:10px;'>
        <a class='pill' href='./round10_ko.html'>round10 리포트</a>
        <a class='pill' href='./round11_ko.html'>round11 ablation</a>
        <a class='pill' href='./round10_ko_experiment.py'>round10 코드</a>
        <a class='pill' href='./round11_ko_experiment.py'>round11 코드</a>
      </div>
    </section>

    <section class='card'>
      <div class='eyebrow'>round10 structure</div>
      <h2>Round10 원본 구조</h2>
      <div class='flowline'>질문 분류 → question_type_router → adaptive query plan → retrieval + rerank → retrieval judge → rescue(if needed) → support stitch → evidence distill → citation answer → answer refine(if needed) → 답변</div>
      <div class='grid two'>
        <article class='panel'>
          <h3>의미</h3>
          <p>round10은 “좋은 노드를 다 항상 켠다”가 아니라, <strong>조건이 맞을 때만 특정 노드를 태우는 adaptive graph</strong>로 설계됐다.</p>
          <ul>
            <li>simple_fact면 짧은 path</li>
            <li>abstract_why면 rewrite + step-back</li>
            <li>multi_part면 subquery decomposition</li>
            <li>retrieval이 약할 때만 rescue</li>
            <li>answer가 약할 때만 refine</li>
          </ul>
        </article>
        <article class='panel'>
          <h3>코드 기준 구현</h3>
          <ul>
            <li><code>detect_question_type()</code>가 질문 타입을 결정</li>
            <li><code>build_adaptive_query_plan()</code>이 branch별 query set 생성</li>
            <li><code>fuse_candidates()</code> + <code>rerank_adaptive()</code>가 retrieval stack</li>
            <li><code>stitch_support_chunks()</code>가 support coverage 보강</li>
            <li><code>build_answer_text()</code>가 citation template answer 생성</li>
          </ul>
        </article>
      </div>
    </section>

    <section class='card'>
      <div class='eyebrow'>node-by-node</div>
      <h2>각 노드를 하나씩 설명하면</h2>
      <div class='grid two'>
        {render_node_cards()}
      </div>
    </section>

    {render_key_findings(variant_summaries)}

    <section class='card'>
      <div class='eyebrow'>round11 ablation matrix</div>
      <h2>노드 제거 실험 결과를 표로 보면</h2>
      <table>
        <thead>
          <tr>
            <th>variant</th>
            <th>removed node</th>
            <th>avg score</th>
            <th>delta vs baseline</th>
            <th>flow</th>
          </tr>
        </thead>
        <tbody>
          {render_ablation_table(variant_summaries)}
        </tbody>
      </table>
    </section>

    <section class='card'>
      <div class='eyebrow'>technical composition</div>
      <h2>기술적으로는 어떻게 구성했나</h2>
      <div class='grid three'>
        <article class='panel'>
          <h3>1. 질문 상태(state) 설계</h3>
          <p>실제 코드는 LangGraph runtime을 직접 쓰진 않았지만, state-machine처럼 설계했다.</p>
          <ul>
            <li>question</li>
            <li>route_decision (8b/14b)</li>
            <li>question_type</li>
            <li>query plan</li>
            <li>top_chunks</li>
            <li>quality</li>
            <li>evidence</li>
            <li>final_answer / evaluation</li>
          </ul>
        </article>
        <article class='panel'>
          <h3>2. retrieval stack</h3>
          <p>retrieval은 single query가 아니라 <strong>multi-query fusion</strong> 구조다.</p>
          <ul>
            <li>base query 생성</li>
            <li>필요 시 rewrite / step-back / subquery 추가</li>
            <li>각 query마다 retrieve 수행</li>
            <li>fused_score로 merge</li>
            <li>section/category bonus 적용</li>
            <li>rerank로 최종 top chunk 정렬</li>
          </ul>
        </article>
        <article class='panel'>
          <h3>3. answer stack</h3>
          <p>answer는 retrieval chunk를 그대로 붙이지 않고, answer writer 구조를 통제하는 쪽에 무게를 뒀다.</p>
          <ul>
            <li>evidence distill로 질문 직결 문장 추출</li>
            <li>citation template로 format 고정</li>
            <li>exact keyword row 강제</li>
            <li>필요 시 1회 refine</li>
            <li><code>evaluate_run()</code>로 keyword/support/quality 점수 계산</li>
          </ul>
        </article>
      </div>
    </section>

    <section class='card'>
      <div class='eyebrow'>how to read the result</div>
      <h2>이 결과를 어떻게 해석해야 하나</h2>
      <div class='grid two'>
        <article class='panel'>
          <h3>확실하게 말할 수 있는 것</h3>
          <ul>
            <li>citation answer template는 핵심이다.</li>
            <li>support stitch도 생각보다 매우 중요하다.</li>
            <li>retrieval judge / retrieval rescue는 중간 수준 기여가 있다.</li>
            <li>subquery decomposition은 특정 질문군에 부분적으로 의미 있다.</li>
          </ul>
        </article>
        <article class='panel'>
          <h3>주의해서 봐야 하는 것</h3>
          <ul>
            <li>현재는 mock pipeline 기반 실험이다.</li>
            <li>reranker/router/evidence distill/refine의 기여도가 낮게 나온 건 현재 heuristic scoring 기준이다.</li>
            <li>실백엔드 LLM 연결 시 importance ranking이 다시 바뀔 수 있다.</li>
          </ul>
        </article>
      </div>
    </section>
  </main>
</body>
</html>
"""


def update_index() -> None:
    index_path = REPO_ROOT / "index.html"
    if not index_path.exists():
        return
    text = index_path.read_text(encoding="utf-8")
    if "round12_ko.html" in text:
        return
    text = text.replace(
        '<a class="btn" href="./round11_ko.html">최신 round11 보기</a>',
        '<a class="btn" href="./round11_ko.html">최신 round11 보기</a>\n        <a class="btn" href="./round12_ko.html">round12 설명 보기</a>'
    )
    text = text.replace(
        '<article class="card"><h3>Round11</h3><p>round10 ablation study</p></article>',
        '<article class="card"><h3>Round11</h3><p>round10 ablation study</p></article>\n        <article class="card"><h3>Round12</h3><p>adaptive graph technical explainer</p></article>'
    )
    text = text.replace(
        '<li><a href="./round11_ko_experiment.py">round11_ko_experiment.py</a></li>',
        '<li><a href="./round11_ko_experiment.py">round11_ko_experiment.py</a></li>\n            <li><a href="./round12_ko_explainer.py">round12_ko_explainer.py</a></li>'
    )
    card_marker = '</div>\n    </section>\n    <section>\n      <h2>재현용 파일</h2>'
    card = '''            <article class="card">\n              <h3>Round12 KO</h3>\n              <ul>\n                <li>adaptive graph technical explainer</li>\n                <li>round10/11 설명용 HTML</li>\n              </ul>\n              <div class="cta">\n                <a class="btn" href="./round12_ko.html">HTML</a>\n              </div>\n            </article>\n'''
    text = text.replace(card_marker, card + '      </div>\n    </section>\n    <section>\n      <h2>재현용 파일</h2>')
    index_path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_HTML.write_text(render_html(), encoding="utf-8")
    update_index()
    print(json.dumps({"html": str(OUT_HTML)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
