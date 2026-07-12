# RAG LangGraph 실험 스택 재구축 가이드

## 문서 목적

이 문서는 현재 `rag-langgraph-benefits` 저장소에서 사용 중인 **한국어 RAG 실험 스택**을 다른 환경에서 다시 만들 수 있도록, 코드 구조를 가능한 한 자세하게 풀어쓴 재구축 문서다.

단순히 "무슨 파일이 있다" 수준이 아니라 아래를 모두 담는다.

- 왜 이런 구조로 설계했는지
- 각 라운드에서 무엇이 추가되었는지
- 현재 최종 구조가 어떤 컴포넌트들로 구성되는지
- 질문이 들어오면 어떤 순서로 처리되는지
- 점수는 어떻게 계산되는지
- HTML 리포트는 어떤 철학으로 만들었는지
- 새 데이터셋, 특히 **한국어 PDF**로 갈아탈 때 무엇을 바꿔야 하는지
- 실제 LLM 실험과 현재 mock 실험의 차이

이 문서만 있으면, 다른 리포지토리나 다른 머신에서도 동일한 구조의 RAG 평가 파이프라인을 재구현할 수 있도록 작성한다.

---

## 1. 저장소의 역할

저장소 경로:

- `/Users/hakvision/work-rag-langgraph-benefits`

원격 저장소:

- `https://github.com/hakvision/rag-langgraph-benefits`

이 저장소는 "실서비스용 RAG 서버"가 아니라, **RAG 구조를 실험하고 설명 가능한 형태로 시각화하는 실험 저장소**에 가깝다.

핵심 산출물은 다음 3가지다.

1. **실험 스크립트 (`round*.py`)**
   - 각 라운드의 파이프라인 구현
2. **머신 리더블 결과물 (`artifacts/*.json`)**
   - 질문별 점수, 청크, 흐름, 근거, 요약
3. **사람이 읽는 HTML 리포트 (`round*.html`)**
   - 질문별 결과와 구조적 차이를 눈으로 볼 수 있게 정리

즉, 이 프로젝트는 단순 답변 생성보다도 아래를 더 중요하게 둔다.

- **재현성**
- **구조 설명 가능성**
- **라운드별 비교 가능성**
- **질문별 retrieval evidence 추적 가능성**
- **정답 기준 기반 평가 가능성**

---

## 2. 전체 진화 과정 요약

이 저장소의 구조는 한 번에 완성된 것이 아니라 라운드를 거치며 확장되었다.

### Round1
기본 lexical retrieval + quality gate + answer + judge

### Round2
reranker 추가

### Round3
hybrid retrieval / fusion 강화

### Round4
retrieval judge 와 answer judge 분리

### Round5
query rewrite + step-back

### Round6
subquery decomposition + multi-query fusion

### Round7
evidence distill

### Round8
citation-constrained answer template

### Round9
judge-guided self-refine

### Round10
위에서 검증된 좋은 요소들을 조합한 **adaptive combination graph**

### Round11
round10 구조의 **ablation study**

### Round12
round10/11 구조를 설명하는 기술 explainer HTML

### Round13
새 **한국어 PDF** 데이터셋에 대해, round10 계열의 핵심 구조를 가져와 다시 태운 PDF branch

현재 "우리가 만든 구조"라고 부를 때는 실질적으로 아래 둘을 뜻한다.

1. **개념적 최종 구조**: round10 adaptive graph
2. **새 PDF 적용 브랜치**: `round13_ko_pdf_experiment.py`

따라서 재구축 문서는 round10을 핵심 아키텍처로 설명하고, round13을 "새 PDF에 이 구조를 이식한 구현 예시"로 설명하는 것이 맞다.

---

## 3. 현재 재구축 시 우선 봐야 할 파일

### 핵심 파일

- `round1_ko_experiment.py`
- `round2_ko_experiment.py`
- `round10_ko_experiment.py`
- `round11_ko_experiment.py`
- `round13_ko_pdf_experiment.py`
- `index.html`

### 파일 역할

#### `round1_ko_experiment.py`
기초 골격을 제공한다.

- 토크나이즈
- 청크 추출
- 청크 스코어링
- retrieval
- 질문 복잡도 분류
- 기본 검색 쿼리 생성
- 검색 품질 평가
- baseline answer 생성
- judged evaluation
- HTML report 렌더링

즉 **모든 이후 라운드의 공통 뼈대**다.

#### `round2_ko_experiment.py`
`rerank_chunks()`를 제공한다.

즉 retrieval 후 후보를 다시 정렬하는 계층을 분리했다.

#### `round10_ko_experiment.py`
현재 구조의 핵심인 adaptive graph를 구현한다.

- question type router
- adaptive query plan
- multi-query fusion
- adaptive rerank
- retrieval rescue
- support stitch
- evidence distill
- citation answer
- answer refine

#### `round11_ko_experiment.py`
round10에서 어떤 노드가 실제로 중요한지 확인하는 ablation 실험이다.

- router 제거
- abstract branch 제거
- subquery 제거
- reranker 제거
- retrieval judge 제거
- retrieval rescue 제거
- support stitch 제거
- evidence distill 제거
- citation template 제거
- answer refine 제거

이 파일은 구조를 이해할 때 매우 중요하다. 왜냐하면 **어떤 노드가 필수이고 어떤 노드는 장식인지**를 실험적으로 보여주기 때문이다.

#### `round13_ko_pdf_experiment.py`
새 한국어 PDF 데이터셋 적용 브랜치다.

round10의 구조를 거의 그대로 가져오되, 소스 문서가 웹 HTML이 아니라 **실제 PDF**라는 점이 다르다.

즉 이 파일은 다음을 보여준다.

- 새 문서 소스 교체 방법
- PDF 텍스트 추출 방법
- 수동 섹션 분할 방법
- 질문/정답 세트 교체 방법
- 동일 평가 파이프라인 재사용 방법

#### `index.html`
GitHub Pages의 루트 랜딩 페이지다.

이 프로젝트에서는 단순히 파일을 푸시하는 것만으로는 충분하지 않다. 사용자가 실제로 새 산출물을 찾을 수 있어야 하므로, 새로운 설명 페이지나 라운드를 만들면 이 landing page에서도 연결해줘야 한다.

---

## 4. 설계 철학

이 프로젝트의 설계 철학은 다음 6개로 요약할 수 있다.

### 4.1 설명 가능한 RAG

단순히 답변만 생성하는 것이 아니라,

- 어떤 쿼리를 만들었는지
- 어떤 청크를 가져왔는지
- 왜 이 청크가 선택되었는지
- 언제 8B에서 14B로 승격했는지
- 왜 refine를 한 번 더 했는지

를 모두 남긴다.

### 4.2 round-based experimentation

한 번에 완벽한 구조를 만들려 하지 않고,

- 노드 1개 추가
- 분기 1개 추가
- judge 1개 분리
- prompt template 1개 교체

식으로 라운드를 나눠 비교 가능하게 만든다.

### 4.3 judged report first

이 저장소의 점수는 BLEU/ROUGE 같은 일반 텍스트 metric이 아니라,

- 사전에 정의한 gold answer
- 키워드 세트
- 반드시 잡아야 하는 support chunk
- retrieval quality

를 기반으로 **human-auditable rubric**으로 계산한다.

### 4.4 HTML is a first-class artifact

결과를 JSON으로만 두지 않고 HTML로 시각화한다.

이유는 다음과 같다.

- 사용자가 빠르게 훑어볼 수 있음
- 구조 변화가 round 단위로 눈에 띔
- 질문별 근거 청크와 score를 한 화면에서 확인 가능
- GitHub Pages로 바로 공유 가능

### 4.5 mock와 real backend를 분리해서 말하기

이 저장소는 현재 여러 라운드에서 `qwen3:8b`, `qwen3:14b`라는 이름을 쓰지만, 모든 라운드가 실제 로컬 추론을 돌리는 것은 아니다.

특히 round13 PDF branch는 **template-based mock synthesis**다.

즉,

- 구조 검증
- retrieval 흐름 검증
- judged evaluation 구조 검증

은 가능하지만,

- 실제 생성 품질 비교
- 실제 latency 측정
- 실제 LLM hallucination 비교

는 아니다.

이 구분을 문서와 HTML에 명확히 적는 이유는, polished report가 실제 모델 벤치마크처럼 오해되는 것을 막기 위해서다.

### 4.6 새 데이터셋으로 바꿀 때 구조는 유지하고 source adapter만 바꾼다

새 문서가 들어오면 바뀌는 것은 주로 아래다.

- source fetch 방식
- source chunking 방식
- 질문/정답 세트
- category hint / section hint

반면 아래는 최대한 유지한다.

- classify_question
- detect_question_type
- adaptive query plan 철학
- retrieval → rerank → judge → answer → evaluate 흐름
- summary/report artifact schema

즉 이 구조는 "문서 소스 독립적인 평가 프레임워크"를 목표로 한다.

---

## 5. 현재 최종 구조를 한 문장으로 요약하면

현재 최종 구조는 다음과 같이 요약할 수 있다.

> 질문을 난이도와 유형에 따라 분기하고, 질문 유형에 맞춰 복수 검색 쿼리를 생성한 뒤, 가중 fusion과 rerank로 후보 청크를 정렬하고, 검색 품질을 판정해 필요하면 rescue/승격을 수행한 후, 핵심 근거를 압축하여 citation-aware answer를 만들고, gold-answer rubric으로 다시 채점하는 explainable judged RAG pipeline이다.

---

## 6. round10/13 기준 최종 파이프라인 흐름

아래는 구현 기준의 실제 흐름이다.

1. source document 준비
2. source를 chunk 단위로 분할
3. 질문/정답 세트 준비
4. 각 질문마다
   1. 질문 복잡도 분류 (`classify_question`)
   2. 질문 타입 분류 (`detect_question_type`)
   3. adaptive query plan 생성 (`build_adaptive_query_plan`)
   4. 각 query별 retrieval 수행 (`retrieve`)
   5. 결과를 fusion (`fuse_candidates`)
   6. fusion 결과 rerank (`rerank_chunks` 또는 `rerank_adaptive` 계열)
   7. support chunk 보강 (`stitch_support_chunks`)
   8. 검색 품질 판정 (`grade_search`)
   9. 필요 시 rescue/14B 승격 후 재시도
   10. 근거 압축 (`distill_evidence`)
   11. citation template 기반 answer 작성 (`build_answer_text`)
   12. judged evaluation (`evaluate_run`)
   13. 점수가 낮으면 1회 refine
5. run summary 생성 (`build_summary`)
6. JSON/HTML artifact 생성 (`render_report`)
7. index landing page 갱신 (필요 시)

---

## 7. round1에서 가져온 기초 골격

`round1_ko_experiment.py`는 사실상 베이스 프레임워크다.

### 핵심 함수 목록

- `tokenize`
- `clean_excerpt`
- `fetch_source_document`
- `extract_chunks`
- `score_chunk`
- `retrieve`
- `classify_question`
- `build_search_query`
- `grade_search`
- `build_answer`
- `evaluate_run`
- `run_one`
- `render_report`
- `main`

### 왜 이 골격이 중요한가

후속 라운드들이 바뀌어도, 결국 아래 기본 cycle은 유지된다.

- query 생성
- retrieval
- 품질 판정
- answer 생성
- judged evaluation
- HTML report 생성

즉 round10/13도 복잡해 보이지만, 본질적으로 round1의 확장판이다.

---

## 8. 질문 분류 계층

질문 분류는 2단계로 이뤄진다.

### 8.1 모델 lane 분류: `classify_question`

이 함수는 질문을 먼저 `8b` 또는 `14b` 경로로 보낸다.

#### 입력
- 질문 문자열

#### 출력
- route: `"8b"` 또는 `"14b"`
- analysis dict
  - token_count
  - matched_hard_terms
  - complexity_score
  - reason
  - source

#### 판단 기준
구현은 heuristic이다.

예를 들어 아래 조건으로 complexity를 올린다.

- 토큰 수가 많다
- 질문에 `왜`, `어떻게`, `권고`, `시사점`, `차이`, `비교` 같은 단어가 있다
- `보상`, `투명성`, `법적`, `대안` 같은 reasoning-heavy 용어가 있다

#### 왜 이렇게 했나
실제 작은 모델/큰 모델 라우팅을 흉내 내기 위해서다.

- simple factual question → 8B lane
- multi-step reasoning question → 14B lane

현재는 mock heuristic이지만, 나중에 real backend를 붙일 때도 이 레이어는 그대로 쓸 수 있다.

### 8.2 question type 분류: `detect_question_type`

route와 별개로 질문을 다음 세 타입으로 나눈다.

- `simple_fact`
- `abstract_why`
- `multi_part`

#### 왜 8b/14b만으로는 부족한가
같은 14B lane이라도 필요한 query plan이 다르기 때문이다.

예를 들어,

- 단순 정의형은 focus query만 있으면 충분
- 왜/의미/법적 평가형은 rewrite + step-back이 좋음
- 복합 질의는 subquery decomposition이 유리함

즉 **질문 난이도 라우팅**과 **질문 구조 라우팅**을 분리한 것이다.

---

## 9. query plan 계층

핵심 함수:

- `build_search_query`
- `build_adaptive_query_plan`

### 9.1 `build_search_query`

기본 검색 쿼리를 만든다.

#### 8B 경로
- 질문 토큰 중 앞부분을 짧게 사용
- 단순하고 압축된 쿼리

#### 14B 경로
- 질문 토큰을 더 넓게 사용
- 특정 유형이면 추가 힌트를 붙임
  - 권고/대안 → `표준화`, `플랫폼`
  - 투명성/보상 → `옵트아웃`, `권리`
  - 의미/시사점 → `법적`, `평가`, `정당성`

#### 설계 의도
8B는 "짧고 거친 query", 14B는 "더 넓고 설명적인 query"를 흉내 낸다.

### 9.2 `build_adaptive_query_plan`

이 함수가 round10/13 구조의 핵심이다.

기본적으로 `base` query를 만들고, 질문 타입에 따라 추가 branch를 붙인다.

#### simple_fact
- `focus`

#### abstract_why
- `rewrite`
- `step_back`

#### multi_part
- `subquery_1`
- `subquery_2`

#### retry 상황
- `retrieval_rescue`

각 plan item은 다음 필드를 가진다.

- `label`
- `query`
- `weight`
- `branch`

#### 왜 plan을 list of query로 만들었는가
한 번의 retrieval로 모든 질문을 잡기 어렵기 때문이다.

특히 아래 질문들은 단일 쿼리로 약하다.

- 이유를 묻는 추상 질문
- 항목이 여러 개인 복합 질문
- 문서의 결론/시사점 같은 후반부 섹션 질문

따라서 **질문을 여러 투영(projection)으로 변환해서 검색한 뒤 합치는 방식**이 더 안정적이다.

---

## 10. retrieval 계층

핵심 함수:

- `tokenize`
- `score_chunk`
- `retrieve`

### 10.1 `tokenize`

정규표현식으로 한글/영문/숫자 토큰을 추출하고 불용어를 제거한다.

#### 특징
- 한국어 조사/일반 stopword 제거
- 1글자 비숫자 제거
- lower-case 기준 처리

#### 왜 단순 토크나이저를 유지했나
이 프로젝트는 production dense retriever가 아니라 **구조 실험 저장소**이기 때문이다.

즉 retrieval 품질 절대값보다 다음이 더 중요하다.

- 어떤 변화가 어떤 영향이 있는지
- branch와 fusion이 실제로 improvement를 주는지
- judged score가 어떤 구조에서 좋아지는지

### 10.2 `score_chunk`

청크 텍스트와 section name 안에서 query token hit를 세어 점수를 준다.

주요 아이디어:

- text hit 가중치
- section hit 보너스
- 여러 token이 동시에 hit될 때 추가 보너스

#### 왜 section hit를 중요하게 보는가
질문이 문서 섹션 제목과 잘 맞는 경우가 많기 때문이다.

예를 들어
- `정책 권고`
- `결론 및 시사점`
- `법적 지위`

같은 section label은 retrieval에서 강한 prior가 된다.

### 10.3 `retrieve`

각 chunk에 대해 `score_chunk`를 적용해 정렬하고 상위 N개를 뽑는다.

출력 chunk 구조는 다음과 같다.

- `chunk_id`
- `section`
- `text`
- `preview`
- `score`
- `term_hits`

추가로 stats도 만든다.

- `query_tokens`
- `top_score`
- `distinct_sections`
- `candidate_count`

#### 왜 stats를 별도로 남기나
나중에 retrieval judge에서 활용하기 위해서다.

---

## 11. fusion 계층

핵심 함수:

- `adaptive_section_bonus`
- `fuse_candidates`

### 11.1 `adaptive_section_bonus`

질문 category와 chunk section 간의 의미적 궁합을 보정한다.

예를 들어 round13 PDF branch에서는

- `overview`는 K01
- `tdm`은 K02
- `compensation`은 K03
- `transparency`는 K04
- `output-status`는 K05
- `recommendations`는 K06
- `implication`은 K07

와의 정합도가 높다.

즉 단순 텍스트 매칭 외에 **문서 구조적 prior**를 반영한다.

### 11.2 `fuse_candidates`

각 query branch에서 상위 청크를 뽑고, 동일 chunk에 대한 점수를 누적한다.

점수는 대략 아래 성분으로 이뤄진다.

- retrieval score
- query weight
- rank-based bonus
- section/category bonus
- hint word bonus

그리고 최종 fused chunk에는 다음 메타데이터가 붙는다.

- `source_queries`
- `branch_hits`
- `term_hits`
- `fused_score`

#### 왜 이 구조가 필요한가
좋은 청크는 보통 여러 쿼리에서 반복적으로 등장한다.

예를 들어 `정책 권고` 관련 질문에서는,

- base query
- subquery
- rescue query

모두에서 K06이 떠오를 수 있다.

이 반복성을 신뢰 신호로 해석해 점수를 누적한다.

즉 이는 간단한 RRF류 사고방식을 heuristic 형태로 구현한 것이다.

---

## 12. rerank 계층

### round2의 의미
`round2_ko_experiment.py`에서 처음 `rerank_chunks()` 계층이 등장한다.

이전에는 retrieval score 순서만 믿었지만, 이후부터는 "후보를 다시 정렬하는 계층"이 별도로 존재하게 됐다.

### round10/13의 rerank
round13에서는 `rerank_chunks(question_item, candidates, model_name)`로 구현되어 있다.

rerank score는 대략 다음을 더한다.

- fused retrieval score
- adaptive section bonus
- rubric exact hit 수

#### 왜 rerank가 필요한가
retrieval은 보통 recall-oriented다.

즉 관련 후보를 넓게 모으는 데는 좋지만, 최종 answer에 가장 좋은 청크 순서를 보장하지는 않는다.

rerank는 이 후보를 answer-friendly ordering으로 다시 정렬한다.

#### 왜 rubric keyword도 rerank에 반영했는가
이 저장소는 judged evaluation이 매우 중요하므로,

- retrieval relevance
- 평가 기준 적합성

을 동시에 잡아야 한다.

즉 단순 검색 relevance가 아니라 **이 질문의 expected answer를 잘 뒷받침하는 청크**가 위로 오르도록 만든다.

---

## 13. support stitch 계층

핵심 함수:

- `stitch_support_chunks`

이 함수는 judged evaluation에서 "이 질문에 꼭 잡아야 하는 support chunk"가 누락되면, 해당 chunk를 강제로 보강한다.

예를 들어 질문에 `support_chunks = ["K06"]`이 있는데 retrieval 결과 상위권에 K06이 없으면,

- chunk map에서 K06을 찾아
- score 0으로라도 selected list에 붙인다

### 왜 이런 노드를 넣었나
round11 ablation에서 support stitch 제거 시 점수 하락이 컸다.

즉 이 저장소에서는 support stitch가 단순 치트가 아니라,

- retrieval recall의 빈틈을 보완하고
- judged report의 근거 coverage를 보장하며
- answer가 문서 핵심 section을 놓치지 않게 하는

중요한 안정화 계층이라는 뜻이다.

### 언제 특히 유용한가
- 문서 섹션 구조가 뚜렷할 때
- gold rubric이 분명할 때
- 질문별 핵심 support chunk를 미리 정할 수 있을 때

---

## 14. retrieval quality gate

핵심 함수:

- `grade_search`

이 함수는 retrieval 결과가 충분히 괜찮은지 평가한다.

주요 입력:

- question
- selected chunks
- top score
- distinct sections
- model lane

주요 계산:

- `coverage`: 질문 토큰이 chunk hit와 얼마나 겹치는가
- `top_score`
- `distinct_sections`

결과 action:

- `answer`
- `escalate_to_14b`

### 왜 judge를 retrieval 단계와 answer 단계로 나눴나
round4 이후 철학이다.

저품질 answer의 원인이 항상 answer synthesis에 있는 것은 아니다.

원인은 보통 둘 중 하나다.

1. 애초에 검색이 안 좋았다
2. 검색은 괜찮았는데 답변 구성 방식이 약했다

이 둘을 분리하지 않으면 개선 포인트가 흐려진다.

### 왜 8B에서만 승격을 허용하나
현재 구조상 8B는 cheap first pass, 14B는 richer second pass 역할이다.

따라서
- 8B가 얕으면 → 14B 승격
- 14B도 약하면 → 현재 evidence로 answer

이라는 구조로 단순화했다.

---

## 15. retrieval rescue 계층

round10의 중요한 특징 중 하나는 무조건 모든 노드를 돌리지 않는다는 점이다.

즉 rescue는 기본 노드가 아니라 **조건부 노드**다.

### 동작 방식
- 첫 retrieval pass가 약하다
- `retry_reason`을 품질 게이트에서 받는다
- `build_adaptive_query_plan(..., retry_reason=...)`으로 rescue query를 추가한다
- current model을 `14b`로 승격해 한 번 더 retrieval한다

### 왜 항상 rescue하지 않나
질문이 쉬우면 rescue는 오버헤드만 늘린다.

따라서 현재 구조는

- 쉬운 질문은 짧은 경로
- 어려운 질문만 긴 경로

를 목표로 한다.

이것이 adaptive graph의 핵심 가치다.

---

## 16. evidence distill 계층

핵심 함수:

- `distill_evidence`
- `clean_excerpt`

retrieved chunk 전체를 그대로 answer에 넣으면 너무 길고 noisy하다.

따라서 answer 전에 핵심 근거를 짧은 리스트로 압축한다.

출력 구조:

- `chunk_id`
- `section`
- `text` (짧은 excerpt)

### 왜 필요한가
round7에서 도입된 노드로, answer 단계가 "chunk dump"가 되는 것을 막아준다.

이 노드가 없으면 answer template가 좋아도 다음 문제가 생긴다.

- 불필요하게 장문이 됨
- 핵심이 흐려짐
- judge keyword hit는 나와도 사람이 읽기 어려움

즉 evidence distill은 readability와 answer controllability를 동시에 위한 계층이다.

---

## 17. answer 작성 계층

핵심 함수:

- `build_answer_text`

현재 answer는 자유 생성이 아니라 **template-based answer assembly**에 가깝다.

주요 구성 요소:

- 직답 (`DIRECT_ANSWERS` 기반)
- 원 질문
- 질문 타입
- 근거 청크 ID 목록
- 모드/quality 표기
- 핵심 근거 bullet
- exact keyword row
- 필요 시 refine note
- mock/real 한계 고지

### 왜 이렇게 템플릿화했나
round8 이후 철학 때문이다.

질문이 judged rubric으로 평가된다면, answer도 아래를 명시적으로 드러내는 편이 유리하다.

- 답의 핵심 문장
- 어떤 근거를 썼는지
- 어떤 키워드를 충족해야 하는지
- 현재 모드가 mock인지 real인지

이는 단순 생성 성능보다 **감사 가능성(auditability)** 을 높인다.

### `DIRECT_ANSWERS`의 의미
현재 mock branch에서는 실제 생성 대신 gold answer를 중심으로 직답을 조립한다.

이것은 smoke test에서는 유용하지만, real benchmark에서는 교체해야 한다.

실백엔드로 갈 때는 `DIRECT_ANSWERS`를 제거하거나, 최소한 answer 초안 생성만 LLM에게 맡기고 평가는 별도로 유지하는 방향이 바람직하다.

---

## 18. answer refine 계층

round9에서 중요한 아이디어는 "한 번 더 고친다"이다.

round13에서도 이 아이디어가 이어진다.

### 현재 동작
- draft answer를 평가한다
- `total_score < 92`이면
- `build_answer_text(..., answer_score_before=...)`로 한 번 더 조립한다
- 최종 점수를 다시 계산한다

### 왜 1회만 허용하나
무한 루프를 막고, 구조를 단순하게 유지하기 위해서다.

즉 이 프로젝트는 agentic loop를 실험하지만, 과도한 self-refine로 복잡도를 폭증시키지는 않는다.

### refine의 목적
- 누락된 키워드 노출
- answer template 강화
- judged score 보정

---

## 19. judged evaluation 계층

핵심 함수:

- `evaluate_run`

이 함수는 각 질문의 최종 답변을 점수화한다.

### 입력
- `question_item`
- `final_answer`
- `top_chunks`
- `quality`

### 계산 요소

#### 1) keyword score
- rubric keyword를 answer가 얼마나 포함하는가
- 현재 가중치 55

#### 2) support score
- expected support chunk를 실제 top chunks가 잡았는가
- 현재 가중치 25

#### 3) quality score
- retrieval coverage 기반
- 현재 가중치 20

총점:

- `keyword_score + support_score + quality_score`

### verdict 기준
- 90 이상: 좋음
- 75 이상: 무난
- 55 이상: 아쉬움
- 그 미만: 미흡

### 왜 이 평가가 좋은가
이 방식은 단순 문자열 유사도보다 훨씬 디버깅 친화적이다.

예를 들어 점수가 낮아도 원인이 분리된다.

- keyword가 부족했다
- support chunk를 못 잡았다
- retrieval coverage가 낮았다

즉 점수는 낮지만 "왜 낮았는지"를 설명할 수 있다.

---

## 20. PDF branch에서 source adapter가 어떻게 바뀌었는가

`round13_ko_pdf_experiment.py`는 round10 철학을 유지하되, source adapter를 PDF용으로 교체한 브랜치다.

### 20.1 문서 다운로드: `ensure_pdf`

역할:
- `datasets/` 디렉터리 준비
- PDF가 없으면 URL에서 다운로드

왜 필요한가:
- source를 로컬에 고정해 재현성을 확보하기 위해
- 실험 시점마다 원문 링크 상태에 의존하지 않기 위해

### 20.2 텍스트 추출: `extract_pdf_text`

역할:
- `PdfReader`로 페이지별 텍스트 추출
- 노이즈 제거 (`clean_text`)
- 추출 텍스트를 artifact 파일로 저장

왜 추출 텍스트도 저장하나:
- chunk 경계를 사후 감사할 수 있게 하기 위해
- PDF 파서가 어떤 텍스트를 실제로 읽었는지 기록으로 남기기 위해

### 20.3 수동 section split: `SECTION_SPECS` + `build_chunks`

PDF는 HTML처럼 `<h2>`를 안정적으로 추출할 수 없기 때문에, start/end marker 기반으로 명시적 구간 분할을 한다.

예:
- `1. 개요` ~ `2. 주요내용1) ...`
- `2) AI 학습 이용에 대한 저작자의 권리 및 보상` ~ `3) AI법과 투명성 의무`
- `3. 결론 및 시사점` ~ `참 고 자 료`

### 왜 marker-based split을 썼나
실제 PDF branch에서 중요한 것은 완벽한 일반화보다 **이 문서에 대한 안정적 재실행**이었다.

즉
- 빠르게 검증 가능하고
- 사람이 읽어도 납득 가능한 section 경계
- retrieval과 judged score가 흔들리지 않는 방식

을 우선했다.

---

## 21. 질문/정답 세트 설계

round13의 질문 세트는 아래 원칙으로 설계했다.

1. 문서의 핵심 섹션을 골고루 덮는다
2. 단순 fact / abstract why / multi-part를 섞는다
3. support chunk를 명확히 지정할 수 있어야 한다
4. gold answer가 너무 길지 않되 채점 가능해야 한다

각 question item 구조는 다음과 같다.

- `label`
- `category`
- `question`
- `gold_answer`
- `rubric_keywords`
- `support_chunks`

### 왜 category가 필요한가
category는 단순 분류용이 아니라, 아래 여러 계층에서 사용된다.

- `detect_question_type`
- `CATEGORY_HINTS`
- `SECTION_HINTS`
- `adaptive_section_bonus`
- `DIRECT_ANSWERS`

즉 category는 이 파이프라인의 **제어축(control axis)** 이다.

---

## 22. artifact 설계

round13에서 생성되는 주요 산출물은 다음과 같다.

- `round13_ko_pdf.html`
- `artifacts/round13_ko_pdf_results.json`
- `artifacts/round13_ko_pdf_summary.json`
- `artifacts/round13_ko_pdf_questions.json`
- `artifacts/round13_ko_pdf_source_document.json`
- `artifacts/round13_ko_pdf_extracted.txt`
- `datasets/genai_copyright_report_ko.pdf`

### 왜 이렇게 나눴나

#### HTML
사람이 읽는 결과

#### results.json
질문별 상세 실행 결과

#### summary.json
대시보드/비교용 요약

#### questions.json
채점 타깃을 외부에서 검증 가능하게 하기 위함

#### source_document.json
chunk map과 source metadata를 남기기 위함

#### extracted.txt
PDF 파싱 결과를 raw에 가깝게 보관하기 위함

#### dataset pdf
source immutability 보장

이 구조는 재구축 시에도 그대로 유지하는 것이 좋다.

---

## 23. HTML report 설계 철학

핵심 함수:

- `render_report`

HTML report는 단순 꾸밈이 아니라 구조 설명 도구다.

### round13 report의 주요 섹션

1. hero
   - 라운드 이름
   - 문서 이름
   - 평균 점수
   - rescue 횟수
   - answer revision 횟수

2. source chunk map
   - chunk id
   - section
   - preview

3. 질문별 카드
   - Expected answer
   - Final answer
   - Judge
   - Flow
   - Retrieved chunks
   - Evidence distill

### 왜 이런 구성인가
사용자가 알고 싶은 것은 보통 아래 3가지다.

1. 전체적으로 잘 됐나?
2. 각 질문에서 왜 그 점수가 나왔나?
3. 어떤 청크를 가져왔나?

따라서 report는 이 3가지 질문에 바로 답하는 구조로 설계했다.

---

## 24. round11 ablation이 주는 구조적 해석

`round11_ko_experiment.py`는 단순 부록이 아니라, 어떤 노드가 진짜 중요한지 알려주는 증거다.

특히 다음 해석이 중요하다.

### support stitch는 중요하다
지원 청크 보강이 없으면 점수가 크게 흔들린다.

### retrieval judge / rescue는 가치가 있다
낮은 품질 검색을 그냥 answer로 보내지 않는 것이 안정성에 도움된다.

### evidence distill / citation template / refine는 answer quality layer다
retrieval이 맞아도 answer formatting과 grounding 표현 방식이 약하면 최종 judged score가 떨어진다.

즉 현재 구조는 단순 retriever 하나가 아니라,

- retrieval quality layer
- grounding layer
- answer control layer
- judged evaluation layer

로 계층화된 시스템으로 보는 것이 맞다.

---

## 25. 이 구조를 다른 곳에서 재구축할 때 최소 구현 순서

다른 팀이나 다른 저장소에서 이 구조를 다시 만든다면 아래 순서를 추천한다.

### 단계 1: baseline부터 복제
먼저 아래만 구현한다.

- source fetch / load
- chunk build
- tokenize
- lexical retrieve
- simple quality gate
- judged evaluation
- html report

즉 round1 수준부터 만든다.

### 단계 2: reranker 계층 추가
- fused candidate 없이도 우선 rerank부터 분리

### 단계 3: question type router 추가
- simple_fact
- abstract_why
- multi_part

### 단계 4: adaptive query plan 추가
- rewrite
- step_back
- subquery

### 단계 5: fusion 추가
- query별 점수 누적
- source_queries 기록

### 단계 6: retrieval rescue + support stitch 추가
- quality 낮으면 재시도
- 핵심 support chunk 보강

### 단계 7: evidence distill + citation template + refine 추가
- answer readability와 judged score 개선

### 단계 8: ablation framework 추가
- 노드 제거 시 delta를 볼 수 있게

이 순서가 좋은 이유는, 디버깅이 쉽기 때문이다.

---

## 26. 새 문서/새 도메인으로 바꿀 때 바꿔야 하는 것

### 반드시 바꿔야 하는 것
- `PDF_URL` 또는 source path
- `PDF_TITLE` / source metadata
- `SECTION_SPECS` 또는 HTML chunking logic
- `QUESTIONS`
- `DIRECT_ANSWERS`
- `CATEGORY_HINTS`
- `SECTION_HINTS`
- `support_chunks`

### 가능하면 유지할 것
- `classify_question`
- `detect_question_type`의 큰 틀
- `build_adaptive_query_plan` 패턴
- `fuse_candidates`
- `grade_search`
- `distill_evidence`
- `evaluate_run` schema
- `render_report` 레이아웃 철학

### 언제 재설계가 필요한가
아래 경우는 구조 일부를 다시 설계해야 한다.

1. 문서가 장편/수백 페이지라 marker split이 불가능한 경우
2. 표/도표/수식 중심 문서인 경우
3. 질문이 자유 생성형이라 support chunk를 미리 못 정하는 경우
4. 실제 LLM generation quality가 주평가 대상인 경우

---

## 27. 실제 LLM backend로 바꿀 때 바뀌는 포인트

현재 mock 기반 구조를 real backend로 전환하려면 다음을 바꿔야 한다.

### 27.1 answer 생성
현재:
- `DIRECT_ANSWERS` + template assembly

변경 후:
- selected chunk를 context로 넣고 LLM이 answer 초안 생성
- citation template는 유지 가능

### 27.2 router
현재:
- heuristic `classify_question`

변경 후:
- small model classifier 또는 규칙 기반 + telemetry

### 27.3 reranker
현재:
- heuristic rerank

변경 후:
- real reranker model / cross-encoder / LLM ranker

### 27.4 quality gate
현재:
- token coverage + top score

변경 후:
- retrieval confidence, answerability classifier, judge model 등 사용 가능

### 27.5 mock disclosure 제거 금지
실백엔드로 전환해도, mock와 real을 혼동하지 않도록 아래 필드는 계속 남기는 것이 좋다.

- provider_mode
- backend_available
- models

---

## 28. 실제로 재현 실행하는 예시

현재 PDF branch는 다음처럼 실행했다.

```bash
cd /Users/hakvision/work-rag-langgraph-benefits
uv run --with pypdf python round13_ko_pdf_experiment.py
```

이 명령의 장점:

- 전역 패키지 설치 불필요
- `pypdf`를 ephemeral dependency로 사용 가능
- macOS 기본 Python 환경을 더럽히지 않음

### 왜 `uv run --with pypdf`를 선택했나
환경에 `pdftotext`, `pdfinfo`, `mutool`, `pymupdf`, `PyPDF2`가 없었기 때문이다.

따라서 PDF branch에서 중요한 운영 원칙은 다음과 같다.

> 시스템 전역 설치가 막혀 있으면, 우선 `uv run --with ...`로 필요한 최소 의존성만 주입해서 artifact를 만든다.

---

## 29. GitHub Pages 배포 원칙

이 저장소는 GitHub Pages가 이미 살아 있다.

루트:

- `https://hakvision.github.io/rag-langgraph-benefits/`

이 프로젝트에서 새 설명 사이트를 배포할 때 중요한 원칙은 다음과 같다.

1. 새 HTML 파일만 만들지 말 것
2. `index.html` 루트 랜딩에서도 접근 가능하게 만들 것
3. 푸시 후 실제 live URL을 열어 검증할 것
4. 200 응답만 보지 말고, title/hero text까지 확인할 것

즉 "깃푸시했다"가 완료 조건이 아니다.

---

## 30. 이 구조의 장점

### 장점 1: 매우 설명 가능하다
각 질문별로
- query
- selected chunk
- flow
- evidence
- score

를 모두 남긴다.

### 장점 2: round-based 개선이 쉽다
노드 하나 추가하고 점수 변화를 비교하기 쉽다.

### 장점 3: mock 상태에서도 구조 회귀 테스트가 가능하다
실모델이 없어도
- retrieval 흐름
- judged schema
- HTML artifact
- source adapter

를 검증 가능하다.

### 장점 4: 새 문서에 이식이 쉽다
source adapter와 QA set만 바꾸면 같은 파이프라인을 재사용할 수 있다.

---

## 31. 이 구조의 한계

### 한계 1: 현재 retrieval이 heuristic lexical 중심이다
실제 dense retriever보다 단순하다.

### 한계 2: answer가 아직 진짜 생성형이 아니다
특히 round13은 template-based mock synthesis다.

### 한계 3: support stitch가 강한 prior를 주기 때문에, pure retrieval benchmark로 보기 어렵다
이는 구조적 안정화에는 좋지만, strict retrieval purity 측정에는 불리하다.

### 한계 4: PDF chunking이 범용적이지 않다
현재는 marker-based split이라 특정 문서에는 매우 잘 맞지만, 범용 문서 파서라고 보긴 어렵다.

---

## 32. 다른 환경에서 그대로 따라 만들기 위한 구현 체크리스트

### 필수 파일
- `round1`에 해당하는 baseline skeleton
- `round10`에 해당하는 adaptive graph implementation
- PDF branch용 source adapter 파일
- HTML report template

### 필수 데이터 구조
- `QUESTIONS`
- `Chunk`
- run result dict
- summary dict

### 필수 함수
- `tokenize`
- `retrieve`
- `classify_question`
- `detect_question_type`
- `build_adaptive_query_plan`
- `fuse_candidates`
- `rerank_chunks`
- `stitch_support_chunks`
- `grade_search`
- `distill_evidence`
- `build_answer_text`
- `evaluate_run`
- `render_report`

### 필수 artifact
- source raw/extracted text
- source chunk metadata
- question set json
- results json
- summary json
- html report

---

## 33. 권장 재구현 형태

다른 곳에서 처음부터 다시 만든다면 아래 패키지 구조를 추천한다.

```text
rag_project/
  datasets/
  artifacts/
  docs/
  src/
    source_adapters/
      html_source.py
      pdf_source.py
    retrieval/
      tokenize.py
      lexical.py
      fusion.py
      rerank.py
    routing/
      classify.py
      question_type.py
      query_plan.py
    answer/
      evidence.py
      template.py
      refine.py
    eval/
      judged.py
    reports/
      html_report.py
  experiments/
    round1.py
    round10.py
    round13_pdf.py
```

현재 저장소는 round별 단일 스크립트 구조라 빠르게 실험하기엔 좋지만, 장기적으로는 위처럼 모듈화하면 유지보수성이 더 높다.

---

## 34. 마지막 요약

현재 우리가 만든 구조의 본질은 아래와 같다.

1. **질문을 먼저 분류한다**
2. **질문 유형에 따라 검색 전략을 바꾼다**
3. **여러 검색 결과를 합쳐 더 강한 후보 집합을 만든다**
4. **후보를 다시 정렬한다**
5. **검색 품질이 낮으면 rescue/승격을 수행한다**
6. **반드시 필요한 support chunk를 보강한다**
7. **근거를 압축해 사람이 읽기 쉬운 answer로 만든다**
8. **gold answer 기반 rubric으로 점수를 다시 매긴다**
9. **모든 과정을 HTML/JSON artifact로 남긴다**

즉 이 저장소는 단순한 RAG 데모가 아니라,

> 구조를 반복 실험하고, 각 노드의 역할을 비교하고, 새 데이터셋에 빠르게 이식하며, 결과를 사람이 읽을 수 있는 HTML로 설명하는 judged RAG experimentation framework

라고 보는 것이 가장 정확하다.

---

## 35. 관련 파일 경로

### 이번에 추가된 PDF branch 관련
- `/Users/hakvision/work-rag-langgraph-benefits/round13_ko_pdf_experiment.py`
- `/Users/hakvision/work-rag-langgraph-benefits/round13_ko_pdf.html`
- `/Users/hakvision/work-rag-langgraph-benefits/rag_architecture_rebuild_guide_ko.md`

### 참고 round 스크립트
- `/Users/hakvision/work-rag-langgraph-benefits/round1_ko_experiment.py`
- `/Users/hakvision/work-rag-langgraph-benefits/round2_ko_experiment.py`
- `/Users/hakvision/work-rag-langgraph-benefits/round10_ko_experiment.py`
- `/Users/hakvision/work-rag-langgraph-benefits/round11_ko_experiment.py`
- `/Users/hakvision/work-rag-langgraph-benefits/round12_ko_explainer.py`

### 참고 artifact
- `/Users/hakvision/work-rag-langgraph-benefits/artifacts/round13_ko_pdf_results.json`
- `/Users/hakvision/work-rag-langgraph-benefits/artifacts/round13_ko_pdf_summary.json`
- `/Users/hakvision/work-rag-langgraph-benefits/artifacts/round13_ko_pdf_questions.json`
- `/Users/hakvision/work-rag-langgraph-benefits/artifacts/round13_ko_pdf_source_document.json`
- `/Users/hakvision/work-rag-langgraph-benefits/artifacts/round13_ko_pdf_extracted.txt`

---

## 36. 재현용 명령 모음

```bash
cd /Users/hakvision/work-rag-langgraph-benefits
uv run --with pypdf python round13_ko_pdf_experiment.py
```

```bash
python3 -m http.server 8000
```

```bash
git status --short --branch
git add .
git commit -m "docs: add deep RAG architecture rebuild guide and site"
git push origin main
```

---

## 37. 구현시 반드시 기억할 운영 메모

- mock 실험이면 mock라고 반드시 적을 것
- PDF source는 로컬로 저장한 뒤 실행할 것
- extracted text를 artifact로 남길 것
- questions.json을 먼저 만들어 evaluation target을 고정할 것
- GitHub Pages에 올릴 때는 `index.html`에서 접근 가능하게 만들 것
- live URL을 직접 열어 title/hero text까지 확인할 것

이 6가지는 이 프로젝트에서 반복적으로 중요했던 운영 규칙이다.
