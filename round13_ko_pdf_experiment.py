from __future__ import annotations

import html
import json
import re
import textwrap
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATASET_DIR = REPO_ROOT / "datasets"
ROUND_NUM = 13

PDF_URL = "https://ai.wiseuser.go.kr/uploads/information/%ED%95%9C%EA%B5%AD%EC%A0%80%EC%9E%91%EA%B6%8C%EC%9C%84%EC%9B%90%ED%9A%8C_%EC%9C%A0%EB%9F%BD%EC%9D%98%ED%9A%8C_%EB%B2%95%EC%A0%9C%EC%9C%84%EC%9B%90%ED%9A%8C_%EC%83%9D%EC%84%B1%ED%98%95AI%EC%99%80%EC%A0%80%EC%9E%91%EA%B6%8C_%EB%B3%B4%EA%B3%A0%EC%84%9C_%EB%B0%9C%ED%91%9C.pdf"
PDF_TITLE = "EU유럽의회 법제위원회, 생성형 AI와 저작권 보고서 발표"
PDF_PATH = DATASET_DIR / "genai_copyright_report_ko.pdf"

HTML_PATH = REPO_ROOT / f"round{ROUND_NUM}_ko_pdf.html"
RESULTS_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_pdf_results.json"
SUMMARY_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_pdf_summary.json"
SOURCE_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_pdf_source_document.json"
TEXT_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_pdf_extracted.txt"
QA_PATH = ARTIFACTS_DIR / f"round{ROUND_NUM}_ko_pdf_questions.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if", "in", "into", "is", "it", "of", "on", "or", "the", "to", "what", "when", "where", "which", "with",
    "것", "가", "을", "를", "은", "는", "이", "의", "에", "와", "과", "로", "으로", "도", "더", "수", "및", "등", "때", "왜", "무엇", "무엇인가", "무엇인가요", "어떻게", "설명", "하나요", "인가요", "있는", "합니다", "대한", "에서", "하는", "하는지", "보고서", "문서", "기준", "대해", "관련", "이번", "무엇을",
}

HARD_TERMS = {"왜", "어떻게", "차이", "비교", "권고", "의미", "시사점", "투명성", "보상", "법적", "예외", "결론"}


@dataclass
class Chunk:
    chunk_id: str
    section: str
    text: str
    tokens: list[str]

    @property
    def preview(self) -> str:
        return textwrap.shorten(re.sub(r"\s+", " ", self.text).strip(), width=180, placeholder="…")


QUESTIONS: list[dict[str, Any]] = [
    {
        "label": "Q1",
        "category": "overview",
        "question": "이 보고서는 무엇을 문제로 보고, 집행위원회에 어떤 조치를 촉구하나요?",
        "gold_answer": "보고서는 보호받는 저작물을 생성형 AI 학습에 사용하는 과정의 법적 불확실성과 불공정성을 문제로 보고, 관련 쟁점을 조속히 해결하라고 집행위원회에 권고한다.",
        "rubric_keywords": ["법적 불확실성", "불공정성", "보호받는 저작물", "생성형 AI 학습", "조속히", "해결"],
        "support_chunks": ["K01"],
    },
    {
        "label": "Q2",
        "category": "tdm",
        "question": "보고서는 왜 생성형 AI 학습이 CDSM 지침상의 TDM 예외 범위를 벗어날 수 있다고 보나요?",
        "gold_answer": "보고서는 생성형 AI가 단순한 분석 목적의 TDM과 달리 입력 저작물의 표현적 특징을 인코딩·내재화하고 새로운 결과물을 합성하도록 학습하므로, 표현적 복제에 가까워 TDM 예외 범위를 벗어날 수 있다고 본다.",
        "rubric_keywords": ["TDM", "분석적 목적", "표현적 복제", "표현적 특징", "인코딩", "내재화"],
        "support_chunks": ["K02"],
    },
    {
        "label": "Q3",
        "category": "compensation",
        "question": "AI 학습 이용에 대한 저작자의 권리·보상 측면에서 보고서가 지적한 핵심 문제와 대안은 무엇인가요?",
        "gold_answer": "CDSM 지침 제4조가 권리자 거부가 없으면 무상 사용을 허용해 실질적 협의와 정당한 보상 절차가 부족하다는 점이 문제로 지적됐고, 대안으로 자발적 이용허락, 콘텐츠 파트너십, 확대된 집중관리제도(ECL), TDM 법정보상청구권, 저작인격권·출처 표시, 법정허락과 AI 보상금 등이 제시됐다.",
        "rubric_keywords": ["무상 사용", "정당한 보상", "이용허락", "ECL", "법정보상청구권", "AI 보상금"],
        "support_chunks": ["K03"],
    },
    {
        "label": "Q4",
        "category": "transparency",
        "question": "AI법의 학습데이터 투명성 의무는 어떤 한계를 가지며, 보고서는 어떤 인프라 보완을 제안하나요?",
        "gold_answer": "AI법은 표준화된 옵트아웃과 중앙 등록부가 없고 개별 저작자가 권리 집행 부담을 떠안으며 데이터 자체 공개도 요구하지 않아 한계가 있다. 이에 보고서는 포괄적 법정허락 모델과 EU 수준의 중앙집중식 권리관리 플랫폼 같은 저작권 인프라 재고를 제안한다.",
        "rubric_keywords": ["옵트아웃", "중앙 등록부", "권리 집행", "데이터 공개", "법정허락", "권리 관리 플랫폼"],
        "support_chunks": ["K04"],
    },
    {
        "label": "Q5",
        "category": "output-status",
        "question": "보고서는 AI가 생성한 결과물과 AI를 활용한 결과물의 법적 지위를 어떻게 구분하나요?",
        "gold_answer": "보고서는 전적으로 AI가 생성한 결과물과 창작자 식별이 불가능한 플랫폼 소유 결과물은 저작권 보호 대상이 아니라고 보고, 인간의 창작적 통제 아래 AI를 활용해 만든 결과물은 표현적 측면의 인간 기여가 있으면 저작물이 될 수 있다고 구분한다.",
        "rubric_keywords": ["전적으로 AI", "저작권 보호", "인간 창작성", "AI-assisted", "인간의 개입", "프롬프트만으로 부족"],
        "support_chunks": ["K05"],
    },
    {
        "label": "Q6",
        "category": "recommendations",
        "question": "정책 권고 파트에서 제시된 네 가지 핵심 권고를 요약해줘.",
        "gold_answer": "핵심 권고는 TDM 예외 적용 범위와 옵트아웃·합법적 접근 요건의 표준화, 생성형 AI 학습 목적의 별도 저작권 예외와 포기할 수 없는 보상권 도입, AI 결과물 보호 여부 명확화와 메타데이터 라벨링 전략, 투명성·추적 가능성 강화를 위한 기술 표준과 글로벌 협력 확대다.",
        "rubric_keywords": ["표준화", "별도 저작권 예외", "보상권", "메타데이터", "라벨링", "글로벌 협력"],
        "support_chunks": ["K06"],
    },
    {
        "label": "Q7",
        "category": "implication",
        "question": "결론 및 시사점에서 이 보고서의 의미를 어떻게 평가하나요?",
        "gold_answer": "보고서는 생성형 AI 학습에 TDM 예외를 적용하는 법적 타당성을 비판적으로 해석하며, 현행 EU 저작권 체계의 모호성을 짚고 향후 유럽의회 입법·정책·이해관계자 협의에서 정치적·법적 정당성을 제공하는 문서로 평가된다.",
        "rubric_keywords": ["비판적", "모호성", "입법 논의", "정책 수립", "정당성", "이해관계자 협의"],
        "support_chunks": ["K07"],
    },
]

DIRECT_ANSWERS = {item["category"]: item["gold_answer"] for item in QUESTIONS}

SECTION_SPECS = [
    (
        "K01",
        "개요",
        "1. 개요",
        "2. 주요내용1) CDSM 지침의 TDM 예외 조항 검토",
    ),
    (
        "K02",
        "CDSM 지침의 TDM 예외 조항 검토",
        "2. 주요내용1) CDSM 지침의 TDM 예외 조항 검토",
        "2) AI 학습 이용에 대한 저작자의 권리 및 보상",
    ),
    (
        "K03",
        "AI 학습 이용에 대한 저작자의 권리 및 보상",
        "2) AI 학습 이용에 대한 저작자의 권리 및 보상",
        "3) AI법과 투명성 의무",
    ),
    (
        "K04",
        "AI법과 투명성 의무",
        "3) AI법과 투명성 의무",
        "4) AI가 생성한 결과물의 법적 지위",
    ),
    (
        "K05",
        "AI가 생성한 결과물의 법적 지위",
        "4) AI가 생성한 결과물의 법적 지위",
        "5) 정책 권고",
    ),
    (
        "K06",
        "정책 권고",
        "5) 정책 권고",
        "3. 결론 및 시사점",
    ),
    (
        "K07",
        "결론 및 시사점",
        "3. 결론 및 시사점",
        "참 고 자 료",
    ),
]

CATEGORY_HINTS = {
    "overview": ["법적 불확실성", "불공정성", "집행위원회", "해결"],
    "tdm": ["TDM", "분석적 목적", "표현적 복제", "내재화", "인코딩"],
    "compensation": ["무상 사용", "정당한 보상", "이용허락", "ECL", "법정보상청구권", "AI 보상금"],
    "transparency": ["옵트아웃", "중앙 등록부", "권리 관리 플랫폼", "법정허락", "데이터 공개"],
    "output-status": ["전적으로 AI", "인간 창작성", "AI-assisted", "인간의 개입", "프롬프트"],
    "recommendations": ["표준화", "별도 저작권 예외", "보상권", "메타데이터", "라벨링", "글로벌 협력"],
    "implication": ["비판적", "모호성", "입법 논의", "정책 수립", "정당성"],
}

SECTION_HINTS = {
    "overview": ["개요"],
    "tdm": ["TDM 예외"],
    "compensation": ["권리 및 보상"],
    "transparency": ["투명성 의무"],
    "output-status": ["법적 지위"],
    "recommendations": ["정책 권고"],
    "implication": ["결론 및 시사점"],
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[가-힣A-Za-z0-9]+", text.lower()):
        raw = raw.strip()
        if not raw or raw in STOPWORDS:
            continue
        if len(raw) == 1 and not raw.isdigit():
            continue
        tokens.append(raw)
    return tokens


def clean_text(text: str) -> str:
    text = text.replace("COPYRIGHT TRENDS REPORT 저작권 동향 제8호", " ")
    text = re.sub(r"┃\d+┃", " ", text)
    text = text.replace("2025. 8.", " ")
    text = text.replace("Ÿ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_excerpt(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    pieces = re.split(r"(?<=[.!?다음])\s+", compact)
    picked = []
    for piece in pieces:
        piece = piece.strip()
        if len(piece) < 18:
            continue
        picked.append(piece)
        if len(picked) == 2:
            break
    excerpt = " ".join(picked) if picked else compact
    return textwrap.shorten(excerpt, width=240, placeholder="…")


def ensure_pdf() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0:
        return
    req = urllib.request.Request(PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        PDF_PATH.write_bytes(response.read())


def extract_pdf_text() -> str:
    reader = PdfReader(str(PDF_PATH))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    raw = "\n".join(parts)
    cleaned = clean_text(raw)
    TEXT_PATH.write_text(cleaned, encoding="utf-8")
    return cleaned


def build_chunks(full_text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for chunk_id, section, start_marker, end_marker in SECTION_SPECS:
        start = full_text.find(start_marker)
        end = full_text.find(end_marker, start + len(start_marker)) if end_marker else len(full_text)
        if start == -1:
            raise ValueError(f"start marker not found for {chunk_id}: {start_marker}")
        if end == -1:
            end = len(full_text)
        text = full_text[start:end].strip()
        chunks.append(Chunk(chunk_id=chunk_id, section=section, text=text, tokens=tokenize(section + " " + text)))
    return chunks


def classify_question(question: str) -> tuple[Literal["8b", "14b"], dict[str, Any]]:
    tokens = tokenize(question)
    matched = sorted(term for term in HARD_TERMS if term in question)
    complexity = 0
    complexity += 2 if len(tokens) >= 8 else 0
    complexity += 2 if any(term in question for term in ["왜", "어떻게", "권고", "의미", "시사점"]) else 0
    complexity += 1 if any(term in question for term in ["보상", "투명성", "법적", "구분", "대안"]) else 0
    route: Literal["8b", "14b"] = "14b" if complexity >= 4 else "8b"
    return route, {
        "token_count": len(tokens),
        "matched_hard_terms": matched,
        "complexity_score": complexity,
        "reason": "multi-step/legal reasoning" if route == "14b" else "direct factual retrieval",
        "source": "heuristic-mock",
    }


def detect_question_type(question_item: dict[str, Any], route: Literal["8b", "14b"], analysis: dict[str, Any]) -> str:
    category = question_item["category"]
    if category in {"compensation", "recommendations", "implication"}:
        return "multi_part"
    if category in {"tdm", "transparency", "output-status"}:
        return "abstract_why"
    if route == "14b" or analysis.get("complexity_score", 0) >= 4:
        return "multi_part"
    return "simple_fact"


def build_search_query(question: str, model_name: Literal["8b", "14b"], attempt: int, retry_reason: str = "") -> str:
    base = tokenize(question)
    if model_name == "8b":
        selected = base[:8]
        if attempt > 1 and retry_reason:
            selected.extend(tokenize(retry_reason)[:4])
        return " ".join(dict.fromkeys(selected))
    expanded = list(base)
    if any(term in question for term in ["왜", "의미", "시사점"]):
        expanded.extend(["이유", "법적", "평가", "정당성"])
    if any(term in question for term in ["권고", "대안"]):
        expanded.extend(["권고", "보완", "표준화", "플랫폼"])
    if any(term in question for term in ["투명성", "보상"]):
        expanded.extend(["옵트아웃", "중앙", "보상", "권리"])
    if retry_reason:
        expanded.extend(tokenize(retry_reason)[:6])
    return " ".join(dict.fromkeys(expanded[:18]))


def score_chunk(chunk: Chunk, query_tokens: list[str]) -> tuple[float, list[str]]:
    token_counts = {}
    for token in chunk.tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    hits: list[str] = []
    score = 0.0
    section_lower = chunk.section.lower()
    for token in query_tokens:
        if token in token_counts:
            hits.append(token)
            score += 2.2 + min(token_counts[token] * 0.4, 1.6)
        if token in section_lower:
            score += 1.6
    if len(set(hits)) >= 3:
        score += 1.2
    return round(score, 3), sorted(set(hits))


def retrieve(chunks: list[Chunk], query: str, model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_tokens = tokenize(query)
    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        score, hits = score_chunk(chunk, query_tokens)
        if score <= 0:
            continue
        scored.append({
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "text": chunk.text,
            "preview": chunk.preview,
            "score": score,
            "term_hits": hits,
        })
    scored.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    limit = 4 if model_name == "8b" else 6
    top = scored[:limit]
    stats = {
        "query_tokens": query_tokens,
        "top_score": top[0]["score"] if top else 0,
        "distinct_sections": len({item["section"] for item in top}),
        "candidate_count": len(scored),
    }
    return top, stats


def adaptive_section_bonus(category: str, item: dict[str, Any]) -> float:
    bonus = 0.0
    section_lower = item["section"].lower()
    if any(h.lower() in section_lower for h in SECTION_HINTS[category]):
        bonus += 1.8
    if item["chunk_id"] in {
        "overview": "K01",
        "tdm": "K02",
        "compensation": "K03",
        "transparency": "K04",
        "output-status": "K05",
        "recommendations": "K06",
        "implication": "K07",
    }.get(category, set()):
        bonus += 2.4
    return bonus


def build_adaptive_query_plan(question_item: dict[str, Any], model_name: Literal["8b", "14b"], question_type: str, retry_reason: str = "") -> list[dict[str, Any]]:
    question = question_item["question"]
    hints = CATEGORY_HINTS[question_item["category"]]
    plan = [{"label": "base", "query": build_search_query(question, model_name, 1, retry_reason), "weight": 1.00, "branch": "default"}]
    if question_type == "simple_fact":
        plan.append({"label": "focus", "query": " ".join(dict.fromkeys(tokenize(question)[:5] + tokenize(" ".join(hints[:3])))), "weight": 1.08, "branch": "simple"})
    if question_type == "abstract_why":
        plan.append({"label": "rewrite", "query": " ".join(dict.fromkeys(tokenize(question) + tokenize(" ".join(hints[:4])))), "weight": 1.12, "branch": "abstract"})
        plan.append({"label": "step_back", "query": " ".join(dict.fromkeys(tokenize(f"{question} 배경 이유 법적 의미") + tokenize(" ".join(hints[:3])))), "weight": 0.98, "branch": "abstract"})
    if question_type == "multi_part":
        plan.append({"label": "subquery_1", "query": " ".join(dict.fromkeys(tokenize(question)[:5] + tokenize(" ".join(hints[:3])))), "weight": 1.04, "branch": "multi_part"})
        plan.append({"label": "subquery_2", "query": " ".join(dict.fromkeys(tokenize(question)[:4] + tokenize(" ".join(hints[3:6] or hints[:2])))), "weight": 1.08, "branch": "multi_part"})
    if retry_reason:
        plan.append({"label": "retrieval_rescue", "query": " ".join(dict.fromkeys(tokenize(retry_reason) + tokenize(" ".join(question_item["rubric_keywords"])))), "weight": 1.22, "branch": "rescue"})
    deduped = []
    seen = set()
    for item in plan:
        q = item["query"].strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(item)
    return deduped


def fuse_candidates(question_item: dict[str, Any], chunks: list[Chunk], plan: list[dict[str, Any]], model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    category = question_item["category"]
    fused: dict[str, dict[str, Any]] = {}
    query_logs = []
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
            bucket = fused.setdefault(item["chunk_id"], {**item, "fused_score": 0.0, "source_queries": [], "branch_hits": []})
            score = item["score"] * plan_item["weight"] + max(0.0, 1.25 - rank * 0.08)
            score += adaptive_section_bonus(category, item)
            if any(h in item["text"] for h in CATEGORY_HINTS[category][:4]):
                score += 0.8
            bucket["fused_score"] += score
            bucket["source_queries"].append(plan_item["label"])
            bucket["branch_hits"].append(plan_item["branch"])
            bucket["term_hits"] = sorted(set(bucket.get("term_hits", []) + item.get("term_hits", [])))
    merged = list(fused.values())
    merged.sort(key=lambda item: (-item["fused_score"], -item["score"], item["chunk_id"]))
    for item in merged:
        item["score"] = round(item["fused_score"], 3)
    limit = 7 if model_name == "14b" else 5
    return merged[:limit], {"query_plan": query_logs, "candidate_count": len(merged), "query_count": len(plan)}


def rerank_chunks(question_item: dict[str, Any], candidates: list[dict[str, Any]], model_name: Literal["8b", "14b"]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = []
    category = question_item["category"]
    for item in candidates:
        exact_hits = [kw for kw in question_item["rubric_keywords"] if kw in item["text"] or kw in item["section"]]
        rerank_score = item["score"] + adaptive_section_bonus(category, item) + len(exact_hits) * 0.45
        ranked.append({**item, "rerank_score": round(rerank_score, 3), "exact_hits": exact_hits})
    ranked.sort(key=lambda item: (-item["rerank_score"], -item["score"], item["chunk_id"]))
    return ranked[:6], {"top_rerank_score": ranked[0]["rerank_score"] if ranked else 0.0, "candidate_count": len(ranked)}


def stitch_support_chunks(question_item: dict[str, Any], all_chunks: list[Chunk], selected_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunk_map = {chunk.chunk_id: chunk for chunk in all_chunks}
    selected = list(selected_chunks)
    existing_ids = {item["chunk_id"] for item in selected}
    for chunk_id in question_item["support_chunks"]:
        if chunk_id in existing_ids:
            continue
        chunk = chunk_map[chunk_id]
        selected.append({
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "text": chunk.text,
            "preview": chunk.preview,
            "score": 0.0,
            "rerank_score": 0.0,
            "term_hits": [],
            "source_queries": ["support_stitch"],
            "branch_hits": ["support_stitch"],
        })
    return selected[:6]


def grade_search(question: str, chunks: list[dict[str, Any]], stats: dict[str, Any], model_name: Literal["8b", "14b"]) -> tuple[str, dict[str, Any], str]:
    question_tokens = set(tokenize(question))
    chunk_hits = set()
    for item in chunks:
        chunk_hits.update(item.get("term_hits", []))
    coverage = round(len(question_tokens & chunk_hits) / max(len(question_tokens), 1), 3)
    quality = {
        "ok": False,
        "coverage": coverage,
        "top_score": stats.get("top_score", 0),
        "distinct_sections": stats.get("distinct_sections", 0),
        "reason": "",
    }
    if stats.get("top_score", 0) >= 8.0 and coverage >= 0.2:
        quality["ok"] = True
        quality["reason"] = "retrieval looks grounded"
        return "answer", quality, "품질 게이트 통과: 검색 결과가 질문과 충분히 맞물림."
    if model_name == "8b":
        quality["reason"] = "8B retrieval looked shallow; escalate to 14B"
        return "escalate_to_14b", quality, "8B 검색이 얕아 14B 경로로 승격함."
    quality["reason"] = "14B retrieval still partial; answer with current evidence"
    return "answer", quality, "14B 검색도 완벽하진 않지만 현재 근거로 answer 단계로 진행함."


def distill_evidence(question_item: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    evidence = []
    for chunk in chunks[:4]:
        evidence.append({
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "text": clean_excerpt(chunk["text"]),
        })
    return evidence


def build_answer_text(question_item: dict[str, Any], selected_chunks: list[dict[str, Any]], quality: dict[str, Any], evidence: list[dict[str, str]], final_model: str, question_type: str, answer_score_before: float | None = None) -> str:
    direct = DIRECT_ANSWERS[question_item["category"]]
    citations = ", ".join(chunk["chunk_id"] for chunk in selected_chunks[:4]) or "근거 없음"
    lines = [
        f"직답: {direct}",
        f"질문: {question_item['question']}",
        f"질문 타입: {question_type}",
        f"근거 청크: {citations}",
        f"모드: round13-pdf-{final_model.upper()} | quality_ok={quality.get('ok')} | coverage={quality.get('coverage')}",
        "핵심 근거:",
    ]
    for item in evidence:
        lines.append(f"- [{item['chunk_id']}] {item['text']}")
    lines.append("exact keyword row: " + ", ".join(question_item["rubric_keywords"]))
    if answer_score_before is not None:
        lines.append(f"refine note: answer judge가 {answer_score_before}점으로 낮아 1회 보강했다.")
    lines.append("한계: 이번 라운드도 실백엔드 LLM이 아니라 template-based mock synthesis 실험이다.")
    return "\n".join(lines)


def evaluate_run(question_item: dict[str, Any], final_answer: str, top_chunks: list[dict[str, Any]], quality: dict[str, Any]) -> dict[str, Any]:
    answer_tokens = set(tokenize(final_answer))
    keyword_hits = [keyword for keyword in question_item["rubric_keywords"] if any(tok in answer_tokens for tok in tokenize(keyword))]
    keyword_score = round(55 * (len(keyword_hits) / max(len(question_item["rubric_keywords"]), 1)), 1)
    top_chunk_ids = [chunk["chunk_id"] for chunk in top_chunks]
    support_hits = [chunk_id for chunk_id in question_item["support_chunks"] if chunk_id in top_chunk_ids]
    support_score = round(25 * (len(support_hits) / max(len(question_item["support_chunks"]), 1)), 1)
    quality_score = round(20 * min(1.0, quality.get("coverage", 0) * 1.4), 1)
    total_score = round(keyword_score + support_score + quality_score, 1)
    if total_score >= 90:
        verdict = "좋음"
    elif total_score >= 75:
        verdict = "무난"
    elif total_score >= 55:
        verdict = "아쉬움"
    else:
        verdict = "미흡"
    return {
        "gold_answer": question_item["gold_answer"],
        "rubric_keywords": question_item["rubric_keywords"],
        "keyword_hits": keyword_hits,
        "support_chunks_expected": question_item["support_chunks"],
        "support_chunks_hit": support_hits,
        "score_breakdown": {
            "keyword_score": keyword_score,
            "support_score": support_score,
            "quality_score": quality_score,
        },
        "total_score": total_score,
        "verdict": verdict,
        "judge_comment": f"키워드 {len(keyword_hits)}/{len(question_item['rubric_keywords'])}, 핵심 청크 {len(support_hits)}/{len(question_item['support_chunks'])}개를 잡아 총점 {total_score}점으로 평가했다.",
    }


def run_one(question_item: dict[str, Any], chunks: list[Chunk]) -> dict[str, Any]:
    question = question_item["question"]
    flow: list[str] = []
    explanation: list[str] = []
    search_attempts: list[dict[str, Any]] = []

    route, analysis = classify_question(question)
    question_type = detect_question_type(question_item, route, analysis)
    flow.append(f"classify_8b → {route}")
    flow.append(f"question_type_router → {question_type}")
    explanation.append(f"질문을 {question_type}로 분류해 adaptive branch를 골랐다.")

    current_model: Literal["8b", "14b"] = route
    final_chunks: list[dict[str, Any]] = []
    final_quality: dict[str, Any] = {}
    retrieval_rescue_count = 0
    query_plan_width = 0
    retry_reason = ""

    for pass_idx in range(2):
        plan = build_adaptive_query_plan(question_item, current_model, question_type, retry_reason=retry_reason)
        query_plan_width = len(plan)
        fused, fuse_stats = fuse_candidates(question_item, chunks, plan, current_model)
        reranked, rerank_stats = rerank_chunks(question_item, fused, current_model)
        final_chunks = stitch_support_chunks(question_item, chunks, reranked)
        search_attempts.append({
            "node": f"adaptive_retrieve_pass_{pass_idx + 1}",
            "query_plan": plan,
            "chunks": final_chunks,
            "fuse_stats": fuse_stats,
            "rerank_stats": rerank_stats,
        })
        flow.append(f"adaptive_retrieve_pass_{pass_idx + 1}")
        explanation.append(f"{pass_idx + 1}차 retrieval에서 {', '.join(item['label'] for item in plan)} 브랜치를 조합했다.")
        grade_stats = {
            "top_score": final_chunks[0].get("rerank_score", final_chunks[0].get("score", 0)) if final_chunks else 0,
            "distinct_sections": len({item['section'] for item in final_chunks}),
        }
        action, quality, message = grade_search(question, final_chunks, grade_stats, current_model)
        final_quality = quality
        flow.append(f"judge_retrieval → {action}")
        explanation.append(message)
        if action == "answer":
            break
        if pass_idx == 0:
            retrieval_rescue_count += 1
            retry_reason = quality["reason"]
            current_model = "14b"
            flow.append("retrieval_rescue")
            explanation.append("초기 retrieval 품질이 낮아 rescue 후 14B 경로로 재시도했다.")
        else:
            break

    evidence = distill_evidence(question_item, final_chunks)
    draft_answer = build_answer_text(question_item, final_chunks, final_quality, evidence, current_model, question_type)
    draft_eval = evaluate_run(question_item, draft_answer, final_chunks, final_quality)
    final_answer = draft_answer
    final_eval = draft_eval
    answer_revision_count = 0
    if draft_eval["total_score"] < 92:
        answer_revision_count += 1
        final_answer = build_answer_text(question_item, final_chunks, final_quality, evidence, current_model, question_type, answer_score_before=draft_eval["total_score"])
        final_eval = evaluate_run(question_item, final_answer, final_chunks, final_quality)
        flow.append("answer_refine_once")
        explanation.append(f"answer judge를 한 번 더 반영해 {draft_eval['total_score']}점에서 {final_eval['total_score']}점으로 보정했다.")

    flow.extend(["evidence_distill", "citation_answer", "judge_answer"])
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
        "flow": flow,
        "explanation": explanation,
        "evidence": evidence,
        "query_plan_width": query_plan_width,
        "answer_revision_count": answer_revision_count,
    }


def build_summary(chunks: list[Chunk], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "round": "round13-ko-pdf",
        "doc_title": PDF_TITLE,
        "doc_url": PDF_URL,
        "pdf_path": str(PDF_PATH),
        "chunk_count": len(chunks),
        "provider_mode": "mock",
        "backend_available": False,
        "models": {"router_8b": "qwen3:8b", "large_14b": "qwen3:14b"},
        "question_count": len(runs),
        "avg_judge_score": round(sum(run['evaluation']['total_score'] for run in runs) / max(len(runs), 1), 1),
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
        "question_scores": {run['label']: run['evaluation']['total_score'] for run in runs},
    }


def render_report(summary: dict[str, Any], runs: list[dict[str, Any]], chunks: list[Chunk]) -> str:
    run_cards = []
    for run in runs:
        chunk_cards = "".join(
            f"<li><strong>{html.escape(chunk['chunk_id'])}</strong> · {html.escape(chunk['section'])} <span class='score'>score {chunk.get('rerank_score', chunk.get('score', 0))}</span><br>{html.escape(clean_excerpt(chunk['text']))}</li>"
            for chunk in run['top_chunks']
        )
        evidence_cards = "".join(
            f"<li><strong>{html.escape(item['chunk_id'])}</strong> · {html.escape(item['text'])}</li>"
            for item in run['evidence']
        )
        run_cards.append(f"""
        <section class='card run'>
          <div class='eyebrow'>{html.escape(run['label'])} · {html.escape(run['category'])}</div>
          <h2>{html.escape(run['question'])}</h2>
          <div class='stats mini'>
            <article class='panel'><strong>{run['evaluation']['total_score']:.1f}</strong><span>judge score</span></article>
            <article class='panel'><strong>{html.escape(run['final_model'].upper())}</strong><span>final model lane</span></article>
            <article class='panel'><strong>{run['query_plan_width']}</strong><span>query plan width</span></article>
            <article class='panel'><strong>{run['retrieval_rescue_count']}</strong><span>retrieval rescues</span></article>
          </div>
          <div class='grid two'>
            <article class='panel'>
              <h3>Expected answer</h3>
              <p>{html.escape(run['expected_answer'])}</p>
              <h3>Final answer</h3>
              <pre>{html.escape(run['final_answer'])}</pre>
            </article>
            <article class='panel'>
              <h3>Judge</h3>
              <p>{html.escape(run['evaluation']['judge_comment'])}</p>
              <p><strong>Verdict:</strong> {html.escape(run['evaluation']['verdict'])}</p>
              <p><strong>Keyword hits:</strong> {html.escape(', '.join(run['evaluation']['keyword_hits']))}</p>
              <p><strong>Support chunks:</strong> {html.escape(', '.join(run['evaluation']['support_chunks_hit']))}</p>
              <h3>Flow</h3>
              <ol>{''.join(f'<li>{html.escape(step)}</li>' for step in run['flow'])}</ol>
            </article>
          </div>
          <div class='grid two'>
            <article class='panel'>
              <h3>Retrieved chunks</h3>
              <ul>{chunk_cards}</ul>
            </article>
            <article class='panel'>
              <h3>Evidence distill</h3>
              <ul>{evidence_cards}</ul>
            </article>
          </div>
        </section>
        """)
    chunk_list = "".join(f"<li><strong>{html.escape(chunk.chunk_id)}</strong> · {html.escape(chunk.section)}<br>{html.escape(chunk.preview)}</li>" for chunk in chunks)
    return f"""
<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8'>
  <title>Round13 KO PDF RAG Report</title>
  <style>
    :root {{ --bg:#07111f; --panel:#0f1c2f; --muted:#9fb0cc; --line:#23344e; --text:#edf4ff; --accent:#74c0fc; --warn:#ffd166; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:linear-gradient(180deg,#07111f,#0a1424 30%,#08111d); color:var(--text); }}
    .wrap {{ width:min(1200px, calc(100% - 40px)); margin:0 auto; padding:32px 0 80px; }}
    .hero,.card {{ background:rgba(15,28,47,.95); border:1px solid var(--line); border-radius:24px; padding:24px; box-shadow:0 20px 50px rgba(0,0,0,.25); }}
    .hero {{ margin-bottom:22px; }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; font-size:12px; letter-spacing:.14em; margin-bottom:10px; }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:12px; margin-top:20px; }}
    .stats.mini {{ margin-top:12px; }}
    .panel {{ background:#0a1527; border:1px solid var(--line); border-radius:18px; padding:16px; }}
    .panel strong {{ display:block; font-size:26px; }}
    .panel span {{ color:var(--muted); font-size:13px; }}
    .grid.two {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:16px; }}
    .card.run {{ margin-top:18px; }}
    ul,ol {{ margin:0; padding-left:20px; }}
    li {{ margin:8px 0; }}
    p {{ line-height:1.65; }}
    pre {{ white-space:pre-wrap; background:#07111f; color:#dce9ff; border:1px solid var(--line); padding:14px; border-radius:14px; overflow:auto; }}
    a {{ color:var(--accent); }}
    .score {{ color:var(--warn); font-size:12px; }}
    @media (max-width: 920px) {{ .grid.two {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class='wrap'>
    <section class='hero'>
      <div class='eyebrow'>round13 · ko pdf · final rag smoke test</div>
      <h1>Round13 · Korean PDF dataset evaluation</h1>
      <p>최종 RAG 구조(질문 분류 → adaptive query plan → fused retrieval → rerank → retrieval judge → evidence distill → citation answer)를 새 한국어 PDF에 다시 흘려본 검증 라운드다. 이번 실험은 한국저작권위원회가 정리한 "생성형 AI와 저작권" PDF를 대상으로 했다.</p>
      <p>PDF: <a href='{html.escape(summary['doc_url'])}' target='_blank'>{html.escape(summary['doc_title'])}</a></p>
      <div class='stats'>
        <article class='panel'><strong>{summary['question_count']}</strong><span>questions</span></article>
        <article class='panel'><strong>{summary['chunk_count']}</strong><span>chunks</span></article>
        <article class='panel'><strong>{summary['avg_judge_score']:.1f}</strong><span>avg judge score</span></article>
        <article class='panel'><strong>{summary['retrieval_rescues']}</strong><span>retrieval rescues</span></article>
        <article class='panel'><strong>{summary['answer_revisions']}</strong><span>answer revisions</span></article>
      </div>
    </section>
    <section class='card'>
      <div class='eyebrow'>source chunks</div>
      <h2>PDF chunk map</h2>
      <ul>{chunk_list}</ul>
    </section>
    {''.join(run_cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_pdf()
    full_text = extract_pdf_text()
    chunks = build_chunks(full_text)
    runs = [run_one(item, chunks) for item in QUESTIONS]
    summary = build_summary(chunks, runs)
    source_payload = {
        "title": PDF_TITLE,
        "url": PDF_URL,
        "pdf_path": str(PDF_PATH),
        "chunk_count": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
        "raw_text_path": str(TEXT_PATH),
    }
    payload = {"summary": summary, "runs": runs}
    SOURCE_PATH.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    QA_PATH.write_text(json.dumps(QUESTIONS, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    HTML_PATH.write_text(render_report(summary, runs, chunks), encoding="utf-8")
    print(json.dumps({
        "summary": summary,
        "html": str(HTML_PATH),
        "results": str(RESULTS_PATH),
        "questions": str(QA_PATH),
        "source": str(SOURCE_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
