"""Second-stage rerankers for medical case retrieval."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import List, Optional

from config import LLMConfig, RerankerConfig
from llm_case_extractor import LLMCaseExtractor
from llm_case_extractor import _direct_urlopen

logger = logging.getLogger(__name__)


RERANKER_SYSTEM_PROMPT = (
    "你是重症医学相似病例检索的二阶段精排器。"
    "你只根据给定的查询病例和候选病例判断是否适合推荐给医生参考。"
    "最重要的是最终诊断/病因链是否一致，其次才是病程发展、器官功能、关键治疗。"
    "不要因为同样出现休克、插管、ICU、呼吸衰竭等泛化危重表现就给高分。"
    "如果病因链明显不同，应判为 reject 或 borderline。"
    "只输出 JSON，不要输出解释性正文。"
)


def _clip(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n...[中间省略]...\n\n{tail}"


def _compact_days(days: List[dict], max_chars: int = 12000) -> str:
    parts = []
    for day in days:
        parts.append(
            f"### 查询 Day {day.get('day_index')} - {day.get('visit_date', '')}\n"
            f"{day.get('day_text', '')}"
        )
    return _clip("\n\n".join(parts), max_chars)


def _candidate_text(candidate: dict, max_chars: int = 12000) -> str:
    return _clip(
        "\n".join([
            f"候选患者ID：{candidate.get('id')}",
            f"初筛综合相似度：{candidate.get('similarity')}",
            f"天级对齐明细：{candidate.get('daily_match_details', [])}",
            candidate.get("full_text", ""),
        ]),
        max_chars,
    )


def _build_rerank_prompt(query_days: List[dict], candidate: dict) -> str:
    return f"""请判断候选病例是否适合作为查询病例的相似病例推荐给医生。

评分原则：
1. final_diagnoses / primary_diagnosis_axis / etiology_axis / etiology_chain 一致性最重要。
2. 感染病例必须区分感染来源和诱因：例如泌尿系感染+结石/梗阻，不应仅因同为感染性休克而高分。
3. 心血管病例必须区分冠心病/急性心梗、肺栓塞、主动脉夹层、心律失常等病因。
4. 如果病因链不同，即使休克、插管、CRRT 等危重表现相似，也不要给高分。
5. 如果信息不足但没有明显冲突，可以给 borderline；宁缺毋滥，不要为了凑 Top5 强行推荐。

请输出 JSON：
{{
  "relevance_score": 0,
  "decision": "excellent|good|borderline|reject",
  "same_final_diagnosis_or_etiology": true,
  "same_disease_stage": true,
  "key_matches": [],
  "key_conflicts": [],
  "reason": ""
}}

【查询病例】
{_compact_days(query_days)}

【候选病例】
候选患者ID：{candidate.get('id')}
初筛综合相似度：{candidate.get('similarity')}
天级对齐明细：{candidate.get('daily_match_details', [])}

{_clip(candidate.get('full_text', ''), 12000)}
"""


class HTTPReranker:
    """Use a deployed cross-encoder reranker service, such as Qwen3-Reranker."""

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()

    @property
    def is_available(self) -> bool:
        return self.config.is_configured

    def rerank_candidates(
        self,
        query_days: List[dict],
        candidates: List[dict],
        top_n: int,
        interval: float = 0.0,
    ) -> List[dict]:
        if top_n <= 0 or not candidates or not self.is_available:
            return candidates

        rerank_count = min(top_n, len(candidates))
        target = candidates[:rerank_count]
        query = _compact_days(query_days, max_chars=self.config.max_query_chars)
        documents = [
            _candidate_text(candidate, max_chars=self.config.max_doc_chars)
            for candidate in target
        ]

        try:
            scores = self._score_documents(query, documents)
        except Exception as exc:
            logger.warning("HTTP reranker failed, keep original order: %s", exc)
            return candidates

        reranked = []
        for candidate, score in zip(target, scores):
            item = dict(candidate)
            normalized = _normalize_score_to_100(score)
            item["rerank"] = {
                "relevance_score": normalized,
                "raw_score": score,
                "decision": _decision_from_score(normalized),
                "source": "http_reranker",
                "same_final_diagnosis_or_etiology": normalized >= 70,
                "same_disease_stage": normalized >= 60,
                "key_matches": [],
                "key_conflicts": [],
                "reason": "HTTP reranker score from deployed reranker service",
            }
            item["rerank_score"] = normalized
            item["rerank_raw_score"] = score
            item["rerank_decision"] = item["rerank"]["decision"]
            reranked.append(item)

        tail = [dict(item) for item in candidates[rerank_count:]]
        reranked.sort(
            key=lambda item: (
                item.get("rerank_score", 0),
                item.get("similarity", 0),
            ),
            reverse=True,
        )
        if interval > 0:
            time.sleep(interval)
        return reranked + tail

    def _score_documents(self, query: str, documents: List[str]) -> List[float]:
        payloads = [
            {
                "model": self.config.model,
                "query": query,
                "documents": documents,
                "instruction": self.config.instruction,
            },
            {
                "model": self.config.model,
                "query": query,
                "texts": documents,
                "instruction": self.config.instruction,
            },
        ]
        last_error = None
        for payload in payloads:
            for attempt in range(self.config.max_retries):
                try:
                    data = self._post_json(payload)
                    scores = _extract_scores(data, expected_count=len(documents))
                    if len(scores) == len(documents):
                        return scores
                    last_error = RuntimeError(f"reranker 返回分数数量不匹配: {data}")
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    # Retry with the alternate payload when the server rejects documents/texts shape.
                    if exc.code in {400, 404, 422}:
                        break
                    time.sleep(min(2 ** attempt, 8))
                except Exception as exc:
                    last_error = exc
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"reranker 请求失败: {last_error}")

    def _post_json(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.config.url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.config.api_key:
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
        with _direct_urlopen(req, timeout=self.config.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class LLMReranker:
    """Use the configured LLM as a cross-case reranker."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.extractor = LLMCaseExtractor(self.config)

    @property
    def is_available(self) -> bool:
        return self.config.is_configured

    def rerank_one(self, query_days: List[dict], candidate: dict) -> Optional[dict]:
        if not self.is_available:
            return None
        record_id = f"rerank_{candidate.get('id', 'unknown')}"
        prompt = _build_rerank_prompt(query_days, candidate)
        try:
            raw = self.extractor._call_llm(prompt, system_prompt=RERANKER_SYSTEM_PROMPT)
            parsed = self.extractor._parse_json_response(raw, record_id, log_failure=False)
        except Exception as exc:
            logger.warning("LLM rerank failed for %s: %s", candidate.get("id"), exc)
            return None
        if not isinstance(parsed, dict):
            return None
        return _normalize_rerank_result(parsed)

    def rerank_candidates(
        self,
        query_days: List[dict],
        candidates: List[dict],
        top_n: int,
        interval: float = 0.0,
    ) -> List[dict]:
        if top_n <= 0 or not candidates or not self.is_available:
            return candidates

        rerank_count = min(top_n, len(candidates))
        reranked = []
        for index, candidate in enumerate(candidates[:rerank_count]):
            result = self.rerank_one(query_days, candidate)
            item = dict(candidate)
            if result:
                item["rerank"] = result
                item["rerank_score"] = result["relevance_score"]
                item["rerank_decision"] = result["decision"]
            else:
                item["rerank"] = {
                    "relevance_score": 0,
                    "decision": "unknown",
                    "source": "llm_reranker",
                    "same_final_diagnosis_or_etiology": False,
                    "same_disease_stage": False,
                    "key_matches": [],
                    "key_conflicts": ["LLM rerank failed"],
                    "reason": "LLM rerank failed",
                }
                item["rerank_score"] = 0
                item["rerank_decision"] = "unknown"
            reranked.append(item)
            if interval > 0 and index < rerank_count - 1:
                time.sleep(interval)

        # Keep rejected candidates below non-rejected ones, but still visible if not enough candidates remain.
        tail = [dict(item) for item in candidates[rerank_count:]]
        reranked.sort(
            key=lambda item: (
                0 if item.get("rerank_decision") == "reject" else 1,
                item.get("rerank_score", 0),
                item.get("similarity", 0),
            ),
            reverse=True,
        )
        return reranked + tail


def _normalize_rerank_result(value: dict) -> dict:
    try:
        score = int(float(value.get("relevance_score", 0)))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    decision = str(value.get("decision") or "").strip().lower()
    if decision not in {"excellent", "good", "borderline", "reject"}:
        if score >= 85:
            decision = "excellent"
        elif score >= 70:
            decision = "good"
        elif score >= 50:
            decision = "borderline"
        else:
            decision = "reject"

    return {
        "relevance_score": score,
        "decision": decision,
        "source": "llm_reranker",
        "same_final_diagnosis_or_etiology": bool(value.get("same_final_diagnosis_or_etiology")),
        "same_disease_stage": bool(value.get("same_disease_stage")),
        "key_matches": _string_list(value.get("key_matches")),
        "key_conflicts": _string_list(value.get("key_conflicts")),
        "reason": str(value.get("reason") or "").strip(),
    }


def _normalize_score_to_100(score: float) -> int:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0
    if 0.0 <= value <= 1.0:
        value *= 100.0
    return max(0, min(100, int(round(value))))


def _decision_from_score(score: int) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "borderline"
    return "reject"


def _extract_scores(data, expected_count: int) -> List[float]:
    if isinstance(data, list):
        return _scores_from_list(data, expected_count)
    if not isinstance(data, dict):
        return []
    for key in ("scores", "relevance_scores"):
        scores = data.get(key)
        if isinstance(scores, list):
            return [float(score) for score in scores[:expected_count]]
    for key in ("results", "data"):
        values = data.get(key)
        if isinstance(values, list):
            return _scores_from_list(values, expected_count)
    return []


def _scores_from_list(values: list, expected_count: int) -> List[float]:
    if not values:
        return []
    if all(isinstance(item, (int, float)) for item in values):
        return [float(item) for item in values[:expected_count]]
    indexed_scores = []
    positional_scores = []
    for pos, item in enumerate(values):
        if not isinstance(item, dict):
            continue
        score = (
            item.get("relevance_score")
            if "relevance_score" in item
            else item.get("score", item.get("similarity"))
        )
        if score is None:
            continue
        index = item.get("index", item.get("document_index", pos))
        try:
            indexed_scores.append((int(index), float(score)))
        except (TypeError, ValueError):
            positional_scores.append(float(score))
    if indexed_scores:
        scores_by_index = {index: score for index, score in indexed_scores}
        return [scores_by_index.get(index, 0.0) for index in range(expected_count)]
    return positional_scores[:expected_count]


def _string_list(value) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]
