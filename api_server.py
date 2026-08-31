"""相似病例检索 HTTP 服务。

服务直接接收问答后端提供的结构化日级病历，不经 Excel 或临时表。
启动时只初始化一次检索系统及内存索引，后续请求复用该实例。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import DBConfig
from day_retrieval_system import DayLevelRetrievalSystem

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
MAX_TOP_K = 10
CONTROL_POOL_SIZE = 30
DEFAULT_MAX_EVIDENCE_CHARS = 1500


class DailyRecord(BaseModel):
    """与 ``medical_records`` 日级行兼容的查询记录。"""

    model_config = ConfigDict(extra="allow")

    patient_id: str
    visit_date: date
    patient_info: Optional[str] = None
    chief_complaint: Optional[str] = None
    instrument_test: Optional[str] = None
    checkout: Optional[str] = None
    examine: Optional[str] = None
    doctor_advice: Optional[str] = None
    inspection_visit: Optional[str] = None
    history_illness: Optional[str] = None
    surgery_record: Optional[str] = None
    monitor: Optional[str] = None
    operation_record: Optional[str] = None

    @field_validator("patient_id")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class SimilarCaseSearchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    query_patient_id: str
    daily_records: list[DailyRecord] = Field(min_length=1)
    cutoff_date: Optional[date] = None
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    max_days: int = Field(default=7, ge=0, le=60)
    rerank_top_n: int = Field(default=10, ge=0, le=50)
    rerank_interval: float = Field(default=0.0, ge=0.0, le=30.0)
    candidate_pool_size: int = Field(default=30, ge=5, le=200)
    strategy: Literal["similar", "low_similarity_control"] = "similar"
    max_evidence_chars: int = Field(default=DEFAULT_MAX_EVIDENCE_CHARS, ge=300, le=5000)
    include_query_details: bool = False

    @field_validator("query_patient_id")
    @classmethod
    def patient_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query_patient_id 不能为空")
        return value


class SimilarCaseItem(BaseModel):
    case_id: str
    similarity: float
    rerank_score: Optional[float] = None
    rerank_decision: Optional[str] = None
    key_matches: list[str] = Field(default_factory=list)
    key_conflicts: list[str] = Field(default_factory=list)
    daily_match_details: list[dict[str, Any]] = Field(default_factory=list)
    evidence: str = ""


class SimilarCaseSearchResponse(BaseModel):
    request_id: str
    strategy: str
    query_days: int
    retrieval_version: str
    latency_ms: int
    cases: list[SimilarCaseItem]
    query_details: Optional[list[dict[str, Any]]] = None


_PHI_PATTERNS = (
    re.compile(r"(?i)(患者ID|patient[_ ]?id|住院号|病案号|身份证号)\s*[：:]?\s*[^\s，,；;]+"),
    re.compile(r"(?i)(姓名)\s*[：:]?\s*[\u4e00-\u9fff·]{2,8}"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
)


def _anonymous_case_id(patient_id: str) -> str:
    salt = os.getenv("CASE_ANONYMIZATION_SALT", "similar-case-service")
    digest = hashlib.sha256(f"{salt}:{patient_id}".encode("utf-8")).hexdigest()[:10]
    return f"case-{digest}"


def redact_case_evidence(text: str, patient_id: str, max_chars: int) -> str:
    """移除常见直接标识符，并限制单病例注入长度。"""
    value = (text or "").replace(patient_id, "[历史病例]")
    for pattern in _PHI_PATTERNS:
        value = pattern.sub("[已脱敏]", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if len(value) > max_chars:
        value = value[:max_chars].rstrip() + "\n[证据已截断]"
    return value


def _redact_structured_value(value: Any, patient_id: str, max_chars: int) -> Any:
    """递归脱敏病例卡，同时保留 JSON 的字典/列表结构。"""
    if isinstance(value, dict):
        return {
            str(key): _redact_structured_value(item, patient_id, max_chars)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_structured_value(item, patient_id, max_chars) for item in value
        ]
    if isinstance(value, str):
        return redact_case_evidence(value, patient_id, max_chars)
    return value


def _as_float(value: Any) -> Optional[float]:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_public_case(result: dict[str, Any], max_evidence_chars: int) -> SimilarCaseItem:
    raw_patient_id = str(result.get("id") or "")
    rerank = result.get("rerank") or {}
    return SimilarCaseItem(
        case_id=_anonymous_case_id(raw_patient_id),
        similarity=float(result.get("similarity") or 0.0),
        rerank_score=_as_float(result.get("rerank_score", rerank.get("relevance_score"))),
        rerank_decision=result.get("rerank_decision", rerank.get("decision")),
        key_matches=list(rerank.get("key_matches") or []),
        key_conflicts=list(rerank.get("key_conflicts") or []),
        daily_match_details=list(result.get("daily_match_details") or []),
        evidence=redact_case_evidence(
            str(result.get("full_text") or ""), raw_patient_id, max_evidence_chars
        ),
    )


def _filter_records(req: SimilarCaseSearchRequest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(req.daily_records, start=1):
        if record.patient_id != req.query_patient_id:
            raise ValueError("daily_records 中包含不属于 query_patient_id 的记录")
        if req.cutoff_date and record.visit_date > req.cutoff_date:
            continue
        row = record.model_dump(mode="json")
        row["id"] = index
        rows.append(row)
    if not rows:
        raise ValueError("cutoff_date 之前没有可用于检索的日级病历")
    return rows


class SimilarCaseService:
    def __init__(self, system: Optional[DayLevelRetrievalSystem] = None) -> None:
        self.system = system or DayLevelRetrievalSystem(DBConfig.from_env())
        # 部分组件内部带有可变缓存；串行化搜索可避免并发请求互相污染。
        self._search_lock = threading.Lock()

    def search(self, req: SimilarCaseSearchRequest) -> SimilarCaseSearchResponse:
        started = time.perf_counter()
        rows = _filter_records(req)
        with self._search_lock:
            query_days_by_patient = self.system.build_query_days(rows, max_days=req.max_days)
            query_days = query_days_by_patient.get(req.query_patient_id) or []
            if not query_days:
                raise ValueError("查询病例过滤后没有可用于相似度计算的医生关注文本")

            if req.strategy == "low_similarity_control":
                fetch_k = max(req.top_k, min(CONTROL_POOL_SIZE, req.candidate_pool_size))
                results = self.system.search(
                    query_days,
                    top_k=fetch_k,
                    exclude_patient_ids={req.query_patient_id},
                    max_days=req.max_days,
                    rerank_top_n=0,
                    rerank_interval=0.0,
                    candidate_pool_size=max(fetch_k, req.candidate_pool_size),
                )
                results = list(reversed(results))[:req.top_k]
            else:
                results = self.system.search(
                    query_days,
                    top_k=req.top_k,
                    exclude_patient_ids={req.query_patient_id},
                    max_days=req.max_days,
                    rerank_top_n=req.rerank_top_n,
                    rerank_interval=req.rerank_interval,
                    candidate_pool_size=req.candidate_pool_size,
                )

        query_details = None
        if req.include_query_details:
            query_details = []
            for day in query_days:
                query_details.append(
                    {
                        "day_index": day.get("day_index"),
                        "visit_date": day.get("visit_date"),
                        "day_text": redact_case_evidence(
                            str(day.get("day_text") or ""),
                            req.query_patient_id,
                            req.max_evidence_chars,
                        ),
                        "case_card": _redact_structured_value(
                            day.get("case_card") or {},
                            req.query_patient_id,
                            req.max_evidence_chars,
                        ),
                    }
                )

        return SimilarCaseSearchResponse(
            request_id=uuid.uuid4().hex,
            strategy=req.strategy,
            query_days=len(query_days),
            retrieval_version="day-v3.0-case-card-embedding",
            latency_ms=round((time.perf_counter() - started) * 1000),
            cases=[_to_public_case(item, req.max_evidence_chars) for item in results],
            query_details=query_details,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.similar_case_service = SimilarCaseService()
        app.state.startup_error = None
    except Exception as exc:  # 服务仍可启动，并通过 health 明确暴露错误。
        logger.exception("相似病例检索系统初始化失败")
        app.state.similar_case_service = None
        app.state.startup_error = str(exc)
    yield


app = FastAPI(title="Similar Case Retrieval API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "similar_case_service", None)
    error = getattr(request.app.state, "startup_error", None)
    if service is None:
        return {"status": "unavailable", "error": error}
    return {
        "status": "ok",
        "patients": service.system.patient_count,
        "days": service.system.day_count,
    }


@app.post("/api/v1/similar-cases/search", response_model=SimilarCaseSearchResponse)
async def search_similar_cases(
    payload: SimilarCaseSearchRequest, request: Request
) -> SimilarCaseSearchResponse:
    service = getattr(request.app.state, "similar_case_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(request.app.state, "startup_error", None) or "检索服务未就绪",
        )
    try:
        return await run_in_threadpool(service.search, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("相似病例检索失败")
        raise HTTPException(status_code=502, detail=f"相似病例检索失败: {exc}") from exc
