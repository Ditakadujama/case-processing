"""Candidate retriever abstraction for separable top-level patient filtering.

Provides two implementations:
- LegacyFullScanCandidateRetriever: exhaustive O(N) scan (current behavior)
- InMemoryVectorCandidateRetriever: numpy matrix prefilter (no external vector DB)
"""

from __future__ import annotations

import logging
import heapq
from collections import OrderedDict
from typing import Dict, List, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class CandidateRetriever(Protocol):
    """Interface for patient-level candidate retrieval."""

    def retrieve_patient_ids(
        self,
        query_days: List[dict],
        patient_limit: int = 200,
        day_limit_per_query: int = 300,
        filters: Optional[dict] = None,
    ) -> List[str]:
        ...

    @property
    def backend_name(self) -> str:
        ...

    def load_index(self) -> None:
        ...


class LegacyFullScanCandidateRetriever:
    """Exhaustive scan — returns all patient IDs.

    This is the fallback backend. No filtering; the caller applies
    patient_limit after full scoring.
    """

    def __init__(self, day_store) -> None:
        self._day_store = day_store
        self._all_patient_ids: List[str] = []

    @property
    def backend_name(self) -> str:
        return "legacy_full_scan"

    def load_index(self) -> None:
        self._all_patient_ids = self._day_store.list_all_patient_ids()
        logger.info("LegacyFullScan loaded %d patient IDs", len(self._all_patient_ids))

    def retrieve_patient_ids(
        self,
        query_days: List[dict],
        patient_limit: int = 200,
        day_limit_per_query: int = 300,
        filters: Optional[dict] = None,
    ) -> List[str]:
        return list(self._all_patient_ids)


class InMemoryVectorCandidateRetriever:
    """Vector similarity prefilter using in-memory numpy matrix.

    Builds a normalized float32 [N, D] matrix from all indexed
    case_card_embedding blobs. Query uses a single matrix multiplication.
    No external vector DB required.
    """

    def __init__(self, day_store, extractor_version: str, embedding_model: str) -> None:
        self._day_store = day_store
        self._extractor_version = extractor_version
        self._embedding_model = embedding_model
        self._matrix: Optional[np.ndarray] = None  # float32 [N, D]
        self._day_record_ids: List[str] = []
        self._patient_ids: List[str] = []
        self._day_indexes: List[int] = []
        self._dimension: int = 0

    @property
    def backend_name(self) -> str:
        return "in_memory_vector"

    @property
    def record_count(self) -> int:
        return self._matrix.shape[0] if self._matrix is not None else 0

    def load_index(self) -> None:
        """Load normalized embedding matrix and metadata from MySQL.

        Only loads case_card_embedding. Excludes records with missing
        embedding, dimension mismatch, or zero vectors.
        """
        self._matrix = None
        self._day_record_ids = []
        self._patient_ids = []
        self._day_indexes = []
        self._dimension = 0

        conn = self._day_store._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT c.day_record_id, c.patient_id, c.day_index,
                           c.case_card_embedding, c.embedding_dimension
                    FROM record_day_case_cards c
                    JOIN record_day_processing_jobs j
                      ON j.day_record_id = c.day_record_id
                    JOIN record_days d
                      ON d.day_record_id = c.day_record_id
                    WHERE c.case_card_embedding IS NOT NULL
                      AND c.extractor_version = %s
                      AND c.embedding_model = %s
                      AND j.processing_status = 'indexed'
                      AND j.source_hash = c.source_hash
                      AND j.extraction_input_hash = c.extraction_input_hash
                      AND d.source_hash = c.source_hash
                      AND d.extraction_input_hash = c.extraction_input_hash
                    ORDER BY c.patient_id, c.day_index
                """, (self._extractor_version, self._embedding_model))
                rows = cursor.fetchall()
        finally:
            conn.close()

        vectors = []
        excluded_count = 0

        for row in rows:
            blob = row["case_card_embedding"]
            dim = row.get("embedding_dimension") or 0

            if self._dimension == 0 and dim > 0:
                self._dimension = dim
            elif dim > 0 and dim != self._dimension:
                excluded_count += 1
                continue

            vec = np.frombuffer(blob, dtype=np.float32)

            if self._dimension > 0 and len(vec) != self._dimension:
                excluded_count += 1
                continue
            if self._dimension == 0 and len(vec) > 0:
                self._dimension = len(vec)

            # Normalize at load time (one-time cost)
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm == 0:
                excluded_count += 1
                continue
            vec = vec / norm

            vectors.append(vec)
            self._day_record_ids.append(row["day_record_id"])
            self._patient_ids.append(row["patient_id"])
            self._day_indexes.append(row["day_index"])

        if vectors:
            self._matrix = np.stack(vectors, axis=0).astype(np.float32)
            logger.info("InMemoryVector loaded %d vectors, dim=%d (excluded=%d)",
                        self._matrix.shape[0], self._dimension, excluded_count)
        else:
            logger.warning("InMemoryVector index is empty")

    def retrieve_patient_ids(
        self,
        query_days: List[dict],
        patient_limit: int = 200,
        day_limit_per_query: int = 300,
        filters: Optional[dict] = None,
    ) -> List[str]:
        """Pre-filter using dot-product matrix multiplication.

        For each query day, computes cosine similarity against all indexed
        records, then aggregates by patient using max score, top-N avg,
        and day coverage.
        """
        if self._matrix is None or self._matrix.shape[0] == 0:
            raise RuntimeError("Vector index is empty")

        # Collect and normalize query vectors
        query_vectors = []
        for day in query_days:
            emb = day.get("case_card_embedding")
            if emb is not None:
                if isinstance(emb, np.ndarray):
                    norm = float(np.linalg.norm(emb))
                    if norm > 0:
                        query_vectors.append(emb.astype(np.float32) / norm)

        if not query_vectors:
            raise RuntimeError("Query has no valid embeddings, cannot use vector retriever")

        query_matrix = np.stack(query_vectors, axis=0)  # [Q, D]
        if query_matrix.shape[1] != self._matrix.shape[1]:
            raise RuntimeError(
                f"Query embedding dimension {query_matrix.shape[1]} does not match "
                f"index dimension {self._matrix.shape[1]}"
            )

        # Single matrix multiplication: [Q, D] @ [D, N]^T = [Q, N]
        scores = query_matrix @ self._matrix.T  # [Q, N]

        # Only the top records for each query day count as hits. This prevents
        # long admissions from receiving artificial coverage merely because
        # they contain more indexed days.
        hit_scores: Dict[str, Dict[int, float]] = {}
        for query_index in range(scores.shape[0]):
            row_scores = scores[query_index]
            best_by_patient: Dict[str, float] = {}
            for record_index, patient_id in enumerate(self._patient_ids):
                score = float(row_scores[record_index])
                previous = best_by_patient.get(patient_id)
                if previous is None or score > previous:
                    best_by_patient[patient_id] = score

            hit_limit = min(
                max(int(day_limit_per_query), 1),
                len(best_by_patient),
            )
            top_patients = heapq.nlargest(
                hit_limit,
                best_by_patient.items(),
                key=lambda item: item[1],
            )
            for patient_id, score in top_patients:
                per_query = hit_scores.setdefault(patient_id, {})
                per_query[query_index] = score

        # Compute recall score from per-query-day best hits.
        query_day_count = len(query_vectors)
        patient_recall: List[tuple[str, float]] = []
        for patient_id, per_query_scores in hit_scores.items():
            score_list = list(per_query_scores.values())
            max_score = max(score_list)
            top_k_avg = float(np.mean(sorted(score_list, reverse=True)[:3]))
            coverage = len(per_query_scores) / max(query_day_count, 1)
            recall_score = 0.5 * max_score + 0.35 * top_k_avg + 0.15 * min(coverage, 1.0)
            patient_recall.append((patient_id, recall_score))

        patient_recall.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in patient_recall[:patient_limit]]
