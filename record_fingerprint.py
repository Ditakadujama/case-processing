"""Stable record identity and content hashing for idempotent processing.

This module is the SINGLE source of truth for all hash computations.
No other module should define hash logic for record identity, extraction input,
or embedding input.
"""

import hashlib
from typing import Any, Dict

_STANDARD_ENCODING = "utf-8"

# Values that should be treated as empty during canonicalization
_EMPTY_VALUES = frozenset({None, "", "None", "NaN", "nan", "null", "NULL", "N/A", "NaT"})


def canonicalize_value(value: Any) -> str:
    """Normalize None, empty strings, NaN, and Unicode for deterministic hashing.

    Rules:
    - None → ""
    - "\r\n" and "\r" → "\n"
    - Leading/trailing whitespace stripped
    - Known sentinel empty values ("NaN", "null", etc.) → ""
    """
    if value is None:
        return ""
    s = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if s in _EMPTY_VALUES:
        return ""
    return s


def build_source_hash(row: Dict[str, Any]) -> str:
    """SHA-256 over the raw fields that influence build_clinical_day_text().

    Fields included (matching clinical_text_filter.py's text building logic):
      patient_id, visit_date, patient_info, chief_complaint, checkout, examine,
      doctor_advice, inspection_visit, history_illness, surgery_record, monitor,
      operation_record.

    The canonicalization order is lexicographic by field name for determinism.
    """
    field_names = sorted([
        "patient_id", "visit_date",
        "patient_info", "chief_complaint", "checkout", "examine",
        "doctor_advice", "inspection_visit", "history_illness",
        "surgery_record", "monitor", "operation_record",
    ])
    parts = []
    for name in field_names:
        parts.append(f"{name}:{canonicalize_value(row.get(name))}")
    digest = hashlib.sha256("\n".join(parts).encode(_STANDARD_ENCODING)).hexdigest()
    return digest


def build_extraction_input_hash(day_text: str, history_context: str) -> str:
    """SHA-256 over the exact inputs fed to LLM extract_day().

    Args:
        day_text: The filtered day-level clinical text (current day).
        history_context: The history text passed as cumulative context.
    """
    raw = (
        f"day_text:{canonicalize_value(day_text)}\n"
        f"history:{canonicalize_value(history_context)}"
    )
    return hashlib.sha256(raw.encode(_STANDARD_ENCODING)).hexdigest()


def build_embedding_input_hash(embedding_text: str) -> str:
    """SHA-256 over the output of build_case_card_embedding_text()."""
    return hashlib.sha256(
        canonicalize_value(embedding_text).encode(_STANDARD_ENCODING)
    ).hexdigest()
