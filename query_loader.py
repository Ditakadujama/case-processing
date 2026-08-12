"""Query data loading pipeline: Excel → DataFrame → List[dict] → build_query_days().

Replaces the query_records table round-trip for the default search path.
Each search generates a unique request_id for output isolation.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List


def generate_request_id() -> str:
    """Generate a unique request ID for this search invocation."""
    return uuid.uuid4().hex


def load_xlsx_daily_rows(filepath: str) -> List[Dict[str, Any]]:
    """Load query rows from Excel directly, bypassing query_records table.

    Returns rows as List[dict] in the same shape that load_query_daily_rows()
    returns from the DB query.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("请先安装 pandas: pip install pandas")

    df = pd.read_excel(filepath, sheet_name="Sheet1")
    if "date" in df.columns and "visit_date" not in df.columns:
        df = df.rename(columns={"date": "visit_date"})

    required = {"patient_id", "visit_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel 缺少必要列: {', '.join(sorted(missing))}")

    # Fill NaN visit_date consistently with existing behavior
    df["visit_date"] = df["visit_date"].fillna(pd.Timestamp("1900-01-01"))

    # Convert to List[dict] matching output of load_query_daily_rows()
    rows = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        record = {"id": row_number}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                record[col] = None
            elif col == "visit_date":
                record[col] = str(pd.Timestamp(val).date()) if hasattr(val, "date") else str(val)
            else:
                record[col] = val
        rows.append(record)
    return rows
