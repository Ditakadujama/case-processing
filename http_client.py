"""Shared HTTP client with connection pooling, retry, and timeout management.

Replaces urllib.request usage for LLM and embedding services.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})


class HTTPClient:
    """Reusable HTTP client with connection pooling and retry logic.

    Features:
    - Connection keep-alive via httpx.Client
    - Separate connect/read/write/pool timeouts
    - Exponential backoff with jitter on retryable errors
    - Respects Retry-After header
    - Does NOT retry on 4xx (except 429)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        max_connections: int = 10,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries

        limits = httpx.Limits(
            max_keepalive_connections=max_connections,
            max_connections=max_connections + 5,
        )
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            timeout=httpx.Timeout(connect_timeout, read=read_timeout),
            limits=limits,
            headers=headers,
        )

    def post_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON payload, return parsed response dict with retry."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error = None

        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(url, json=payload)

                if resp.status_code in _RETRYABLE_STATUSES:
                    last_error = RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                    if attempt == self.max_retries - 1:
                        break
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = 2 ** attempt
                    else:
                        delay = (2 ** attempt) + (time.monotonic() % 1.0)
                    logger.debug("HTTP %d, retrying in %.1fs (attempt %d/%d)",
                                 resp.status_code, delay, attempt + 1, self.max_retries)
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500 and e.response.status_code not in _RETRYABLE_STATUSES:
                    raise  # 4xx (non-429) is not retryable
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) + (time.monotonic() % 1.0)
                    time.sleep(min(delay, 30))

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) + (time.monotonic() % 1.0)
                    time.sleep(min(delay, 30))

        raise RuntimeError(
            f"HTTP request to {url} failed after {self.max_retries} retries: {last_error}"
        )

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
