"""
Cohere API rate limiter: stay under 100 calls/min by waiting at 90 in a 60s window.
Shared by Cohere embeddings and Cohere reranker.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60
# For text embeddings, production keys allow ~2,000 inputs/min.
# We mostly care about throughput, so this default is high; tune down via
# COHERE_RATE_LIMIT_MAX_CALLS if you ever see 429s.
MAX_CALLS_IN_WINDOW = 120


class CohereRateLimiter:
    """
    Tracks Cohere API calls in a 60-second sliding window.
    Blocks (acquire) when we would exceed 90 calls in that window so we stay under 100/min.
    """

    def __init__(self, window_seconds: int = WINDOW_SECONDS, max_calls: int = MAX_CALLS_IN_WINDOW) -> None:
        self._window = window_seconds
        self._max_calls = max_calls
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until we can make one more call without exceeding the limit."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            while len(self._timestamps) >= self._max_calls:
                sleep_time = self._timestamps[0] + self._window - now
                if sleep_time > 0:
                    logger.debug("Cohere rate limit: waiting %.1fs (90 calls in 60s window)", sleep_time)
                    time.sleep(sleep_time)
                    now = time.monotonic()
                    cutoff = now - self._window
                    while self._timestamps and self._timestamps[0] < cutoff:
                        self._timestamps.popleft()
                else:
                    break

    def record(self) -> None:
        """Record that one Cohere API call was just made."""
        with self._lock:
            self._timestamps.append(time.monotonic())


# Shared singleton for Cohere embeddings and reranker
_rate_limiter: CohereRateLimiter | None = None


def get_cohere_rate_limiter() -> CohereRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        import os
        max_calls = int(os.environ.get("COHERE_RATE_LIMIT_MAX_CALLS", str(MAX_CALLS_IN_WINDOW)))
        _rate_limiter = CohereRateLimiter(max_calls=max_calls)
    return _rate_limiter


# Batch size for embed_documents. Cohere allows large batches; larger = fewer calls.
# With ~3k chunks this finishes in a handful of requests.
COHERE_EMBED_BATCH_SIZE = 256
# How many embed batches to run in parallel (stays under rate limit via shared limiter).
COHERE_EMBED_MAX_WORKERS = 16


class RateLimitedCohereEmbeddings:
    """
    Wraps a Cohere embeddings client and enforces the shared rate limit.
    Splits embed_documents into batches; runs up to COHERE_EMBED_MAX_WORKERS batches in parallel.
    """

    def __init__(self, inner: Any, limiter: CohereRateLimiter | None = None) -> None:
        self._inner = inner
        self._limiter = limiter or get_cohere_rate_limiter()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batches = [
            texts[i : i + COHERE_EMBED_BATCH_SIZE]
            for i in range(0, len(texts), COHERE_EMBED_BATCH_SIZE)
        ]
        # Preserve order: index -> vectors
        results: List[Optional[List[List[float]]]] = [None] * len(batches)

        def do_batch(idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            self._limiter.acquire()
            try:
                return idx, self._inner.embed_documents(batch)
            finally:
                self._limiter.record()

        with ThreadPoolExecutor(max_workers=COHERE_EMBED_MAX_WORKERS) as pool:
            futures = {pool.submit(do_batch, i, b): i for i, b in enumerate(batches)}
            for fut in as_completed(futures):
                idx, vecs = fut.result()
                results[idx] = vecs

        out: List[List[float]] = []
        for r in results:
            assert r is not None
            out.extend(r)
        return out

    def embed_query(self, text: str) -> list[float]:
        self._limiter.acquire()
        try:
            out = self._inner.embed_query(text)
            return out
        finally:
            self._limiter.record()
