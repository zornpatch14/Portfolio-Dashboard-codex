"""Lightweight cache abstraction with optional Redis backend."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[bytes]:
        ...

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


@dataclass
class InMemoryCache(CacheBackend):
    """Simple in-process cache fallback."""

    store: dict[str, bytes]

    def get(self, key: str) -> Optional[bytes]:
        return self.store.get(key)

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> None:  # noqa: ARG002
        self.store[key] = value

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


class RedisCache(CacheBackend):
    """Redis-backed cache; gracefully degrades if Redis is unavailable."""

    def __init__(self, url: str):
        try:
            import redis  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            raise RuntimeError("redis package not installed")

        self.client = redis.Redis.from_url(url, decode_responses=False)

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self.client.get(key)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Redis get failed for key %s", key, exc_info=True)
            return None

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> None:
        try:
            if ttl_seconds:
                self.client.setex(key, ttl_seconds, value)
            else:
                self.client.set(key, value)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Redis set failed for key %s", key, exc_info=True)

    def delete(self, key: str) -> None:
        try:
            self.client.delete(key)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Redis delete failed for key %s", key, exc_info=True)


def build_selection_key(selection_hash: str, data_version: Optional[str], artifact: str) -> str:
    """Construct a namespaced key for selection-level artifacts."""

    dv = data_version or "default"
    return f"api:v1:sel:{selection_hash}:dv:{dv}:{artifact}"


def get_cache_backend() -> CacheBackend:
    """Return the configured cache backend (Redis when REDIS_URL set, else in-memory)."""

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            return RedisCache(redis_url)
        except Exception:
            logger.warning("Falling back to in-memory cache; Redis initialization failed", exc_info=True)
    return InMemoryCache({})


__all__ = [
    "CacheBackend",
    "InMemoryCache",
    "RedisCache",
    "build_selection_key",
    "get_cache_backend",
]
