"""Token bucket primitives for shared throttling."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Protocol


class TokenBucket(Protocol):
    """Simple async token bucket contract."""

    @asynccontextmanager
    async def reserve(self, cost: int = 1):
        ...


class NullTokenBucket(TokenBucket):
    """No-op bucket used when throttling is disabled."""

    @asynccontextmanager
    async def reserve(self, cost: int = 1):  # noqa: D401 - trivial default
        yield


class FixedTokenBucket(TokenBucket):
    """Concurrency-based token bucket."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._semaphore = asyncio.Semaphore(capacity)

    @asynccontextmanager
    async def reserve(self, cost: int = 1):
        if cost != 1:
            raise NotImplementedError("fixed bucket currently supports cost=1")
        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()
