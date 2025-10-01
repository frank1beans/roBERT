"""Adapters for question-answering large language models."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Protocol

import aiohttp
from pydantic import BaseModel, Field

from .prompts import load_prompt_library

__all__ = ["QALLM", "QALLMConfig", "HttpLLM", "MockLLM", "AsyncHttpLLM", "build_prompt"]


LOGGER = logging.getLogger(__name__)


class QALLMConfig(BaseModel):
    """Configuration for QA-oriented LLM clients."""

    endpoint: Optional[str] = Field(default=None, description="HTTP endpoint accepting JSON payloads")
    model: Optional[str] = Field(default=None, description="Remote model identifier")
    timeout: float = Field(default=30.0, ge=1.0, description="Timeout for each request in seconds")
    max_retries: int = Field(default=2, ge=0, description="Number of retry attempts on failure")


class QALLM(Protocol):
    """Protocol implemented by QA-capable LLM adapters."""

    def ask(self, text: str, question: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Answer ``question`` grounding only on ``text``.

        Implementations must return a JSON-serialisable dictionary containing at
        least the ``value`` key. Additional metadata such as ``confidence`` may
        be provided.
        """


def build_prompt(text: str, question: str, schema: Dict[str, Any]) -> str:
    """Construct a deterministic prompt instructing the model to output JSON."""

    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    library = load_prompt_library()
    return library.render(
        "property_question",
        text=text.strip() or "<vuoto>",
        question=question,
        schema=schema_json,
    )


class HttpLLM(QALLM):
    """HTTP client calling an external LLM endpoint."""

    def __init__(self, config: QALLMConfig):
        if not config.endpoint:
            raise ValueError("HttpLLM requires a non-empty endpoint")
        self._config = config

    def ask(self, text: str, question: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self._config.model,
            "prompt": build_prompt(text, question, json_schema),
            "schema": json_schema,
        }
        attempts = self._config.max_retries + 1
        for attempt in range(attempts):
            try:
                body = json.dumps(payload).encode("utf-8")
                request = urllib.request.Request(
                    self._config.endpoint,
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=self._config.timeout) as response:
                    charset = response.headers.get_content_charset("utf-8")
                    data = response.read().decode(charset)
                parsed = json.loads(data)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response must be a JSON object")
                return parsed
            except (urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
                LOGGER.warning(
                    "llm_call_failed",
                    extra={"attempt": attempt + 1, "max_attempts": attempts, "error": str(exc)},
                )
                if attempt + 1 >= attempts:
                    raise
                delay = 2 ** attempt
                time.sleep(delay)
        raise RuntimeError("LLM call failed after retries")


class MockLLM(QALLM):
    """Fallback implementation used when no endpoint is configured."""

    def ask(self, text: str, question: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": None, "confidence": 0.0}


class AsyncHttpLLM:
    """Async HTTP client for parallel LLM requests."""

    def __init__(self, config: QALLMConfig):
        if not config.endpoint:
            raise ValueError("AsyncHttpLLM requires a non-empty endpoint")
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self._config.timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def ask(self, text: str, question: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self._session:
            raise RuntimeError("AsyncHttpLLM must be used as async context manager")

        payload = {
            "model": self._config.model,
            "prompt": build_prompt(text, question, json_schema),
            "schema": json_schema,
        }
        attempts = self._config.max_retries + 1

        for attempt in range(attempts):
            try:
                async with self._session.post(
                    self._config.endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if not isinstance(data, dict):
                        raise ValueError("LLM response must be a JSON object")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                LOGGER.warning(
                    "async_llm_call_failed",
                    extra={"attempt": attempt + 1, "max_attempts": attempts, "error": str(exc)},
                )
                if attempt + 1 >= attempts:
                    raise
                delay = 2 ** attempt
                await asyncio.sleep(delay)

        raise RuntimeError("Async LLM call failed after retries")
