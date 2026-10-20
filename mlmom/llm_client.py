"""OpenAI-compatible chat client configured for RouterAI.

Secrets are loaded from environment variables and are never stored in model
artifacts.  The tiny protocol keeps the task graph testable without network
calls or paid tokens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


DEFAULT_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-1017"


class ChatClient(Protocol):
    """Minimal interface consumed by :class:`TaskAnalyzer`."""

    model: str

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return assistant text for a chat conversation."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 120.0
    max_tokens: int = 4_000

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "LLMSettings":
        """Load RouterAI settings from ``.env`` and the process environment."""

        load_dotenv(dotenv_path=env_file, override=False)
        api_key = os.getenv("ROUTERAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "ROUTERAI_API_KEY is not set. Copy .env.example to .env and add the key locally."
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("ROUTERAI_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
            model=os.getenv("ROUTERAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            timeout_seconds=float(os.getenv("ROUTERAI_TIMEOUT_SECONDS", "120")),
            max_tokens=int(os.getenv("ROUTERAI_MAX_TOKENS", "4000")),
        )


class RouterAIChatClient:
    """Chat Completions adapter for RouterAI's OpenAI-compatible endpoint."""

    def __init__(self, settings: LLMSettings) -> None:
        self.model = settings.model
        self._max_tokens = settings.max_tokens
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _request(self, messages: list[dict[str, str]], *, json_mode: bool) -> str:
        parameters: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._max_tokens,
        }
        if json_mode:
            parameters["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**parameters)  # type: ignore[arg-type]
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("RouterAI returned an empty completion")
        return content

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Prefer JSON mode and fall back if the selected model rejects it."""

        try:
            return self._request(messages, json_mode=True)
        except BadRequestError:
            return self._request(messages, json_mode=False)
