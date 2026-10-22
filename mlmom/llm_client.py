"""OpenAI-compatible chat client configured through environment variables.

Secrets are loaded from environment variables and are never stored in model
artifacts.  The tiny protocol keeps the task graph testable without network
calls or paid tokens.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv


DEFAULT_BASE_URL = ""
PLATFORM_MODEL = "qwen/qwen3.8-27b"
DEFAULT_MODEL = PLATFORM_MODEL


class ChatClient(Protocol):
    """Compatibility interface for text completions; pipeline roles use generate."""

    model: str

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return assistant text for a chat conversation."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float | None = 120.0
    # Accepted for compatibility with older callers; never sent to the provider.
    max_tokens: int | None = None

    @classmethod
    def from_platform_env(cls) -> "LLMSettings":
        """Use only credentials supplied by Boosters; the model is optional."""
        missing = [key for key in ('OPENAI_API_KEY','OPENAI_BASE_URL') if not os.getenv(key,'').strip()]
        if missing:
            raise ValueError('Platform environment missing: '+', '.join(missing))
        return cls(api_key=os.environ['OPENAI_API_KEY'].strip(),
                   base_url=os.environ['OPENAI_BASE_URL'].strip(),
                   model=os.getenv('OPENAI_MODEL','').strip() or PLATFORM_MODEL)

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "LLMSettings":
        """Load RouterAI settings from ``.env`` and the process environment."""

        load_dotenv(dotenv_path=env_file, override=False)
        api_key = os.getenv("OPENAI_API_KEY", os.getenv("ROUTERAI_API_KEY", "")).strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Configure the platform environment or a local .env."
            )
        base_url = (os.getenv("OPENAI_BASE_URL", os.getenv("ROUTERAI_BASE_URL", "")).strip())
        if not base_url:
            raise ValueError(
                "OPENAI_BASE_URL is not set. Configure the platform environment or a local .env."
            )
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=os.getenv("OPENAI_MODEL", os.getenv("ROUTERAI_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL,
            timeout_seconds=float(os.getenv("ROUTERAI_TIMEOUT_SECONDS", "120")),
        )


class RouterAIChatClient:
    """Backward-compatible stage-one adapter sharing traced request handling."""

    def __init__(self, settings: LLMSettings, *, trace=None, deadline=None) -> None:
        # Lazy import avoids the task-analyzer/client dependency cycle.
        from .agents import RoleClient
        self.model = settings.model
        self._role = RoleClient(settings,deadline=deadline or time.time()+3*settings.timeout_seconds+30,trace=trace)
        self._client = self._role.client

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._role.complete(messages)

    def close(self):
        self._client.close()
