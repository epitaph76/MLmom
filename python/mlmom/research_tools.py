"""Search providers and source-normalization helpers for research workflows."""

from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx


class SearchProviderError(RuntimeError):
    """A safe, provider-independent search failure."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    url: str
    title: str
    snippet: str = ""
    engines: tuple[str, ...] = ()


class SearchProvider(Protocol):
    async def search(self, query: str, limit: int) -> list[SearchHit]: ...


def canonical_url(value: str) -> str | None:
    """Accept HTTP(S) URLs and remove fragments/trailing-root noise."""

    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def classify_solution_platform(url: str) -> str:
    """Classify by the actual result host rather than by the query used."""

    host = (urlsplit(url).hostname or "").lower()
    if host == "kaggle.com" or host.endswith(".kaggle.com"):
        return "kaggle"
    if host == "github.com" or host.endswith(".github.com"):
        return "github"
    return "other"


class SearxngSearchProvider:
    """Minimal async client for the SearXNG JSON Search API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        language: str = "en",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("SEARXNG_URL must use http:// or https://")
        if timeout_seconds <= 0:
            raise ValueError("SearXNG timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.language = language
        self.transport = transport

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if limit < 1:
            raise ValueError("Search result limit must be positive")

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
                headers={"User-Agent": "MLmom/0.1 research service"},
            ) as client:
                response = await client.get(
                    "/search",
                    params={
                        "q": query,
                        "format": "json",
                        "language": self.language,
                        "safesearch": 1,
                        "categories": "general",
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchProviderError(
                f"SearXNG request failed ({type(exc).__name__})"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise SearchProviderError("SearXNG returned invalid JSON") from exc
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise SearchProviderError("SearXNG response has no results list")

        hits: list[SearchHit] = []
        seen: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = canonical_url(str(item.get("url", "")))
            if not url or url in seen:
                continue
            raw_engines = item.get("engines") or []
            if not isinstance(raw_engines, list):
                raw_engines = []
            engine = item.get("engine")
            if engine and engine not in raw_engines:
                raw_engines.append(engine)
            seen.add(url)
            hits.append(
                SearchHit(
                    url=url,
                    title=str(item.get("title") or url),
                    snippet=str(item.get("content") or ""),
                    engines=tuple(str(value) for value in raw_engines),
                )
            )
            if len(hits) >= limit:
                break
        return hits


class FixtureSearchProvider:
    """Offline provider with clearly marked results for deterministic tests."""

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        lowered = query.lower()
        if "site:kaggle.com" in lowered:
            hit = SearchHit(
                "https://www.kaggle.com/code/example/mlmom-fixture",
                "FIXTURE: kaggle solution candidate",
                "Fixture only; no Kaggle request was made.",
                ("fixture",),
            )
        elif "site:github.com" in lowered:
            hit = SearchHit(
                "https://github.com/example/mlmom-fixture",
                "FIXTURE: github solution candidate",
                "Fixture only; no GitHub request was made.",
                ("fixture",),
            )
        elif "reusable ml solution" in lowered:
            hit = SearchHit(
                "https://huggingface.co/example/mlmom-fixture",
                "FIXTURE: other solution candidate",
                "Fixture only; no external request was made.",
                ("fixture",),
            )
        else:
            hit = SearchHit(
                "https://example.org/mlmom/evidence-fixture",
                "FIXTURE: supporting research source",
                "Fixture only; no evidence search was performed.",
                ("fixture",),
            )
        return [hit][:limit]


async def collect_search_results(
    provider: SearchProvider,
    solution_queries: list[dict[str, Any]],
    evidence_queries: list[str],
    *,
    results_per_query: int,
) -> list[dict[str, Any]]:
    """Search in priority order, validate domains and deduplicate URLs."""

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in sorted(solution_queries, key=lambda value: value["priority"]):
        expected_platform = item["platform"]
        hits = await provider.search(item["query"], results_per_query)
        for hit in hits:
            actual_platform = classify_solution_platform(hit.url)
            if expected_platform != actual_platform:
                continue
            if hit.url in seen:
                continue
            seen.add(hit.url)
            results.append(
                {
                    "url": hit.url,
                    "title": hit.title,
                    "snippet": hit.snippet,
                    "engines": list(hit.engines),
                    "source_type": "github" if actual_platform == "github" else "web",
                    "platform": actual_platform,
                    "priority": item["priority"],
                    "query": item["query"],
                }
            )

    for query in evidence_queries:
        hits = await provider.search(query, results_per_query)
        for hit in hits:
            if hit.url in seen:
                continue
            seen.add(hit.url)
            results.append(
                {
                    "url": hit.url,
                    "title": hit.title,
                    "snippet": hit.snippet,
                    "engines": list(hit.engines),
                    "source_type": (
                        "github"
                        if classify_solution_platform(hit.url) == "github"
                        else "web"
                    ),
                    "platform": "evidence",
                    "priority": 4,
                    "query": query,
                }
            )
    return results
