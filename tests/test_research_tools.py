import asyncio

import httpx
import pytest

from mlmom.research_tools import (
    SearchHit,
    SearchProviderError,
    SearxngSearchProvider,
    canonical_url,
    collect_search_results,
)


def test_searxng_client_parses_filters_and_deduplicates_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["format"] == "json"
        assert request.url.params["q"] == "classification f1"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://www.kaggle.com/code/user/notebook#section",
                        "title": "Notebook",
                        "content": "Result snippet",
                        "engines": ["duckduckgo"],
                    },
                    {
                        "url": "https://www.kaggle.com/code/user/notebook",
                        "title": "Duplicate",
                    },
                    {"url": "ftp://invalid.example/file", "title": "Invalid"},
                ]
            },
        )

    provider = SearxngSearchProvider(
        "http://searxng.test",
        transport=httpx.MockTransport(handler),
    )
    hits = asyncio.run(provider.search("classification f1", 5))
    assert hits == [
        SearchHit(
            "https://www.kaggle.com/code/user/notebook",
            "Notebook",
            "Result snippet",
            ("duckduckgo",),
        )
    ]


def test_searxng_errors_are_provider_independent():
    provider = SearxngSearchProvider(
        "http://searxng.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    with pytest.raises(SearchProviderError, match="SearXNG request failed"):
        asyncio.run(provider.search("query", 3))


class MappingProvider:
    async def search(self, query: str, limit: int) -> list[SearchHit]:
        if "site:kaggle.com" in query:
            return [
                SearchHit("https://kaggle.com/code/u/n", "Kaggle"),
                SearchHit("https://github.com/wrong/result", "Wrong domain"),
            ]
        if "site:github.com" in query:
            return [SearchHit("https://github.com/u/repo", "GitHub")]
        if "reusable" in query:
            return [
                SearchHit("https://github.com/duplicate/repo", "Excluded"),
                SearchHit("https://huggingface.co/u/model", "Other"),
            ]
        return [SearchHit("https://paperswithcode.com/task/example", "Evidence")]


def test_collection_enforces_priority_domains_and_keeps_other_sources():
    results = asyncio.run(
        collect_search_results(
            MappingProvider(),
            [
                {"platform": "kaggle", "priority": 1, "query": "site:kaggle.com x"},
                {"platform": "github", "priority": 2, "query": "site:github.com x"},
                {"platform": "other", "priority": 3, "query": "x reusable"},
            ],
            ["x evidence"],
            results_per_query=5,
        )
    )
    assert [item["platform"] for item in results] == [
        "kaggle",
        "github",
        "other",
        "evidence",
    ]
    assert [item["priority"] for item in results] == [1, 2, 3, 4]


def test_canonical_url_rejects_non_http_and_removes_fragments():
    assert canonical_url("javascript:alert(1)") is None
    assert canonical_url("https://Example.org/path/#part") == "https://example.org/path"
