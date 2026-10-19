"""LangGraph workflow for one bounded ML research job."""

import os
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from mlmom.contracts import Hypothesis, ResearchCard, ResearchRequest, ResearchResult
from mlmom.research_tools import (
    FixtureSearchProvider,
    SearchProvider,
    SearxngSearchProvider,
    collect_search_results,
)

SolutionPlatform = Literal["kaggle", "github", "other"]


class SolutionQuery(TypedDict):
    platform: SolutionPlatform
    priority: int
    query: str


class ResearchState(TypedDict, total=False):
    request: ResearchRequest
    task_spec: dict[str, Any]
    evidence_queries: list[str]
    solution_queries: list[SolutionQuery]
    search_results: list[dict[str, Any]]
    crawled_pages: list[dict[str, Any]]
    cards: list[ResearchCard]
    hypotheses: list[Hypothesis]
    knowledge_gaps: list[str]
    search_round: int
    max_search_rounds: int
    result: ResearchResult


def parse_request(state: ResearchState) -> dict[str, Any]:
    request = state["request"]
    return {
        "task_spec": {
            "task": request.task,
            "task_type": request.task_type or "unspecified",
            "metric": request.metric,
            "dataset_summary": request.dataset_summary or {},
            "constraints": request.constraints or {},
        },
        "search_round": state.get("search_round", 0),
        "max_search_rounds": state.get("max_search_rounds", 2),
    }


def plan_queries(state: ResearchState) -> dict[str, Any]:
    """Create a Kaggle-first structured plan; an LLM will replace the wording."""

    spec = state["task_spec"]
    topic = " ".join(
        value
        for value in (spec["task_type"], spec.get("metric"))
        if value and value != "unspecified"
    ) or "machine learning"
    return {
        "evidence_queries": [
            f"{topic} baseline methods",
            f"{topic} evaluation best practices",
        ],
        "solution_queries": [
            {
                "platform": "kaggle",
                "priority": 1,
                "query": f"site:kaggle.com/code {topic} notebook solution",
            },
            {
                "platform": "github",
                "priority": 2,
                "query": f"site:github.com {topic} open source implementation",
            },
            {
                "platform": "other",
                "priority": 3,
                "query": f"{topic} reusable ML solution implementation",
            },
        ],
    }


def build_research_graph(
    search_provider: SearchProvider | None = None,
    *,
    results_per_query: int = 3,
):
    """Compile a graph with an injectable search provider."""

    if results_per_query < 1:
        raise ValueError("results_per_query must be at least 1")
    provider = search_provider or FixtureSearchProvider()

    async def search_sources(state: ResearchState) -> dict[str, list[dict[str, Any]]]:
        results = await collect_search_results(
            provider,
            state["solution_queries"],
            state["evidence_queries"],
            results_per_query=results_per_query,
        )
        return {"search_results": results}

    def crawl_sources(state: ResearchState) -> dict[str, list[dict[str, Any]]]:
        """Preserve discovered URLs; Crawl4AI replaces placeholder content next."""

        return {
            "crawled_pages": [
                {
                    **source,
                    "markdown": "# Not crawled yet\n\nOnly search metadata is available.",
                    "success": True,
                }
                for source in state["search_results"]
            ]
        }

    def extract_cards(state: ResearchState) -> dict[str, list[ResearchCard]]:
        cards = []
        for index, page in enumerate(state["crawled_pages"], start=1):
            cards.append(
                ResearchCard(
                    id=f"source-{index}",
                    url=page["url"],
                    title=page["title"],
                    source_type=page["source_type"],
                    findings=[
                        page.get("snippet")
                        or "Search result discovered; page content is not analyzed"
                    ],
                    limitations=["Page content has not been crawled or verified"],
                )
            )
        return {"cards": cards}

    def assess_gaps(state: ResearchState) -> dict[str, Any]:
        platforms = {item["platform"] for item in state.get("search_results", [])}
        gaps = [
            f"No {platform} solution candidate was found"
            for platform in ("kaggle", "github", "other")
            if platform not in platforms
        ]
        if "evidence" not in platforms:
            gaps.append("No supporting evidence source was found")
        return {
            "knowledge_gaps": gaps,
            "search_round": state.get("search_round", 0) + 1,
        }

    def rank_solutions(state: ResearchState) -> dict[str, list[Hypothesis]]:
        card_ids = [card.id for card in state.get("cards", [])]
        if not card_ids:
            return {"hypotheses": []}
        return {
            "hypotheses": [
                Hypothesis(
                    id="fixture-hypothesis-1",
                    description="FIXTURE: compare a deterministic baseline",
                    rationale=(
                        "Search candidates were discovered in Kaggle-first order, "
                        "but their page content and model quality are not verified."
                    ),
                    card_ids=card_ids,
                    proposed_experiment={
                        "kind": "baseline",
                        "source_priority": ["kaggle", "github", "other"],
                        "mock": True,
                    },
                )
            ]
        }

    def build_result(state: ResearchState) -> dict[str, ResearchResult]:
        return {
            "result": ResearchResult(
                cards=state.get("cards", []),
                hypotheses=state.get("hypotheses", []),
            )
        }

    builder = StateGraph(ResearchState)
    builder.add_node("parse_request", parse_request)
    builder.add_node("plan_queries", plan_queries)
    builder.add_node("search_sources", search_sources)
    builder.add_node("crawl_sources", crawl_sources)
    builder.add_node("extract_cards", extract_cards)
    builder.add_node("assess_gaps", assess_gaps)
    builder.add_node("rank_solutions", rank_solutions)
    builder.add_node("build_result", build_result)
    builder.add_edge(START, "parse_request")
    builder.add_edge("parse_request", "plan_queries")
    builder.add_edge("plan_queries", "search_sources")
    builder.add_edge("search_sources", "crawl_sources")
    builder.add_edge("crawl_sources", "extract_cards")
    builder.add_edge("extract_cards", "assess_gaps")
    builder.add_conditional_edges(
        "assess_gaps",
        route_after_assessment,
        {"research_more": "plan_queries", "rank": "rank_solutions"},
    )
    builder.add_edge("rank_solutions", "build_result")
    builder.add_edge("build_result", END)
    return builder.compile()


def route_after_assessment(state: ResearchState) -> Literal["research_more", "rank"]:
    if state.get("knowledge_gaps") and state["search_round"] < state["max_search_rounds"]:
        return "research_more"
    return "rank"


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def build_configured_research_graph():
    """Select fixture or SearXNG search from environment configuration."""

    provider_name = os.getenv("SEARCH_PROVIDER", "fixture").lower()
    if provider_name == "fixture":
        provider: SearchProvider = FixtureSearchProvider()
    elif provider_name == "searxng":
        provider = SearxngSearchProvider(
            os.getenv("SEARXNG_URL", "http://127.0.0.1:8088"),
            timeout_seconds=_positive_int("SEARXNG_TIMEOUT_MS", 10_000) / 1000,
            language=os.getenv("SEARXNG_LANGUAGE", "en"),
        )
    else:
        raise ValueError("SEARCH_PROVIDER must be fixture or searxng")
    return build_research_graph(
        provider,
        results_per_query=_positive_int("SEARXNG_RESULTS_PER_QUERY", 3),
    )


research_graph = build_research_graph()
