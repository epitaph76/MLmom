import asyncio

from mlmom.contracts import ResearchRequest, ResearchResult
from mlmom.research_graph import research_graph, route_after_assessment


def test_fixture_graph_returns_contract_valid_result():
    request = ResearchRequest(
        request_id="graph-test",
        task="Find a CPU baseline for an imbalanced tabular classification task",
        task_type="classification",
        metric="f1",
        dataset_summary={"rows": 50_000, "columns": 20},
        constraints={"compute": "cpu"},
    )
    state = asyncio.run(research_graph.ainvoke(
        {"request": request, "search_round": 0, "max_search_rounds": 2}
    ))
    result = ResearchResult.model_validate(state["result"])
    assert len(result.cards) == 4
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].card_ids == [card.id for card in result.cards]
    assert result.hypotheses[0].proposed_experiment["mock"] is True
    assert state["task_spec"]["constraints"] == {"compute": "cpu"}
    assert state["search_round"] == 1


def test_solution_discovery_is_kaggle_first_and_not_github_only():
    request = ResearchRequest(request_id="source-priority", task="ML task")
    state = asyncio.run(research_graph.ainvoke({"request": request}))
    assert [item["platform"] for item in state["solution_queries"]] == [
        "kaggle",
        "github",
        "other",
    ]
    assert [item["platform"] for item in state["search_results"][:3]] == [
        "kaggle",
        "github",
        "other",
    ]


def test_uncrawled_results_are_never_presented_as_verified_evidence():
    request = ResearchRequest(request_id="fixture-labels", task="ML task")
    result = asyncio.run(research_graph.ainvoke({"request": request}))["result"]
    assert all(card.title.startswith("FIXTURE:") for card in result.cards)
    assert all(
        any("not been crawled" in limitation for limitation in card.limitations)
        for card in result.cards
    )
    assert "not verified" in result.hypotheses[0].rationale


def test_gap_route_is_bounded():
    assert route_after_assessment(
        {"knowledge_gaps": ["missing"], "search_round": 1, "max_search_rounds": 2}
    ) == "research_more"
    assert route_after_assessment(
        {"knowledge_gaps": ["missing"], "search_round": 2, "max_search_rounds": 2}
    ) == "rank"
    assert route_after_assessment(
        {"knowledge_gaps": [], "search_round": 1, "max_search_rounds": 2}
    ) == "rank"
