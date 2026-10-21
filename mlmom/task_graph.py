"""LangGraph workflow for LLM task analysis and deterministic validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .contracts import (
    DatasetProfile,
    TaskAnalysisResult,
    TaskDefinition,
    TaskValidationReport,
)
from .llm_client import ChatClient, LLMSettings, RouterAIChatClient
from .task_analyzer import TaskAnalyzer
from .task_validator import TaskValidator
from .tracing import span, emit, TraceSink


class TaskGraphState(TypedDict, total=False):
    profile: DatasetProfile
    task: TaskDefinition | None
    raw_response: str
    generation_error: str | None
    validation: TaskValidationReport
    attempts: int
    max_attempts: int


def build_task_graph(client: ChatClient):
    """Compile analyze -> validate -> repair workflow for one chat client."""

    analyzer = TaskAnalyzer(client)
    validator = TaskValidator()

    def generate(state: TaskGraphState) -> TaskGraphState:
        result = analyzer.analyze(state["profile"])
        emit('task_analysis.attempt', {'raw_response':result.raw_response,'error':result.error,'task':result.task},
             attempt=state.get('attempts',0)+1)
        return {
            "task": result.task,
            "raw_response": result.raw_response,
            "generation_error": result.error,
            "attempts": state.get("attempts", 0) + 1,
        }

    def repair(state: TaskGraphState) -> TaskGraphState:
        result = analyzer.analyze(
            state["profile"],
            previous_task=state.get("task"),
            feedback=state["validation"].findings,
        )
        emit('task_analysis.attempt', {'raw_response':result.raw_response,'error':result.error,'task':result.task},
             attempt=state['attempts']+1)
        return {
            "task": result.task,
            "raw_response": result.raw_response,
            "generation_error": result.error,
            "attempts": state["attempts"] + 1,
        }

    def validate(state: TaskGraphState) -> TaskGraphState:
        task = state.get('task')
        if task is not None:
            exclusions = list(dict.fromkeys(task.excluded_feature_columns + task.target_columns + task.id_columns))
            if exclusions != task.excluded_feature_columns:
                emit('task_analysis.normalized_exclusions', {'before':task.excluded_feature_columns,'after':exclusions})
                task = task.model_copy(update={'excluded_feature_columns':exclusions})
        report = validator.validate(
            state["profile"],
            task,
            generation_error=state.get("generation_error"),
        )
        emit('task_analysis.validation', report)
        return {"validation": report, 'task':task}

    def route(state: TaskGraphState) -> str:
        if state["validation"].valid:
            return "finish"
        if state["attempts"] >= state["max_attempts"]:
            return "finish"
        return "repair"

    graph = StateGraph(TaskGraphState)
    graph.add_node("analyze", generate)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "validate")
    graph.add_conditional_edges(
        "validate",
        route,
        {"repair": "repair", "finish": END},
    )
    graph.add_edge("repair", "validate")
    # Stage one has its own Pydantic state; the outer runtime persists its JSON result.
    # Do not inherit an enclosing graph's JSON-only checkpointer.
    return graph.compile(checkpointer=False)


def analyze_task(
    profile: DatasetProfile,
    client: ChatClient,
    *,
    max_attempts: int = 3,
) -> TaskAnalysisResult:
    """Run the graph and return a persistable validated-or-failed result."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    with span('task_analysis', stage='inspection'):
        final = build_task_graph(client).invoke({
            "profile": profile,
            "attempts": 0,
            "max_attempts": max_attempts,
        })
    validation = final["validation"]
    return TaskAnalysisResult(
        status="validated" if validation.valid else "failed",
        attempts=final["attempts"],
        model=client.model,
        task=final.get("task"),
        validation=validation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and validate an AutoML TaskDefinition with RouterAI."
    )
    parser.add_argument("--profile", type=Path, required=True, help="Data Inspector profile.json")
    parser.add_argument("--output", type=Path, required=True, help="Validated task JSON")
    parser.add_argument("--env-file", help="Optional path to a local .env file")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    profile = DatasetProfile.model_validate_json(args.profile.read_text(encoding="utf-8"))
    settings = LLMSettings.from_env(args.env_file)
    import uuid
    run_id='task-analysis-'+uuid.uuid4().hex
    trace = TraceSink(args.output.parent/'trace'/run_id,run_id=run_id,
                      task_id=profile.dataset_name,secrets=[settings.api_key])
    client = RouterAIChatClient(settings,trace=trace)
    try:
        with trace.activate(stage='inspection',role='task_definition'):
            result = analyze_task(profile,client,max_attempts=args.max_attempts)
    finally:
        client.close()
        trace.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result.status,
        "attempts": result.attempts,
        "model": result.model,
        "output": str(args.output),
        "errors": sum(
            finding.severity == "error" for finding in result.validation.findings
        ),
    }, ensure_ascii=False, indent=2))
    if result.status != "validated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
