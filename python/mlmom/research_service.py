"""Contract-compatible asynchronous Research API backed by LangGraph."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI

from mlmom.contracts import ApiError, Health, ResearchJob, ResearchRequest, ResearchResult
from mlmom.http import error_response, install_validation_handler
from mlmom.research_graph import build_configured_research_graph, research_graph

TERMINAL = {"succeeded", "failed", "cancelled"}


def create_app(
    graph: Any = research_graph,
    max_search_rounds: int | None = None,
    mock: bool = True,
) -> FastAPI:
    rounds = max_search_rounds
    if rounds is None:
        rounds = int(os.getenv("RESEARCH_MAX_SEARCH_ROUNDS", "2"))
    if rounds < 1:
        raise ValueError("RESEARCH_MAX_SEARCH_ROUNDS must be at least 1")

    jobs: dict[str, ResearchJob] = {}
    requests: dict[str, tuple[dict[str, Any], str]] = {}
    tasks: dict[str, asyncio.Task[None]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    app = FastAPI(title="MLmom Research API", version="1.0.0", lifespan=lifespan)
    install_validation_handler(app, mock=mock)
    errors = {404: {"model": ApiError}, 409: {"model": ApiError}, 422: {"model": ApiError}}

    async def run(job_id: str, request: ResearchRequest) -> None:
        await asyncio.sleep(0)
        if jobs[job_id].status in TERMINAL:
            return
        jobs[job_id] = jobs[job_id].model_copy(update={"status": "running"})
        try:
            state = await graph.ainvoke(
                {
                    "request": request,
                    "search_round": 0,
                    "max_search_rounds": rounds,
                }
            )
            result = ResearchResult.model_validate(state["result"])
        except asyncio.CancelledError:
            if jobs[job_id].status not in TERMINAL:
                jobs[job_id] = jobs[job_id].model_copy(update={"status": "cancelled"})
            return
        except Exception:
            if jobs[job_id].status not in TERMINAL:
                error = ApiError(
                    code="RESEARCH_WORKFLOW_FAILED",
                    message="Research workflow failed",
                    retryable=False,
                    request_id=request.request_id,
                    mock=mock,
                )
                jobs[job_id] = jobs[job_id].model_copy(
                    update={"status": "failed", "error": error}
                )
            return
        if jobs[job_id].status not in TERMINAL:
            jobs[job_id] = jobs[job_id].model_copy(
                update={"status": "succeeded", "result": result}
            )

    @app.get("/health", response_model=Health)
    async def health():
        return Health(mock=mock)

    @app.post(
        "/v1/research/jobs",
        response_model=ResearchJob,
        status_code=202,
        responses=errors,
    )
    async def create_job(request: ResearchRequest):
        body = request.model_dump(mode="json")
        previous = requests.get(request.request_id)
        if previous:
            old_body, job_id = previous
            if old_body != body:
                return error_response(
                    409,
                    "REQUEST_ID_CONFLICT",
                    "request_id already used for a different request",
                    request.request_id,
                    mock=mock,
                )
            return jobs[job_id]
        job_id = str(uuid4())
        job = ResearchJob(
            job_id=job_id,
            request_id=request.request_id,
            status="queued",
            mock=mock,
        )
        jobs[job_id] = job
        requests[request.request_id] = (body, job_id)
        tasks[job_id] = asyncio.create_task(run(job_id, request))
        tasks[job_id].add_done_callback(
            lambda _task, completed_job_id=job_id: tasks.pop(completed_job_id, None)
        )
        return job

    @app.get(
        "/v1/research/jobs/{job_id}", response_model=ResearchJob, responses=errors
    )
    async def get_job(job_id: str):
        if job_id not in jobs:
            return error_response(
                404, "JOB_NOT_FOUND", "Research job does not exist", mock=mock
            )
        return jobs[job_id]

    @app.post(
        "/v1/research/jobs/{job_id}/cancel",
        response_model=ResearchJob,
        responses=errors,
    )
    async def cancel_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return error_response(
                404, "JOB_NOT_FOUND", "Research job does not exist", mock=mock
            )
        if job.status not in TERMINAL:
            jobs[job_id] = job.model_copy(update={"status": "cancelled"})
            task = tasks.get(job_id)
            if task:
                task.cancel()
        return jobs[job_id]

    return app


# Search is real when SEARCH_PROVIDER=searxng, while mock stays true until
# Crawl4AI and evidence analysis replace the remaining placeholder nodes.
app = create_app(graph=build_configured_research_graph(), mock=True)
