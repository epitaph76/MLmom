"""In-memory reference implementation for the teammate's Research HTTP API."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, Header

from mlmom.contracts import ApiError, Health, Hypothesis, ResearchCard, ResearchJob, ResearchRequest, ResearchResult
from mlmom.http import error_response, install_validation_handler

TERMINAL = {"succeeded", "failed", "cancelled"}


def create_app(delay_ms: int | None = None) -> FastAPI:
    delay = (delay_ms if delay_ms is not None else int(os.getenv("RESEARCH_MOCK_DELAY_MS", "250"))) / 1000
    if delay < 0:
        raise ValueError("RESEARCH_MOCK_DELAY_MS must be nonnegative")
    jobs: dict[str, ResearchJob] = {}
    requests: dict[str, tuple[dict, str]] = {}
    tasks: dict[str, asyncio.Task] = {}

    @asynccontextmanager
    async def lifespan(app):
        yield
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    app = FastAPI(title="MLmom Research API", version="1.0.0", lifespan=lifespan)
    install_validation_handler(app, mock=True)
    errors = {404: {"model": ApiError}, 409: {"model": ApiError}, 422: {"model": ApiError}}

    async def run(job_id: str, scenario: str):
        await asyncio.sleep(0)
        if jobs[job_id].status in TERMINAL:
            return
        job = jobs[job_id]
        jobs[job_id] = job.model_copy(update={"status": "running"})
        await asyncio.sleep(delay)
        if jobs[job_id].status in TERMINAL:
            return
        if scenario == "failed":
            error = ApiError(code="MOCK_FAILURE", message="Synthetic research failure", request_id=job.request_id, mock=True)
            jobs[job_id] = job.model_copy(update={"status": "failed", "error": error})
            return
        result = ResearchResult(cards=[], hypotheses=[])
        if scenario == "success":
            result = ResearchResult(
                cards=[ResearchCard(id="mock-card-1", url="https://example.org/mock-research", title="MOCK: synthetic research card", source_type="web", findings=["Synthetic fixture; no search was performed"], limitations=["Not evidence for an ML decision"])],
                hypotheses=[Hypothesis(id="mock-hypothesis-1", description="MOCK: compare a baseline", rationale="Tests transport only", card_ids=["mock-card-1"], proposed_experiment={"kind": "baseline", "mock": True})],
            )
        jobs[job_id] = job.model_copy(update={"status": "succeeded", "result": result})

    @app.get("/health", response_model=Health)
    async def health():
        return Health(mock=True)

    @app.post("/v1/research/jobs", response_model=ResearchJob, status_code=202, responses=errors)
    async def create_job(request: ResearchRequest, x_mock_scenario: Annotated[Literal["success", "empty", "failed"], Header(description="Mock only; real services may ignore this header")] = "success"):
        body = request.model_dump(mode="json")
        previous = requests.get(request.request_id)
        if previous:
            old_body, job_id = previous
            if old_body != body:
                return error_response(409, "REQUEST_ID_CONFLICT", "request_id already used for a different request", request.request_id, mock=True)
            return jobs[job_id]
        job_id = str(uuid4())
        job = ResearchJob(job_id=job_id, request_id=request.request_id, status="queued", mock=True)
        jobs[job_id] = job
        requests[request.request_id] = (body, job_id)
        tasks[job_id] = asyncio.create_task(run(job_id, x_mock_scenario))
        tasks[job_id].add_done_callback(lambda _: tasks.pop(job_id, None))
        return job

    @app.get("/v1/research/jobs/{job_id}", response_model=ResearchJob, responses=errors)
    async def get_job(job_id: str):
        if job_id not in jobs:
            return error_response(404, "JOB_NOT_FOUND", "Research job does not exist", mock=True)
        return jobs[job_id]

    @app.post("/v1/research/jobs/{job_id}/cancel", response_model=ResearchJob, responses=errors)
    async def cancel_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return error_response(404, "JOB_NOT_FOUND", "Research job does not exist", mock=True)
        if job.status not in TERMINAL:
            jobs[job_id] = job.model_copy(update={"status": "cancelled"})
            if job_id in tasks:
                tasks[job_id].cancel()
        return jobs[job_id]

    return app


app = create_app()
