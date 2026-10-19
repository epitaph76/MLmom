"""Research v1 models. Exported OpenAPI is the language-independent handoff."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

CONTRACT_VERSION = "1.0.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApiError(StrictModel):
    code: str
    message: str
    retryable: bool = False
    request_id: str | None = None
    mock: bool = False


class Health(StrictModel):
    status: Literal["ok"] = "ok"
    contract_version: str = CONTRACT_VERSION
    mock: bool = False


class ResearchRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    task: str = Field(min_length=1, max_length=20000)
    task_type: str | None = None
    metric: str | None = None
    dataset_summary: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None


class ResearchCard(StrictModel):
    id: str
    url: HttpUrl
    title: str
    source_type: Literal["web", "github", "paper"]
    findings: list[str]
    limitations: list[str]


class Hypothesis(StrictModel):
    id: str
    description: str
    rationale: str
    card_ids: list[str]
    proposed_experiment: dict[str, Any]


class ResearchResult(StrictModel):
    cards: list[ResearchCard]
    hypotheses: list[Hypothesis]

    @model_validator(mode="after")
    def valid_references(self):
        ids = [card.id for card in self.cards]
        hypothesis_ids = [hypothesis.id for hypothesis in self.hypotheses]
        if len(set(ids)) != len(ids) or len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("Card and hypothesis IDs must be unique within each list")
        if any(set(h.card_ids) - set(ids) for h in self.hypotheses):
            raise ValueError("Hypotheses must reference existing cards")
        return self


class ResearchJob(StrictModel):
    job_id: str
    request_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    result: ResearchResult | None = None
    error: ApiError | None = None
    mock: bool = False

    @model_validator(mode="after")
    def valid_state(self):
        if (self.status == "succeeded") != (self.result is not None):
            raise ValueError("Only succeeded jobs must have a result")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("Only failed jobs must have an error")
        return self


class Capabilities(StrictModel):
    stage: Literal["infrastructure"] = "infrastructure"
    implemented: list[str] = ["health", "capabilities", "history"]
    unavailable: list[str] = ["profile_data", "train", "evaluate"]
    experiment_interface: Literal["structured_recipe"] = "structured_recipe"


class ProfileRequest(StrictModel):
    dataset_ref: str = Field(min_length=1)


class TrainRequest(StrictModel):
    dataset_ref: str = Field(min_length=1)
    recipe: dict[str, Any]


class EvaluateRequest(StrictModel):
    experiment_id: str = Field(min_length=1)


class History(StrictModel):
    experiments: list[dict[str, Any]] = Field(default_factory=list)
