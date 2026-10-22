"""JSON-only runtime contracts. Stage-one contracts remain backward compatible."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import Field, model_validator
from .contracts import StrictModel, TaskDefinition, TaskType, TableReadOptions


class RunConfig(StrictModel):
    profile: Literal['dev', 'platform'] = 'platform'
    total_seconds: float = Field(10800, gt=0)
    final_reserve_seconds: float = Field(2400, ge=0)
    first_submission_seconds: float = Field(600, gt=0)
    experiment_seconds: float = Field(1800, gt=0)
    initial_experiment_seconds: float = Field(300, gt=0, le=300)
    coding_seconds: float = Field(300, gt=0, le=300)
    prediction_seconds: float = Field(180, gt=0)
    max_rounds: int = Field(100, ge=1)
    plateau_rounds: int = Field(5, ge=1)
    max_candidates: int = Field(2, ge=1, le=2)
    max_workers: int = Field(3, ge=1, le=3)
    max_attempts: int = Field(3, ge=1, le=10)
    runtime_repairs: int = Field(2, ge=0, le=5)
    agent_steps: int = Field(12, ge=1, le=50)
    holdout_fraction: float = Field(.2, gt=0, lt=.5)
    inner_fraction: float = Field(.2, gt=0, lt=.5)
    seed: int = 42
    max_trials: int = Field(20, ge=1)
    patience_trials: int = Field(5, ge=1)
    selection_threshold: float = Field(.01, ge=0)
    reserve_cpus: int = Field(1, ge=0)
    reserve_memory_gb: float = Field(4, ge=0)
    smoke_seconds: float = Field(30, gt=0)
    llm_timeout_seconds: float = Field(300, gt=0)
    # Retired task-definition role; accepted only for old run configurations.
    task_definition_seconds: float = Field(180, gt=0)
    hypothesis_seconds: float = Field(210, gt=0)
    planner_seconds: float = Field(210, gt=0)
    validator_seconds: float = Field(210, gt=0)
    coding_verification_seconds: float = Field(60, gt=0)
    coding_direct_first: bool = True
    coding_tool_steps: int = Field(3, ge=1, le=12)
    planning_direct_first: bool = True
    planning_context_reads: int = Field(2, ge=0, le=5)
    # Legacy configuration key retained for resume compatibility; ignored by clients.
    llm_max_tokens: int | None = None
    metric_options: dict[str, Any] = Field(default_factory=dict)
    submission_kind: str | None = None
    hpo_enabled: bool = True
    hpo_batch_size: int = Field(5, ge=1)
    hpo_max_trials: int = Field(15, ge=1)
    hpo_gain_threshold: float = Field(.005, ge=0)
    hpo_seconds: float = Field(600, gt=0)
    diagnostic_seconds: float = Field(60, ge=0)
    diagnostic_rows: int = Field(1000, ge=1)
    diagnostic_features: int = Field(20, ge=1)
    target_completed_rounds: int = Field(0, ge=0)
    fixed_outer_split: bool = False
    training_control_seconds: float = Field(30, gt=0, le=30)
    initial_reconnaissance_seconds: float = Field(300, gt=0)
    reconnaissance_seconds: float = Field(120, gt=0)
    reconnaissance_steps: int = Field(20, ge=1, le=50)
    reconnaissance_query_seconds: float = Field(30, gt=0)
    context_seconds: float = Field(120, gt=0)
    result_validation_seconds: float = Field(30, gt=0, le=60)

    @model_validator(mode='after')
    def valid_reserve(self):
        if self.final_reserve_seconds >= self.total_seconds:
            raise ValueError('final_reserve_seconds must be smaller than total_seconds')
        return self


class InputFiles(StrictModel):
    task: str
    root: str
    train: str
    test: str
    dictionaries: list[str]
    sample_submission: str | None = None
    manifest: str | None = None


class TaskContext(StrictModel):
    task_id: str
    task: TaskDefinition
    profile_ref: str
    files: InputFiles
    feature_columns: list[str]
    classes: list[Any] = Field(default_factory=list)
    dataset_hash: str
    metric_options: dict[str, Any] = Field(default_factory=dict)
    submission_kind: str = 'value'
    table_formats: dict[str, TableReadOptions] = Field(default_factory=dict)


class SkillManifest(StrictModel):
    skill_id: str
    version: str
    title: str
    status: Literal['placeholder', 'ready']
    task_types: list[TaskType] = Field(default_factory=list)
    ml_tools: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class MLToolSpec(StrictModel):
    tool_id: str
    distribution: str
    import_name: str
    version: str | None = None
    expected_version: str
    available: bool = False
    reason: str | None = None
    task_types: list[TaskType]
    prediction_kinds: list[str]
    skill_ids: list[str]
    cost_tag: Literal['fast', 'medium', 'slow']
    native_hpo: bool = False
    supports_weights: bool = False
    supports_groups: bool = False
    supports_patience: bool = False
    adapter: str


class BudgetSnapshot(StrictModel):
    remaining_seconds: float
    research_seconds: float
    available_cpus: int
    available_memory_gb: float
    spent_seconds: float


class Hypothesis(StrictModel):
    hypothesis_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    description: str
    rationale: str
    baseline: bool = False
    based_on: str | None = None
    based_on_hypothesis_id: str | None = None


class HypothesisBatch(StrictModel):
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=2)


class HypothesisPlan(StrictModel):
    hypothesis: Hypothesis
    tool_id: str
    skill_ids: list[str]
    model_params: dict[str, Any] = Field(default_factory=dict)
    preprocessing: list[str] = Field(default_factory=list)
    strategy: Literal['single', 'optuna', 'native'] = 'single'
    search_space: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cost_tag: Literal['fast', 'medium', 'slow'] = 'fast'
    time_limit_seconds: float = Field(300, gt=0)
    cpus: int = Field(2, ge=1)
    memory_gb: float = Field(4, gt=0)
    max_trials: int = Field(20, ge=1, le=1000)
    patience_trials: int | None = Field(5, ge=1)
    min_delta: float = Field(0, ge=0, description='Legacy absolute comparison threshold; NOT the HPO continuation threshold. Batched search uses RunConfig.hpo_gain_threshold; keep min_delta=0 to retain every improvement.')
    prediction_kind: str = 'value'
    ensemble_components: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: Literal['custom'] = 'custom'


class PlanBatch(StrictModel):
    plans: list[HypothesisPlan] = Field(min_length=1, max_length=2)


class ValidationReport(StrictModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CandidateVersion(StrictModel):
    candidate_id: str
    hypothesis_id: str
    version: int
    code_dir: str
    code_hash: str
    plan_ref: str
    skill_versions: dict[str, dict[str, str]] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)
    based_on: str | None = None
    verification: ValidationReport
    preprocessing_hash: str | None = None


class SplitSpec(StrictModel):
    split_id: str
    round_id: str
    dataset_hash: str
    seed: int
    strategy: str
    train_indices: list[int]
    holdout_indices: list[int]


class TrialResult(StrictModel):
    number: int
    params: dict[str, Any]
    status: str
    score: float | None = None
    duration_seconds: float
    error: str | None = None

    kind: str = 'optuna'


class ExperimentRun(StrictModel):
    experiment_id: str
    round_id: str
    candidate_id: str
    split_id: str
    status: str = 'queued'
    role: Literal['evaluation', 'submission_refit'] = 'evaluation'
    incumbent: bool = False
    model_artifact: str | None = None
    best_params: dict[str, Any] = Field(default_factory=dict)
    internal_score: float | None = None
    trials: list[TrialResult] = Field(default_factory=list)
    stop_reason: str | None = None
    duration_seconds: float = 0
    peak_memory_gb: float = 0
    resource_cpus: int = 1
    resource_memory_gb: float = 2
    native_history_ref: str | None = None
    error: str | None = None
    study_ref: str | None = None
    hpo_batches: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationResult(StrictModel):
    experiment_id: str
    split_id: str
    metric: str
    direction: str
    score: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class Champion(StrictModel):
    candidate: CandidateVersion
    plan: HypothesisPlan
    run: ExperimentRun
    evaluation: EvaluationResult


class SelectedPreviousResults(StrictModel):
    round_id: str
    selected: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    champion_id: str | None = None
    decisions: list[dict[str, Any]] = Field(default_factory=list)


class SubmissionArtifact(StrictModel):
    path: str
    experiment_id: str
    candidate_id: str
    model_role: str
    rows: int
    sha256: str


class ToolResult(StrictModel):
    status: Literal['ok', 'error']
    data: Any = None
    error: str | None = None
    duration_seconds: float
    artifacts: list[str] = Field(default_factory=list)


class Observation(StrictModel):
    observation_id: str
    topic: str
    description: str
    rows: int = Field(ge=0)
    numbers: dict[str, float] = Field(default_factory=dict)
    source_ref: str


class ResearchProposal(StrictModel):
    description: str
    evidence_ids: list[str] = Field(min_length=1)
    expected_effect: str
    verification: str


class ReconnaissanceReport(StrictModel):
    status: Literal['completed', 'partial', 'unavailable'] = 'completed'
    observations: list[Observation] = Field(default_factory=list)
    proposals: list[ResearchProposal] = Field(default_factory=list)
    reason: str | None = None
    unfinished_checks: list[str] = Field(default_factory=list)


class TaskDetail(StrictModel):
    meaning: str = Field(min_length=1, max_length=600)
    source_quote: str = Field(min_length=1, max_length=1200)


class RankedFinding(StrictModel):
    finding_id: str
    priority: Literal['critical', 'high', 'medium', 'low']
    rationale: str = Field(min_length=1, max_length=600)
    planner_action: str = Field(min_length=1, max_length=600)


class PlannerContextDraft(StrictModel):
    task_summary: str = Field(min_length=1, max_length=4000)
    task_details: list[TaskDetail] = Field(default_factory=list, max_length=20)
    ranked_findings: list[RankedFinding] = Field(default_factory=list, max_length=24)
    assumptions: list[str] = Field(default_factory=list, max_length=12)


class ResultReview(StrictModel):
    implementation: Literal['matches', 'partial', 'mismatch', 'unknown']
    effect: Literal['supported', 'not_supported', 'inconclusive']
    evidence_refs: list[str] = Field(default_factory=list)
    explanation: str
    recommendations: list[str] = Field(default_factory=list)


class PreprocessingSpec(StrictModel):
    module: str = 'preprocessing.py'
    factory: str = 'create_preprocessor'
    description: str
    output: Literal['dataframe', 'dense', 'sparse']
    constraints: list[str] = Field(default_factory=list)


class PreprocessingBundle(StrictModel):
    source: str
    spec: PreprocessingSpec


class TrainingCheckpoint(StrictModel):
    checkpoint_id: str
    stage: Literal['initial_fit', 'batch', 'failure']
    plan: dict[str, Any]
    trials: list[dict[str, Any]] = Field(default_factory=list)
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_score: float | None = None
    remaining_seconds: float = Field(ge=0)
    remaining_trials: int = Field(ge=0)
    default_action: Literal['continue', 'stop', 'refit']
    error: str | None = None


class TrainingDecision(StrictModel):
    action: Literal['continue', 'stop', 'refit']
    reason: str
    next_params: list[dict[str, Any]] = Field(default_factory=list, max_length=5)
    recommendations: list[str] = Field(default_factory=list)
