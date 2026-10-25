"""JSON-only runtime contracts. Stage-one contracts remain backward compatible."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import Field, model_validator
from .contracts import StrictModel, TaskDefinition, TaskType, TableReadOptions
from .agent_contracts import AgentNarrative
from .model_spec import TargetTransform, WeightPolicy, PredictionPolicy, Intervention
from .feature_contracts import FeatureSpec, FeatureSet, FeatureEvidence, FeatureSession


class RunConfig(StrictModel):
    initial_boosting_sweep: bool = False  # Legacy opt-in; default starters are agent-selected.
    preparation_workers: int = Field(3, ge=1, le=3)
    cycle_full_refit: bool = True
    feature_search_enabled: bool = True
    feature_max_configurations: int = Field(6, ge=1, le=6)
    feature_plateau_limit: int = Field(3, ge=1, le=6)
    feature_holdout_folds: int = Field(3, ge=3, le=3)
    feature_agent_seconds: float = Field(120, gt=0)
    feature_min_delta: float = Field(0, ge=0)
    thinking_mode: Literal['default', 'off'] = 'default'
    role_thinking: dict[str, Literal['default', 'off']] = Field(default_factory=lambda: {'context_agent':'off', 'training_agent':'off'})
    network_failure_threshold: int = Field(3, ge=1)
    network_cooldown_seconds: float = Field(30, gt=0)
    timeouts_enabled: bool = False
    profile: Literal['dev', 'platform'] = 'platform'
    total_seconds: float = Field(10800, gt=0)
    final_reserve_seconds: float = Field(2400, ge=0)
    first_submission_seconds: float = Field(600, gt=0)
    experiment_seconds: float = Field(1800, gt=0)
    initial_experiment_seconds: float = Field(300, gt=0)
    coding_seconds: float = Field(300, gt=0)
    prediction_seconds: float = Field(180, gt=0)
    max_rounds: int | None = Field(None, ge=1)
    plateau_rounds: int | None = Field(None, ge=1)
    max_candidates: int = Field(2, ge=1, le=2)
    ensemble_search_enabled: bool = True
    ensemble_agent_seconds: float = Field(120, gt=0)
    max_workers: int = Field(3, ge=1, le=3)
    max_attempts: int = Field(3, ge=1, le=10)
    runtime_repairs: int | None = Field(None, ge=0)
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
    coding_verification_seconds: float = Field(60, gt=0)  # Legacy configuration; no mandatory smoke reserve.
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
    hpo_initial_trials: int = Field(15, ge=1)
    hpo_max_trials: int | None = Field(None, ge=1)
    hpo_startup_trials: int = Field(3, ge=1)
    hpo_gain_threshold: float = Field(.005, ge=0)
    hpo_seconds: float = Field(600, gt=0)
    diagnostic_seconds: float = Field(60, ge=0)
    diagnostic_rows: int = Field(1000, ge=1)
    diagnostic_features: int = Field(20, ge=1)
    target_completed_rounds: int = Field(0, ge=0)
    fixed_outer_split: bool = False
    training_control_seconds: float = Field(30, gt=0)
    initial_reconnaissance_seconds: float = Field(300, gt=0)
    reconnaissance_seconds: float = Field(120, gt=0)
    reconnaissance_steps: int = Field(20, ge=1, le=50)
    reconnaissance_query_seconds: float = Field(30, gt=0)
    context_seconds: float = Field(120, gt=0)
    result_validation_seconds: float = Field(30, gt=0)

    @model_validator(mode='after')
    def valid_reserve(self):
        if self.timeouts_enabled and self.final_reserve_seconds >= self.total_seconds:
            raise ValueError('final_reserve_seconds must be smaller than total_seconds')
        return self

    @property
    def continuous(self):
        return (not self.timeouts_enabled and self.max_rounds is None and
                self.plateau_rounds is None and not self.target_completed_rounds)


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
    validation_strategies: list[str] = Field(default_factory=lambda: ['holdout','kfold','stratified_kfold','group_kfold'])


class BudgetSnapshot(StrictModel):
    timeouts_enabled: bool = True
    remaining_seconds: float
    research_seconds: float
    available_cpus: int
    available_memory_gb: float
    spent_seconds: float

    def agent_view(self):
        value = self.model_dump(mode='json')
        if not self.timeouts_enabled:
            value.update(remaining_seconds=None, research_seconds=None)
        return value


class Hypothesis(AgentNarrative):
    hypothesis_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    description: str
    rationale: str
    baseline: bool = False
    based_on: str | None = None
    based_on_hypothesis_id: str | None = None
    kind: Literal['single', 'family', 'ensemble'] = 'single'
    model_family: str | None = None
    family_tools: list[str] = Field(default_factory=list, max_length=3)


class HypothesisBatch(AgentNarrative):
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=2)


class PlannerPlan(AgentNarrative):
    """Agent-owned experiment choices; search-group membership is controller state."""
    priority: int = Field(100, ge=0, description='Lower values run first; the first baseline remains the initial control.')
    feature_set_id: str | None = None
    feature_set: FeatureSet | None = None
    contract_version: Literal[2] = 2
    target_transform: TargetTransform = Field(default_factory=TargetTransform)
    weight_policy: WeightPolicy = Field(default_factory=WeightPolicy)
    prediction_policy: PredictionPolicy = Field(default_factory=PredictionPolicy)
    intervention: Intervention = Field(default_factory=Intervention)
    validation_metadata: dict[str, Any] = Field(default_factory=dict)
    hypothesis: Hypothesis
    tool_id: str
    skill_ids: list[str]
    model_params: dict[str, Any] = Field(default_factory=dict)
    preprocessing: list[str] = Field(default_factory=list)
    strategy: Literal['single', 'optuna', 'native'] = 'single'
    search_space: dict[str, dict[str, Any]] = Field(default_factory=dict)
    hpo_trial_budget: int | None = Field(None, ge=1, description='Agent-selected initial budget for later HPO of this fixed baseline.')
    cost_tag: Literal['fast', 'medium', 'slow'] = 'fast'
    time_limit_seconds: float = Field(300, gt=0)
    cpus: int = Field(2, ge=1, strict=True)
    memory_gb: float = Field(4, gt=0, allow_inf_nan=False)
    max_trials: int = Field(20, ge=1, le=1000)
    patience_trials: int | None = Field(5, ge=1)
    min_delta: float = Field(0, ge=0, description='Legacy absolute comparison threshold; NOT the HPO continuation threshold. Batched search uses RunConfig.hpo_gain_threshold; keep min_delta=0 to retain every improvement.')
    prediction_kind: str = 'value'
    ensemble_components: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: Literal['custom'] = 'custom'
    stopping_policy: Literal['fixed','early_stopping'] = 'fixed'
    component_stopping_policies: dict[str, Literal['fixed','early_stopping']] = Field(default_factory=dict)
    base_config_hash: str | None = None
    change_kind: Literal['experiment','single_change_ablation'] = 'experiment'
    validation_protocol: dict[str, Any] = Field(default_factory=dict)
    parameter_routes: dict[str, str] = Field(default_factory=dict, description='For ensembles, map each flat HPO parameter to its fitted component name.')
    ensemble_sources: list[dict[str, Any]] = Field(default_factory=list, max_length=3)

    @model_validator(mode='before')
    @classmethod
    def normalize_plan_labels(cls, value):
        if not isinstance(value, dict):return value
        value = dict(value)
        label = value.get('change_kind')
        if isinstance(label, str) and label in {'baseline', 'model_change', 'hpo', 'single_configuration', 'ablation'}:
            value['change_kind'] = 'single_change_ablation' if label == 'ablation' else 'experiment'
            value['agent_notes'] = dict(value.get('agent_notes', {}), original_change_kind=label)
        if isinstance(value.get('preprocessing'), str):
            value['preprocessing'] = [value['preprocessing']]
        if isinstance(value.get('intervention'), str):
            value['intervention'] = {'operations': [value['intervention']]}
        return value


class HypothesisPlan(PlannerPlan):
    """Executable plan with a controller-derived view of its owning group."""
    search_group_id: str | None = None
    search_group_kind: Literal['single', 'family', 'ensemble'] = 'single'
    parent_hypothesis: Hypothesis | None = None


class SearchGroupPlan(StrictModel):
    hypothesis: Hypothesis
    kind: Literal['single', 'family', 'ensemble'] = 'single'
    variants: list[HypothesisPlan] = Field(min_length=1, max_length=4)
    skipped: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode='after')
    def valid_variants(self):
        limit = {'single': 1, 'family': 3, 'ensemble': 4}[self.kind]
        ids = [p.hypothesis.hypothesis_id for p in self.variants]
        if len(ids) > limit or len(ids) != len(set(ids)):
            raise ValueError('Invalid/duplicate variants in search group')
        # The containing group owns membership. Flat fields are a derived view for
        # execution/persistence, never independently authored agent decisions.
        self.variants = [p.model_copy(deep=True, update=dict(
            search_group_id=self.hypothesis.hypothesis_id if self.kind != 'single' else None,
            search_group_kind=self.kind,
            parent_hypothesis=self.hypothesis.model_copy(deep=True) if self.kind != 'single' else None,
        )) for p in self.variants]
        return self


class EnsembleProposal(AgentNarrative):
    component_experiment_ids: list[str] = Field(default_factory=list, max_length=3)
    rationale: str = Field(min_length=1,max_length=4000)
    skip: bool = False
    methods: list[Literal['mean', 'median', 'weighted', 'stack']] = Field(default_factory=lambda: ['mean'], min_length=1, max_length=4)
    oof_folds: int = Field(3, ge=2)
    weight_trials: int = Field(15, ge=1)

    @model_validator(mode='after')
    def distinct_components(self):
        if not self.rationale.strip():raise ValueError('An ensemble proposal/skip requires an explanation')
        if not self.skip and (len(self.component_experiment_ids) < 2 or
                              len(set(self.component_experiment_ids)) != len(self.component_experiment_ids)):
            raise ValueError('Select two or three distinct component experiments, or skip with a reason')
        return self


class PlanBatch(AgentNarrative):
    plans: list[PlannerPlan] = Field(min_length=1, max_length=2)

    @model_validator(mode='before')
    @classmethod
    def accept_single_plan(cls, value):
        if isinstance(value, dict) and 'plans' not in value and 'hypothesis' in value and 'tool_id' in value:
            value = {'plans': [value]}
        if isinstance(value, dict) and isinstance(value.get('plans'), list):
            # Old clients/cached responses may still include these obsolete hints.
            # They have no authority and must not turn a valid recipe into a refusal.
            runtime_fields = {'search_group_id', 'search_group_kind', 'parent_hypothesis'}
            plans = []
            for plan in value['plans']:
                if isinstance(plan, PlannerPlan):
                    plan = plan.model_dump(mode='python')
                if isinstance(plan, dict):
                    plan = {key: item for key, item in plan.items() if key not in runtime_fields}
                plans.append(plan)
            return dict(value, plans=plans)
        return value


class ValidationReport(AgentNarrative):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)


class CandidateVersion(StrictModel):
    agent_notes: dict[str, Any] = Field(default_factory=dict)
    feature_set_hash: str | None = None
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
    execution_report: dict[str, Any] | None = None
    execution_report_ref: str | None = None


class SplitSpec(StrictModel):
    split_id: str
    round_id: str
    dataset_hash: str
    seed: int
    strategy: str
    train_indices: list[int]
    holdout_indices: list[int]


class TrialResult(StrictModel):
    origin: Literal['trained', 'baseline'] = 'trained'
    selection_ref: str | None = None
    validation_scope: Literal['model', 'feature'] = 'model'
    number: int
    params: dict[str, Any]
    status: str
    score: float | None = None
    duration_seconds: float
    error: str | None = None

    kind: str = 'optuna'
    validation_protocol_ref: str | None = None
    fold_results: list[dict[str, Any]] = Field(default_factory=list)
    score_std: float | None = None
    oof_score: float | None = None
    oof_ref: str | None = None
    score_kind: str = 'unknown_legacy'
    oof_score_kind: str = 'inner_oof'
    outer_split_id: str | None = None
    metric_options: dict[str, Any] = Field(default_factory=dict)
    weighting: str = 'unweighted'


class ExperimentRun(StrictModel):
    validation_scope: Literal['model', 'feature'] = 'model'
    oof_score_kind: str = 'inner_oof'
    model_spec: dict[str, Any] | None = None
    intervention: Intervention | None = None
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
    validation_protocol_ref: str | None = None
    fold_results: list[dict[str, Any]] = Field(default_factory=list)
    score_std: float | None = None
    oof_score: float | None = None
    oof_ref: str | None = None
    full_refit: bool = False
    execution_report: dict[str, Any] | None = None
    execution_report_ref: str | None = None
    score_kind: str = 'unknown_legacy'


class EvaluationResult(StrictModel):
    experiment_id: str
    split_id: str
    metric: str
    direction: str
    score: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    score_kind: str = 'outer_holdout'
    dataset_hash: str | None = None
    metric_options: dict[str, Any] = Field(default_factory=dict)
    weighting: str = 'unweighted'
    validation_protocol_ref: str | None = None


class Champion(StrictModel):
    candidate: CandidateVersion
    plan: HypothesisPlan
    run: ExperimentRun
    evaluation: EvaluationResult

    @model_validator(mode='after')
    def evaluation_model_only(self):
        if self.run.role != 'evaluation':
            raise ValueError('A submission-only full-data refit cannot participate in Champion comparisons')
        return self


class SelectedPreviousResults(StrictModel):
    round_id: str
    selected: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    champion_id: str | None = None
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    search_groups: list[dict[str, Any]] = Field(default_factory=list)


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
    error_code: str | None = None
    field_errors: list[dict[str, Any]] = Field(default_factory=list)
    hint: str | None = None


class Observation(AgentNarrative):
    observation_id: str
    topic: str
    description: str
    rows: int | None = Field(None, ge=0, description='Measured support in source rows, not SQL result cardinality. Omit for aggregates without an explicit support count.')
    result_rows: int | None = Field(None, ge=0)
    numbers: dict[str, float] = Field(default_factory=dict)
    source_ref: str


class ResearchProposal(AgentNarrative):
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    expected_effect: str
    verification: str


class ReconnaissanceReport(AgentNarrative):
    status: Literal['completed', 'partial', 'unavailable'] = 'completed'
    observations: list[Observation] = Field(default_factory=list)
    proposals: list[ResearchProposal] = Field(default_factory=list)
    reason: str | None = None
    unfinished_checks: list[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def isolate_invalid_items(cls, value):
        if not isinstance(value,dict):return value
        value=dict(value)
        errors=[]
        for key,model in [('observations',Observation),('proposals',ResearchProposal)]:
            items=value.get(key,[])
            if not isinstance(items,list):continue
            valid=[]
            for index,item in enumerate(items):
                try:valid.append(model.model_validate(item))
                except ValueError as exc:errors.append(f'{key}[{index}] rejected: {str(exc)[:350]}')
            value[key]=valid
        if errors:
            value['status']='partial'
            value['unfinished_checks']=list(value.get('unfinished_checks',[]))+errors
        return value


class TaskDetail(AgentNarrative):
    meaning: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)


class RankedFinding(AgentNarrative):
    finding_id: str
    priority: Literal['critical', 'high', 'medium', 'low']
    rationale: str = Field(min_length=1)
    planner_action: str = Field(min_length=1)


class PlannerContextDraft(AgentNarrative):
    task_summary: str = Field(min_length=1)
    task_details: list[TaskDetail] = Field(default_factory=list)
    ranked_findings: list[RankedFinding] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ResultReview(AgentNarrative):
    implementation: Literal['matches', 'partial', 'mismatch', 'unknown']
    effect: Literal['supported', 'not_supported', 'inconclusive']
    evidence_refs: list[str] = Field(default_factory=list)
    explanation: str
    recommendations: list[str] = Field(default_factory=list)


class PreprocessingSpec(AgentNarrative):
    module: str = 'preprocessing.py'
    factory: str = 'create_preprocessor'
    description: str
    output: Literal['dataframe', 'dense', 'sparse']
    constraints: list[str] = Field(default_factory=list)


class PreprocessingBundle(AgentNarrative):
    source: str
    spec: PreprocessingSpec


class TrainingCheckpoint(StrictModel):
    checkpoint_id: str
    stage: Literal['initial_fit', 'batch', 'failure']
    plan: dict[str, Any]
    trials: list[dict[str, Any]] = Field(default_factory=list)
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_score: float | None = None
    remaining_seconds: float | None = Field(None, ge=0)
    timeouts_enabled: bool = True
    spent_seconds: float = Field(0, ge=0)
    trial_limit: int | None = Field(None, ge=1)
    remaining_trials: int = Field(ge=0)
    default_action: Literal['continue', 'stop', 'refit']
    error: str | None = None
    score_kind: str = 'unknown_legacy'
    outer_split_id: str | None = None
    validation_protocol_ref: str | None = None
    metric_options: dict[str, Any] = Field(default_factory=dict)
    weighting: str = 'unweighted'


class TrainingDecision(AgentNarrative):
    action: Literal['continue', 'stop', 'refit']
    reason: str
    additional_trials: int = Field(0, ge=0)
    recommendations: list[str] = Field(default_factory=list)
