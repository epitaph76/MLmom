# Graph Report - mlmom-project  (2026-10-26)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2387 nodes · 9247 edges · 123 communities (101 shown, 22 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 1155 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5f8a63f1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 99
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104
- Community 105
- Community 106
- Community 107
- Community 108
- Community 109
- Community 110
- Community 111
- Community 112
- Community 113
- Community 114
- Community 115
- Community 116
- Community 117
- Community 118
- Community 119
- Community 120
- Community 121
- Community 122

## God Nodes (most connected - your core abstractions)
1. `RunConfig` - 181 edges
2. `Pipeline` - 170 edges
3. `atomic_json()` - 110 edges
4. `case()` - 104 edges
5. `DataInspector` - 103 edges
6. `ExperimentStore` - 98 edges
7. `OfflineLLM` - 94 edges
8. `build_planning_graph()` - 93 edges
9. `train()` - 74 edges
10. `Skills` - 72 edges

## Surprising Connections (you probably didn't know these)
- `SeparatePreprocessingTests` --uses--> `FitContext`  [INFERRED]
  tests/test_agent_roles.py → mlmom/sdk.py
- `ExecutionTests` --uses--> `FitContext`  [INFERRED]
  tests/test_execution.py → mlmom/sdk.py
- `FitFactsTests` --uses--> `FitContext`  [INFERRED]
  tests/test_execution_facts.py → mlmom/sdk.py
- `FeatureDSLTests` --uses--> `FitContext`  [INFERRED]
  tests/test_feature_engineering.py → mlmom/sdk.py
- `FeatureProgramTests` --uses--> `FitContext`  [INFERRED]
  tests/test_feature_programs.py → mlmom/sdk.py

## Import Cycles
- None detected.

## Communities (123 total, 22 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (33): oof_snapshot(), better(), export_params(), configuration_diff(), read_fit_facts(), record_fit(), execution_config(), restore_thread_settings() (+25 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (16): config_hash(), constant_space(), hpo_child(), Intervention, PolicyState, repair_allowed(), FitContext, create() (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (11): dumps(), plain(), SelectedPreviousResults, Pipeline, finished(), group_remaining(), prepare_alternatives(), process_jobs() (+3 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (22): GlobalAPIError, role_window(), feature_configuration_keys(), FeatureEvidence, FeatureExperimentPlan, FeatureProposal, FeatureSession, FeatureSet (+14 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (16): ExperimentStore, component_pool(), PipelineFlowTests, CheckpointIntegrationTests, SubmissionIsolationTests, generate(), generate(), FeaturePipelineTests (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (12): classification(), LibraryCandidate, stopping_metric(), load_adapter(), text_columns(), text_values(), EnsembleCandidate, load_ensemble() (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (25): compact_diagnostics(), comparable_scores(), relative_gain(), FitExecutionFacts, observed_execution(), related_fit_observations(), champion_snapshot(), evaluation_summary() (+17 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (8): ColumnRole, FileRole, SemanticType, TableProfile, DataInspector, InputLayout, _normalise_name(), _resolve_path()

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (22): atomic_json(), file_hash(), CandidateVersion, ExperimentRun, SubmissionArtifact, publish_submission(), snapshot_cycle_submission(), validate_submission_file() (+14 more)

### Community 10 - "Community 10"
Cohesion: 0.10
Nodes (17): inspect_model(), checked_predictions(), diagnostics(), error_metrics(), evaluate(), prediction_kind(), score_predictions(), error_slices() (+9 more)

### Community 11 - "Community 11"
Cohesion: 0.09
Nodes (18): EnsembleProposal, Hypothesis, HypothesisPlan, SearchGroupPlan, owned_group(), PlanningState, bind_group(), collect_groups() (+10 more)

### Community 12 - "Community 12"
Cohesion: 0.10
Nodes (14): InferenceConfidence, MetricDirection, ResultFormat, TaskAnalysisResult, TaskDefinition, TaskType, TaskValidationFinding, TaskValidationReport (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.09
Nodes (11): validate_plan(), InitialPlanningTests, generate(), ReviewPromptPlanner, case(), PlanningLimitTests, Skills, generate() (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.10
Nodes (13): FeatureProgram, FeatureSpec, FeatureAugmentingPreprocessor, FeatureTransformer, validate_features(), remember(), measured(), Base (+5 more)

### Community 15 - "Community 15"
Cohesion: 0.10
Nodes (10): SkillRegistry, check(), render(), RoleSkillTests, generate(), generate(), DecisionSkillCardTests, ResultClient (+2 more)

### Community 16 - "Community 16"
Cohesion: 0.10
Nodes (19): TableReadOptions, select_feature_baseline(), inside(), resolve_inputs(), find(), targets(), validate_tables(), Champion (+11 more)

### Community 17 - "Community 17"
Cohesion: 0.09
Nodes (14): save(), OfflineLLM, OfflinePipeline, ControlledLLM, FeatureClient, NativeStarterClient, RepairLLM, PipelineIntegrationTests (+6 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (14): AdvisoryCostTests, SeparatePreprocessingTests, ClockBudget, Coder, CodingBudgetTests, Sandbox, Store, MethodCardDeliveryTests (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.11
Nodes (9): CodeBundle, extract_json(), traced_node(), bind(), current_sink(), span(), broken_code(), generate() (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.10
Nodes (14): finding_catalog(), prepare_planner_context(), compact_planning_payload(), observation_index(), paged_evidence(), previous_brief_summary(), stored_evidence(), PlannerContextDraft (+6 more)

### Community 21 - "Community 21"
Cohesion: 0.13
Nodes (12): HypothesisBatch, PlanBatch, ValidationReport, validator(), generate(), ContextAndPatchTests, generate(), RepairingPlanningClient (+4 more)

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (5): RoleClient, LLMSettings, AgentProtocolTests, ProviderFailureTests, NarrativeTests

### Community 23 - "Community 23"
Cohesion: 0.14
Nodes (13): execution_context(), TrainingCheckpoint, TrainingDecision, commit_mailbox(), decide(), fallback(), validate_decision(), worker_checkpoint() (+5 more)

### Community 24 - "Community 24"
Cohesion: 0.09
Nodes (10): fixed_model_hash(), paired_evidence(), feature_context(), FitFactsTests, FeatureWorkerTests, run(), SessionAccountingTests, run() (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.15
Nodes (9): PredictionPolicy, TargetTransform, WeightPolicy, PlannerPlan, main(), main(), main(), main() (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.15
Nodes (14): ColumnProfile, DatasetProfile, Severity, build_llm_context(), build_llm_markdown(), _distribution_text(), _escape(), _number() (+6 more)

### Community 27 - "Community 27"
Cohesion: 0.13
Nodes (9): RunConfig, Budget, ContinuousRunTests, finalize(), OptionalTimeoutTests, evaluation(), RotatingHoldoutTests, GroupPipelineTests (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (6): installed_tools(), SandboxManager, AdapterTests, ExecutionTests, NativeValidationTests, tools()

### Community 29 - "Community 29"
Cohesion: 0.14
Nodes (5): load_module(), InnerExecutionTests, composite_case(), CompositeTests, TargetEncodingBoundaryTests

### Community 30 - "Community 30"
Cohesion: 0.19
Nodes (9): _compatible(), component_availability(), component_refit_cost(), _correlation(), diversity_evidence(), _errors(), _gain(), scores() (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.11
Nodes (5): preprocessor(), PreparedCandidate, Base, MeanPreprocessor, Stack

### Community 32 - "Community 32"
Cohesion: 0.12
Nodes (3): JsonSerializer, SQLiteCheckpointer, _unsupported()

### Community 33 - "Community 33"
Cohesion: 0.12
Nodes (6): capacity(), ResourceUnavailable, runtime_versions(), trace_event(), external_stop(), main()

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (13): _load(), protocol_digest(), case_config(), digest(), main(), select_real_settings(), verify_observation_handoff(), verify_run() (+5 more)

### Community 35 - "Community 35"
Cohesion: 0.21
Nodes (7): validator_repair(), Workspace, apply_patch(), read_file(), main(), request(), ComponentTests

### Community 36 - "Community 36"
Cohesion: 0.17
Nodes (6): ToolResult, ResearchTools, ToolSet, read_evidence(), PlanningRecoveryTests, read_skills()

### Community 37 - "Community 37"
Cohesion: 0.19
Nodes (22): build_coding_graph(), check_preprocessing(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing() (+14 more)

### Community 38 - "Community 38"
Cohesion: 0.22
Nodes (7): DistributionBucket, _display_value(), _finite_float(), _format_number(), main(), _rate(), _write_json()

### Community 39 - "Community 39"
Cohesion: 0.26
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 40 - "Community 40"
Cohesion: 0.14
Nodes (11): CodePatch, AgentNarrative, Observation, PreprocessingSpec, RankedFinding, ResearchProposal, ResultReview, TaskDetail (+3 more)

### Community 41 - "Community 41"
Cohesion: 0.15
Nodes (9): StructuredOutputError, plan_ensemble(), compare_oof_components(), measure_components(), read_component_history(), IndependentRecipeTests, builder(), generate() (+1 more)

### Community 42 - "Community 42"
Cohesion: 0.14
Nodes (9): csv_worker(), _json_value(), _lazy_table(), validate_sql(), detect_table_format(), read_table(), predict(), PolarsSQLSkillTests (+1 more)

### Community 43 - "Community 43"
Cohesion: 0.13
Nodes (10): LLMUnavailable, generate(), generate(), RealFailureRegressions, generate(), generate(), generate(), generate() (+2 more)

### Community 44 - "Community 44"
Cohesion: 0.17
Nodes (5): request_messages(), NetworkCircuit, thinking_parameters(), NetworkPolicyTests, handler()

### Community 45 - "Community 45"
Cohesion: 0.19
Nodes (5): eda_matrix(), eda_plan(), pending_eda(), metric_capabilities(), ReconnaissanceSession

### Community 46 - "Community 46"
Cohesion: 0.18
Nodes (11): execution_contract(), plan_field_sources(), normalize_protocol(), build_planning_graph(), for_hypothesis(), generate(), hypotheses(), is_light() (+3 more)

### Community 47 - "Community 47"
Cohesion: 0.26
Nodes (7): evaluate(), process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), worker_record(), ProcessIdentityTests

### Community 48 - "Community 48"
Cohesion: 0.14
Nodes (4): default_space(), interrupt_search(), relative_gain(), PersistentSearchTests

### Community 49 - "Community 49"
Cohesion: 0.19
Nodes (10): run_search(), attempted(), persist(), remaining(), request_control(), room(), SamplerOwnershipTests, control() (+2 more)

### Community 50 - "Community 50"
Cohesion: 0.17
Nodes (7): _allocated_params(), is_thread_parameter(), model_parameters(), thread_budget_errors(), thread_settings(), same_fixed_settings(), normalize_cpu_hints()

### Community 51 - "Community 51"
Cohesion: 0.18
Nodes (11): feature_profile(), features(), AttemptLimit, check_generated(), feature(), preprocessing(), generate(), reference() (+3 more)

### Community 52 - "Community 52"
Cohesion: 0.21
Nodes (9): audit_csv(), audit_cycles(), external_stop(), main(), now(), run_case(), save_report(), stop_pipeline() (+1 more)

### Community 53 - "Community 53"
Cohesion: 0.19
Nodes (6): cases(), evidence(), gate(), main(), package_info(), ReleaseGateTests

### Community 54 - "Community 54"
Cohesion: 0.18
Nodes (7): finish(), PlanningRejections, generate(), RetryTests, planning_call(), generate(), generate()

### Community 55 - "Community 55"
Cohesion: 0.18
Nodes (8): package(), index(), main(), validate(), main(), archive_hash(), main(), FlatPackageTests

### Community 57 - "Community 57"
Cohesion: 0.17
Nodes (4): BudgetSnapshot, InvariantTests, AlternativeFailure, SchedulerIntegrationTests

### Community 58 - "Community 58"
Cohesion: 0.19
Nodes (6): Job, terminate_tree(), watchdog(), budget(), launch(), WatchdogTests

### Community 59 - "Community 59"
Cohesion: 0.18
Nodes (8): SourcePatch, DataIssue, DatasetChecks, SplitColumnStats, StrictModel, TaskSpec, MLToolSpec, SkillManifest

### Community 61 - "Community 61"
Cohesion: 0.28
Nodes (5): size(), _plain(), TraceSink, probe(), main()

### Community 62 - "Community 62"
Cohesion: 0.19
Nodes (3): ReconnaissanceReport, EvidenceContractTests, EDAMethodologyTests

### Community 64 - "Community 64"
Cohesion: 0.18
Nodes (7): require_worker_infrastructure(), WorkerInfrastructureError, WorkerRuntimeError, InfrastructureRoutingTests, generate(), unavailable(), broken()

### Community 65 - "Community 65"
Cohesion: 0.27
Nodes (7): first(), handler(), handler(), handler(), handler(), response(), tool_call()

### Community 67 - "Community 67"
Cohesion: 0.25
Nodes (4): canonical_metric(), metric_catalog(), metric_errors(), normalise_metric()

### Community 70 - "Community 70"
Cohesion: 0.32
Nodes (3): inspect(), main(), names()

### Community 71 - "Community 71"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 72 - "Community 72"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 73 - "Community 73"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 74 - "Community 74"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 75 - "Community 75"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 76 - "Community 76"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 77 - "Community 77"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 78 - "Community 78"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 79 - "Community 79"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 80 - "Community 80"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 81 - "Community 81"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 82 - "Community 82"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 83 - "Community 83"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 84 - "Community 84"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 85 - "Community 85"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 86 - "Community 86"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 87 - "Community 87"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 88 - "Community 88"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 89 - "Community 89"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 90 - "Community 90"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 91 - "Community 91"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 92 - "Community 92"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 93 - "Community 93"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 94 - "Community 94"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 95 - "Community 95"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 96 - "Community 96"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 97 - "Community 97"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 98 - "Community 98"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 99 - "Community 99"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 100 - "Community 100"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 101 - "Community 101"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 102 - "Community 102"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 103 - "Community 103"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 104 - "Community 104"
Cohesion: 0.39
Nodes (3): EntityDiscoverySkillTests, pair_truth(), run_query()

### Community 108 - "Community 108"
Cohesion: 0.29
Nodes (4): batch(), initial(), objective(), control()

### Community 113 - "Community 113"
Cohesion: 0.60
Nodes (3): current_context(), generate(), validate_selection()

### Community 116 - "Community 116"
Cohesion: 0.50
Nodes (3): generate(), __init__(), generate()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 684 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 2` to `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 19`, `Community 20`, `Community 21`, `Community 22`, `Community 23`, `Community 25`, `Community 26`, `Community 27`, `Community 28`, `Community 31`, `Community 32`, `Community 33`, `Community 34`, `Community 36`, `Community 40`, `Community 43`, `Community 45`, `Community 47`, `Community 54`, `Community 56`, `Community 57`, `Community 61`, `Community 62`, `Community 64`, `Community 106`, `Community 109`, `Community 112`, `Community 121`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 47 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07511737089201878 - nodes in this community are weakly interconnected._
- **Why does `RunConfig` connect `Community 27` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 13`, `Community 16`, `Community 17`, `Community 18`, `Community 19`, `Community 20`, `Community 21`, `Community 23`, `Community 24`, `Community 25`, `Community 28`, `Community 33`, `Community 34`, `Community 36`, `Community 40`, `Community 41`, `Community 43`, `Community 47`, `Community 51`, `Community 53`, `Community 54`, `Community 55`, `Community 57`, `Community 59`, `Community 61`, `Community 62`, `Community 64`, `Community 68`, `Community 105`, `Community 112`, `Community 121`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05144230769230769 - nodes in this community are weakly interconnected._