# Graph Report - mlmom-project  (2026-10-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2077 nodes · 7750 edges · 110 communities (94 shown, 16 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 963 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a01caf4d`
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
- Community 68
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

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 151 edges
2. `RunConfig` - 142 edges
3. `DataInspector` - 103 edges
4. `case()` - 81 edges
5. `OfflineLLM` - 78 edges
6. `ExperimentStore` - 76 edges
7. `atomic_json()` - 76 edges
8. `build_planning_graph()` - 72 edges
9. `SandboxManager` - 60 edges
10. `RoleClient` - 60 edges

## Surprising Connections (you probably didn't know these)
- `SchedulerIntegrationTests` --uses--> `Job`  [INFERRED]
  tests/test_scheduler_integration.py → mlmom/resources.py
- `InfrastructureRoutingTests` --uses--> `Job`  [INFERRED]
  tests/test_worker_supervision.py → mlmom/resources.py
- `SchedulerIntegrationTests` --uses--> `ResourceUnavailable`  [INFERRED]
  tests/test_scheduler_integration.py → mlmom/resources.py
- `InvariantTests` --uses--> `SandboxManager`  [INFERRED]
  tests/test_invariants.py → mlmom/resources.py
- `NativeValidationTests` --uses--> `SandboxManager`  [INFERRED]
  tests/test_native_validation.py → mlmom/resources.py

## Import Cycles
- None detected.

## Communities (110 total, 16 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (17): process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), Job, ResourceUnavailable, SandboxManager, trace_event() (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (13): JsonSerializer, SQLiteCheckpointer, _unsupported(), SkillRegistry, check(), render(), RoleSkillTests, generate() (+5 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (18): role_window(), finding_catalog(), prepare_planner_context(), previous_brief_summary(), TableReadOptions, csv_worker(), _json_value(), _lazy_table() (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (27): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+19 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (16): ExperimentStore, RunConfig, CheckpointIntegrationTests, ContextAndPatchTests, SubmissionIsolationTests, FeaturePipelineTests, InitialBoostingIntegrationTests, CustomCandidatePipeline (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (14): FeatureSpec, FeatureAugmentingPreprocessor, FeatureTransformer, screen_features(), validate_features(), dictionary_metadata(), fixed_model_hash(), paired_evidence() (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (10): FileRole, InferenceConfidence, SemanticType, TableProfile, TaskType, DataInspector, InputLayout, _resolve_path() (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (3): Champion, MLToolSpec, capability_matrix()

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (24): audit_fit(), compare_roundtrip(), complete_export(), config_hash(), _equal(), _leaves(), _missing(), require_admitted() (+16 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (10): classification(), LibraryCandidate, load_adapter(), text_columns(), EnsembleCandidate, load_ensemble(), FitContext, smoke() (+2 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (12): MetricDirection, ResultFormat, TaskDefinition, TaskValidationFinding, TaskValidationReport, ValidationStrategy, _normalise_metric(), TaskValidator (+4 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (10): main(), package(), main(), main(), main(), report(), main(), main() (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (9): GeneratedCandidateTests, generate(), case(), PlanningLimitTests, generate(), Skills, generate(), generate() (+1 more)

### Community 13 - "Community 13"
Cohesion: 0.10
Nodes (11): GlobalAPIError, LLMUnavailable, StructuredOutputError, CodingState, extract_json(), traced_node(), bind(), current_sink() (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.12
Nodes (6): RoleClient, LLMSettings, AgentProtocolTests, ProviderFailureTests, OptionalTimeoutTests, unavailable()

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (8): capture(), PolicyState, create(), load(), _load_policy(), PolicyCandidate, Estimator, ModelContractV2Tests

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (18): plan_ensemble(), finish(), read_component_history(), frozen_plan(), EnsembleProposal, Hypothesis, SearchGroupPlan, collect_groups() (+10 more)

### Community 17 - "Community 17"
Cohesion: 0.14
Nodes (15): stopping_metric(), preprocessor(), text_values(), inspect_model(), checked_predictions(), diagnostics(), error_metrics(), evaluate() (+7 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (14): better(), TrainingCheckpoint, TrialResult, runtime_versions(), fallback(), worker_checkpoint(), fit_context(), train() (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (13): relative_gain(), run_search(), apply_decision(), persist(), request_control(), TrainingDecision, TrainingControlTests, batch() (+5 more)

### Community 20 - "Community 20"
Cohesion: 0.15
Nodes (15): features(), read_table(), file_hash(), CandidateVersion, ExperimentRun, SubmissionArtifact, publish_submission(), snapshot_cycle_submission() (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.14
Nodes (16): atomic_json(), Budget, capacity(), terminate_tree(), main(), audit_csv(), audit_cycles(), main() (+8 more)

### Community 22 - "Community 22"
Cohesion: 0.15
Nodes (6): ColumnProfile, ColumnRole, DataIssue, DatasetChecks, TaskSpec, _normalise_name()

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (12): FeatureSet, PredictionPolicy, TargetTransform, WeightPolicy, HypothesisPlan, infrastructure_overrides(), normalize_champion_assignment(), categorical_probe() (+4 more)

### Community 24 - "Community 24"
Cohesion: 0.17
Nodes (12): feature_profile(), targets(), InputFiles, TaskContext, group_keys(), inner_protocol(), make_split(), split_indices() (+4 more)

### Community 25 - "Community 25"
Cohesion: 0.11
Nodes (18): constant_space(), normalize_protocol(), normalize_cpu_hints(), build_planning_graph(), for_hypothesis(), generate(), hypotheses(), is_light() (+10 more)

### Community 26 - "Community 26"
Cohesion: 0.13
Nodes (7): metric_capabilities(), Observation, ReconnaissanceReport, add(), ReconnaissanceSession, collect(), EvidenceContractTests

### Community 27 - "Community 27"
Cohesion: 0.14
Nodes (7): OfflineLLM, OfflinePipeline, ContextAgentTests, generate(), generate(), generate(), SchedulerIntegrationTests

### Community 28 - "Community 28"
Cohesion: 0.17
Nodes (5): dumps(), plain(), SelectedPreviousResults, finished(), emit()

### Community 29 - "Community 29"
Cohesion: 0.21
Nodes (7): validator_repair(), CodePatch, Workspace, apply_patch(), read_file(), main(), request()

### Community 30 - "Community 30"
Cohesion: 0.19
Nodes (14): DatasetProfile, Severity, build_llm_context(), _format_number(), main(), _write_json(), build_llm_markdown(), _distribution_text() (+6 more)

### Community 31 - "Community 31"
Cohesion: 0.11
Nodes (4): PreparedCandidate, Base, MeanPreprocessor, Stack

### Community 32 - "Community 32"
Cohesion: 0.16
Nodes (5): inspect(), main(), archive_hash(), main(), names()

### Community 33 - "Community 33"
Cohesion: 0.26
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (7): tree_hash(), _execute_module(), load_module(), predict(), write_predictions(), composite_case(), CompositeTests

### Community 35 - "Community 35"
Cohesion: 0.13
Nodes (7): PlanBatch, RepairingPlanningClient, StarterPlanner, Planner, generate(), generate(), generate()

### Community 36 - "Community 36"
Cohesion: 0.19
Nodes (5): ToolResult, ResearchTools, ToolSet, PlanningRecoveryTests, read_skills()

### Community 37 - "Community 37"
Cohesion: 0.21
Nodes (3): DeferredTask, group_remaining(), process_jobs()

### Community 38 - "Community 38"
Cohesion: 0.16
Nodes (8): CodeBundle, HypothesisBatch, ValidationReport, ControlledLLM, broken_code(), RepairLLM, PlanClient, AlternativeFailure

### Community 39 - "Community 39"
Cohesion: 0.17
Nodes (14): FeatureEvidence, FeatureSession, complete_feature_session(), feature_session_key(), feature_task_view(), run_feature_session(), evaluate(), save() (+6 more)

### Community 40 - "Community 40"
Cohesion: 0.14
Nodes (8): SourcePatch, SplitColumnStats, StrictModel, TaskAnalysisResult, __getattr__(), PreprocessingSpec, ResearchProposal, SkillManifest

### Community 41 - "Community 41"
Cohesion: 0.19
Nodes (3): FidelityError, Candidate, LightGBMFitEvidenceTests

### Community 42 - "Community 42"
Cohesion: 0.18
Nodes (5): request_messages(), NetworkCircuit, thinking_parameters(), NetworkPolicyTests, handler()

### Community 43 - "Community 43"
Cohesion: 0.20
Nodes (9): AgentNarrative, reuse_assignment(), _validate(), FeatureExperimentPlan, FeatureProposal, PlannerContextDraft, RankedFinding, TaskDetail (+1 more)

### Community 44 - "Community 44"
Cohesion: 0.17
Nodes (7): comparable_scores(), relative_gain(), ResultReview, review_result(), failure_result(), ResearchTests, MetricsAndSplitTests

### Community 45 - "Community 45"
Cohesion: 0.16
Nodes (7): select_feature_baseline(), EvaluationResult, SplitSpec, SessionAccountingTests, run(), renamed(), run()

### Community 46 - "Community 46"
Cohesion: 0.20
Nodes (5): ScoutingTests, generate(), generate(), generate(), generate()

### Community 48 - "Community 48"
Cohesion: 0.29
Nodes (4): DistributionBucket, _display_value(), _finite_float(), _rate()

### Community 49 - "Community 49"
Cohesion: 0.20
Nodes (6): cases(), evidence(), gate(), main(), package_info(), ReleaseGateTests

### Community 50 - "Community 50"
Cohesion: 0.19
Nodes (10): fingerprint(), install(), importing(), patch_loaded(), visit(), _patch(), captured(), _patch_lightgbm_updates() (+2 more)

### Community 52 - "Community 52"
Cohesion: 0.23
Nodes (10): inside(), resolve_inputs(), find(), case_config(), digest(), main(), select_real_settings(), verify_observation_handoff() (+2 more)

### Community 53 - "Community 53"
Cohesion: 0.20
Nodes (6): PlanningRejections, generate(), RetryTests, planning_call(), generate(), generate()

### Community 55 - "Community 55"
Cohesion: 0.19
Nodes (5): compact_planning_payload(), observation_index(), paged_evidence(), hpo_child(), Intervention

### Community 56 - "Community 56"
Cohesion: 0.28
Nodes (3): size(), _plain(), TraceSink

### Community 58 - "Community 58"
Cohesion: 0.32
Nodes (7): RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 59 - "Community 59"
Cohesion: 0.29
Nodes (5): BudgetSnapshot, installed_tools(), InvariantTests, NativeValidationTests, tools()

### Community 60 - "Community 60"
Cohesion: 0.25
Nodes (3): commit_mailbox(), external_stop(), main()

### Community 62 - "Community 62"
Cohesion: 0.27
Nodes (5): compact_diagnostics(), champion_snapshot(), evaluation_summary(), experiment_summary(), plan_summary()

### Community 64 - "Community 64"
Cohesion: 0.22
Nodes (3): FeatureClient, native_source(), NativeStarterClient

### Community 66 - "Community 66"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 67 - "Community 67"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 68 - "Community 68"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 69 - "Community 69"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 70 - "Community 70"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

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

### Community 101 - "Community 101"
Cohesion: 0.38
Nodes (4): _component_objects(), fitted_components(), inspect_component(), _json_params()

### Community 106 - "Community 106"
Cohesion: 0.60
Nodes (3): current_context(), generate(), load()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 608 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataInspector` connect `Community 6` to `Community 2`, `Community 36`, `Community 4`, `Community 7`, `Community 40`, `Community 10`, `Community 13`, `Community 46`, `Community 48`, `Community 18`, `Community 22`, `Community 24`, `Community 27`, `Community 30`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 58 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 58 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06331976481230213 - nodes in this community are weakly interconnected._
- **Why does `Pipeline` connect `Community 57` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 13`, `Community 14`, `Community 16`, `Community 17`, `Community 18`, `Community 19`, `Community 20`, `Community 21`, `Community 23`, `Community 24`, `Community 26`, `Community 27`, `Community 28`, `Community 30`, `Community 36`, `Community 37`, `Community 38`, `Community 39`, `Community 43`, `Community 44`, `Community 45`, `Community 46`, `Community 53`, `Community 56`, `Community 59`, `Community 60`, `Community 62`, `Community 63`, `Community 64`, `Community 65`, `Community 99`, `Community 100`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 37 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05706214689265537 - nodes in this community are weakly interconnected._