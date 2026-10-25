# Graph Report - mlmom-project  (2026-10-25)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2166 nodes · 8230 edges · 105 communities (92 shown, 13 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 1070 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8b962e83`
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

## God Nodes (most connected - your core abstractions)
1. `RunConfig` - 169 edges
2. `Pipeline` - 167 edges
3. `DataInspector` - 103 edges
4. `case()` - 101 edges
5. `ExperimentStore` - 89 edges
6. `OfflineLLM` - 87 edges
7. `build_planning_graph()` - 85 edges
8. `atomic_json()` - 84 edges
9. `Skills` - 72 edges
10. `RoleClient` - 68 edges

## Surprising Connections (you probably didn't know these)
- `seed()` --uses--> `Champion`  [INFERRED]
  scripts/replay_cycle_from_artifacts.py → mlmom/pipeline_contracts.py
- `SubmissionIsolationTests` --uses--> `Champion`  [INFERRED]
  tests/test_cycle_full_refit.py → mlmom/pipeline_contracts.py
- `SessionAccountingTests` --uses--> `Champion`  [INFERRED]
  tests/test_feature_session.py → mlmom/pipeline_contracts.py
- `AtomicReplacementTests` --uses--> `FeatureProposal`  [INFERRED]
  tests/test_advisory_costs_and_replacement.py → mlmom/feature_contracts.py
- `FeaturePlanningTests` --uses--> `FeatureSession`  [INFERRED]
  tests/test_feature_session.py → mlmom/feature_contracts.py

## Import Cycles
- None detected.

## Communities (105 total, 13 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (8): CodingState, reuse_assignment(), Champion, RunState, capability_matrix(), inspect(), main(), names()

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (33): feature_configuration_keys(), feature_hash(), FeatureEvidence, FeatureExperimentPlan, FeatureProgram, FeatureProposal, FeatureSession, FeatureSet (+25 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (13): JsonSerializer, SQLiteCheckpointer, _unsupported(), SkillRegistry, check(), render(), RoleSkillTests, generate() (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (24): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+16 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (12): dumps(), plain(), DeferredTask, Pipeline, finished(), group_remaining(), prepare_alternatives(), process_jobs() (+4 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (8): LLMUnavailable, RoleClient, StructuredOutputError, CodeBundle, LLMSettings, AgentProtocolTests, ProviderFailureTests, InfrastructureRoutingTests

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (11): PolicyState, create(), load(), _load_policy(), PolicyCandidate, PreparedCandidate, Estimator, ModelContractV2Tests (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (24): SourcePatch, DatasetProfile, InferenceConfidence, MetricDirection, ResultFormat, Severity, SplitColumnStats, StrictModel (+16 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (25): TableReadOptions, csv_worker(), _lazy_table(), validate_sql(), diagnostics(), evaluate(), feature_profile(), detect_table_format() (+17 more)

### Community 9 - "Community 9"
Cohesion: 0.09
Nodes (11): installed_tools(), ResourceUnavailable, SandboxManager, synchronized(), AdapterTests, ExecutionTests, NativeValidationTests, CSVEngineTests (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (15): RunConfig, Budget, AdvisoryCostTests, generate(), FeatureClient, FeaturePipelineTests, generate(), CustomCandidatePipeline (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (11): tree_hash(), load_module(), train(), control(), imported_metadata(), search_checkpoint(), FeatureWorkerTests, InnerExecutionTests (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (10): classification(), LibraryCandidate, load_adapter(), preprocessor(), text_columns(), text_values(), EnsembleCandidate, load_ensemble() (+2 more)

### Community 13 - "Community 13"
Cohesion: 0.08
Nodes (15): default_space(), interrupt_search(), relative_gain(), run_search(), attempted(), persist(), remaining(), request_control() (+7 more)

### Community 14 - "Community 14"
Cohesion: 0.12
Nodes (11): BudgetSnapshot, SeparatePreprocessingTests, ClockBudget, Coder, CodingBudgetTests, Sandbox, Store, ShortCycleTests (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.10
Nodes (16): AgentNarrative, config_hash(), hpo_child(), PredictionPolicy, TargetTransform, WeightPolicy, HypothesisPlan, PlannerPlan (+8 more)

### Community 16 - "Community 16"
Cohesion: 0.08
Nodes (13): validate_plan(), MetricsAndSplitTests, PlanningLimitTests, generate(), generate(), Skills, PolicyConsistencyTests, generate() (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.13
Nodes (10): request_messages(), extract_json(), thinking_parameters(), traced_node(), trace_event(), bind(), current_sink(), emit() (+2 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (15): export_params(), _allocated_params(), execution_config(), is_thread_parameter(), model_parameters(), restore_thread_settings(), thread_budget_errors(), thread_settings() (+7 more)

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (10): save(), OfflineLLM, OfflinePipeline, NativeStarterClient, ScoutingTests, generate(), generate(), generate() (+2 more)

### Community 20 - "Community 20"
Cohesion: 0.11
Nodes (9): compact_planning_payload(), observation_index(), paged_evidence(), ToolResult, generate(), ResearchTools, ToolSet, PlanningRecoveryTests (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (8): FeatureAugmentingPreprocessor, FeatureTransformer, validate_features(), Base, definition(), FeatureDSLTests, fit_transform(), transform()

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (15): Hypothesis, SearchGroupPlan, collect_groups(), expand_families(), limit_parent_groups(), case(), RetryTests, planning_call() (+7 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (10): main(), package(), main(), main(), main(), archive_hash(), main(), main() (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.13
Nodes (12): ExperimentStore, component_pool(), PipelineFlowTests, first(), SubmissionIsolationTests, generate(), generate(), FeaturePlanningTests (+4 more)

### Community 25 - "Community 25"
Cohesion: 0.15
Nodes (11): evaluate(), process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), worker_record(), Job, watchdog() (+3 more)

### Community 26 - "Community 26"
Cohesion: 0.14
Nodes (10): HypothesisBatch, PlanBatch, ValidationReport, RepairingPlanningClient, native_source(), ReviewPromptPlanner, StarterPlanner, broken_code() (+2 more)

### Community 27 - "Community 27"
Cohesion: 0.08
Nodes (13): build_planning_graph(), hypotheses(), is_light(), persist(), owned_group(), bind_group(), initial_boosting_resources(), current_context() (+5 more)

### Community 28 - "Community 28"
Cohesion: 0.13
Nodes (17): stopping_metric(), inspect_model(), better(), checked_predictions(), metric_name(), prediction_kind(), score_predictions(), TrialResult (+9 more)

### Community 29 - "Community 29"
Cohesion: 0.13
Nodes (11): plan_ensemble(), finish(), read_component_history(), EnsembleProposal, component_brief(), ControllerGroupTests, graph_input(), GroupClient (+3 more)

### Community 30 - "Community 30"
Cohesion: 0.15
Nodes (14): TrainingCheckpoint, TrainingDecision, commit_mailbox(), decide(), fallback(), validate_decision(), worker_checkpoint(), ExecutablePlanTests (+6 more)

### Community 31 - "Community 31"
Cohesion: 0.11
Nodes (7): AtomicReplacementTests, run(), run(), SessionAccountingTests, run(), renamed(), run()

### Community 32 - "Community 32"
Cohesion: 0.11
Nodes (11): execution_contract(), plan_field_sources(), same_fixed_settings(), constant_space(), normalize_protocol(), for_hypothesis(), plan(), PlanningState (+3 more)

### Community 33 - "Community 33"
Cohesion: 0.16
Nodes (7): FileRole, SemanticType, TableProfile, build_llm_context(), DataInspector, DataInspectorTests, write_dictionary()

### Community 34 - "Community 34"
Cohesion: 0.14
Nodes (11): _json_value(), comparable_scores(), error_metrics(), relative_gain(), ResultReview, data_observations(), error_slices(), add() (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.13
Nodes (4): AssignmentClient, PreparationPool, PreparationConcurrencyTests, assigned()

### Community 36 - "Community 36"
Cohesion: 0.20
Nodes (4): ColumnRole, DatasetChecks, TaskSpec, _normalise_name()

### Community 37 - "Community 37"
Cohesion: 0.17
Nodes (9): prepare_initial_proposal(), SplitSpec, group_keys(), inner_protocol(), make_split(), split_indices(), validation_groups(), execution_case() (+1 more)

### Community 38 - "Community 38"
Cohesion: 0.16
Nodes (11): audit_csv(), audit_cycles(), external_stop(), main(), now(), run_case(), save_report(), stop_pipeline() (+3 more)

### Community 39 - "Community 39"
Cohesion: 0.22
Nodes (7): validator_repair(), CodePatch, Workspace, apply_patch(), read_file(), main(), request()

### Community 40 - "Community 40"
Cohesion: 0.22
Nodes (5): DistributionBucket, _display_value(), _finite_float(), _format_number(), _rate()

### Community 41 - "Community 41"
Cohesion: 0.29
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 42 - "Community 42"
Cohesion: 0.19
Nodes (6): DataIssue, ContextAgentTests, generate(), generate(), generate(), generate()

### Community 43 - "Community 43"
Cohesion: 0.18
Nodes (6): cases(), evidence(), gate(), main(), package_info(), ReleaseGateTests

### Community 44 - "Community 44"
Cohesion: 0.17
Nodes (9): GlobalAPIError, role_window(), finding_catalog(), prepare_planner_context(), previous_brief_summary(), PlannerContextDraft, profile_summary(), WorkerInfrastructureError (+1 more)

### Community 45 - "Community 45"
Cohesion: 0.20
Nodes (9): atomic_json(), main(), rebase(), report(), rows(), seed(), copy_reference(), strings() (+1 more)

### Community 46 - "Community 46"
Cohesion: 0.20
Nodes (10): case_config(), digest(), main(), select_real_settings(), verify_observation_handoff(), verify_run(), write(), create_task() (+2 more)

### Community 48 - "Community 48"
Cohesion: 0.17
Nodes (4): runtime_versions(), worker_runtime(), WorkerRuntimeError, WorkerRuntimeTests

### Community 50 - "Community 50"
Cohesion: 0.19
Nodes (3): Candidate, create_candidate(), load_candidate()

### Community 52 - "Community 52"
Cohesion: 0.35
Nodes (9): ColumnProfile, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent(), _render() (+1 more)

### Community 53 - "Community 53"
Cohesion: 0.28
Nodes (3): size(), _plain(), TraceSink

### Community 54 - "Community 54"
Cohesion: 0.32
Nodes (7): RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 55 - "Community 55"
Cohesion: 0.18
Nodes (3): Candidate, _execute_module(), Preprocessor

### Community 56 - "Community 56"
Cohesion: 0.33
Nodes (3): NetworkCircuit, NetworkPolicyTests, handler()

### Community 57 - "Community 57"
Cohesion: 0.33
Nodes (3): snapshot_cycle_submission(), validate_submission_file(), CycleSubmissionTests

### Community 58 - "Community 58"
Cohesion: 0.27
Nodes (4): compact_diagnostics(), champion_snapshot(), evaluation_summary(), plan_summary()

### Community 59 - "Community 59"
Cohesion: 0.40
Nodes (5): capacity(), terminate_tree(), index(), main(), validate()

### Community 62 - "Community 62"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 63 - "Community 63"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 64 - "Community 64"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 65 - "Community 65"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

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

### Community 97 - "Community 97"
Cohesion: 0.70
Nodes (5): main(), fitted(), predict(), sanitized(), worker()

### Community 99 - "Community 99"
Cohesion: 0.50
Nodes (3): generate(), __init__(), generate()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 633 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 14`, `Community 15`, `Community 19`, `Community 20`, `Community 22`, `Community 23`, `Community 24`, `Community 25`, `Community 26`, `Community 28`, `Community 29`, `Community 30`, `Community 32`, `Community 35`, `Community 37`, `Community 38`, `Community 42`, `Community 44`, `Community 45`, `Community 46`, `Community 48`, `Community 50`, `Community 51`, `Community 53`, `Community 95`, `Community 98`, `Community 102`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 45 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.08942161339421613 - nodes in this community are weakly interconnected._
- **Why does `RunConfig` connect `Community 10` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 8`, `Community 9`, `Community 11`, `Community 12`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 19`, `Community 20`, `Community 22`, `Community 23`, `Community 24`, `Community 25`, `Community 26`, `Community 27`, `Community 28`, `Community 29`, `Community 30`, `Community 31`, `Community 32`, `Community 34`, `Community 38`, `Community 42`, `Community 43`, `Community 45`, `Community 46`, `Community 48`, `Community 51`, `Community 54`, `Community 59`, `Community 97`, `Community 98`, `Community 100`, `Community 102`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 63 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 63 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.07033248081841433 - nodes in this community are weakly interconnected._