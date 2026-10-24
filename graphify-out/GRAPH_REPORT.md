# Graph Report - mlmom-project  (2026-10-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1997 nodes · 7423 edges · 108 communities (90 shown, 18 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 933 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `78f54091`
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

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 151 edges
2. `RunConfig` - 146 edges
3. `DataInspector` - 103 edges
4. `case()` - 87 edges
5. `OfflineLLM` - 82 edges
6. `ExperimentStore` - 77 edges
7. `build_planning_graph()` - 77 edges
8. `atomic_json()` - 71 edges
9. `Skills` - 63 edges
10. `SandboxManager` - 60 edges

## Surprising Connections (you probably didn't know these)
- `OfflineLLM` --uses--> `CodeBundle`  [INFERRED]
  tests/fakes.py → mlmom/coding.py
- `AgentProtocolTests` --uses--> `CodeBundle`  [INFERRED]
  tests/test_agents.py → mlmom/coding.py
- `RepairLLM` --uses--> `CodeBundle`  [INFERRED]
  tests/test_invariants.py → mlmom/coding.py
- `MinimalChecksTests` --uses--> `CodeBundle`  [INFERRED]
  tests/test_minimal_checks.py → mlmom/coding.py
- `Coder` --uses--> `CodeBundle`  [INFERRED]
  tests/test_planning_limits.py → mlmom/coding.py

## Import Cycles
- None detected.

## Communities (108 total, 18 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (8): CodeBundle, inspect_model(), diagnostics(), evaluate(), metric_name(), prediction_kind(), score_predictions(), broken_code()

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (19): evaluate(), process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), worker_record(), Job, ResourceUnavailable (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (23): GlobalAPIError, role_window(), finding_catalog(), prepare_planner_context(), previous_brief_summary(), comparable_scores(), relative_gain(), extract_json() (+15 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (12): PolicyState, create(), load(), _load_policy(), PolicyCandidate, PreparedCandidate, Estimator, ModelContractV2Tests (+4 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (17): ExperimentStore, RunConfig, component_pool(), CheckpointIntegrationTests, SubmissionIsolationTests, FeaturePipelineTests, InitialBoostingIntegrationTests, MinimalChecksTests (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (10): dumps(), plain(), SelectedPreviousResults, DeferredTask, Pipeline, finished(), group_remaining(), process_jobs() (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (28): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+20 more)

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (17): CodePatch, compact_planning_payload(), observation_index(), paged_evidence(), AgentNarrative, CSVTools, metric_capabilities(), Observation (+9 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (24): TableReadOptions, _lazy_table(), checked_predictions(), detect_table_format(), inside(), read_table(), resolve_inputs(), find() (+16 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (16): default_space(), interrupt_search(), relative_gain(), run_search(), apply_decision(), persist(), request_control(), TrainingDecision (+8 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (17): SourcePatch, MetricDirection, ResultFormat, SplitColumnStats, StrictModel, TaskAnalysisResult, TaskDefinition, TaskValidationFinding (+9 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (18): better(), export_params(), _allocated_params(), execution_config(), is_thread_parameter(), model_parameters(), restore_thread_settings(), thread_budget_errors() (+10 more)

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (11): BudgetSnapshot, validate_plan(), installed_tools(), ContextAndPatchTests, generate(), case(), PlanningLimitTests, Skills (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.10
Nodes (10): SkillRegistry, check(), render(), RoleSkillTests, generate(), generate(), DecisionSkillCardTests, ResultClient (+2 more)

### Community 14 - "Community 14"
Cohesion: 0.12
Nodes (11): classification(), LibraryCandidate, stopping_metric(), load_adapter(), text_columns(), EnsembleCandidate, load_ensemble(), FitContext (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (6): RoleClient, LLMSettings, AgentProtocolTests, ProviderFailureTests, NarrativeTests, unavailable()

### Community 16 - "Community 16"
Cohesion: 0.09
Nodes (12): save(), OfflineLLM, OfflinePipeline, ControlledLLM, FeatureClient, NativeStarterClient, RepairLLM, ScoutingTests (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.10
Nodes (18): apply_overrides(), constant_space(), normalize_protocol(), build_planning_graph(), for_hypothesis(), generate(), hypotheses(), is_light() (+10 more)

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (7): feature_hash(), FeatureSpec, FeatureAugmentingPreprocessor, FeatureTransformer, validate_features(), definition(), FeatureDSLTests

### Community 19 - "Community 19"
Cohesion: 0.16
Nodes (5): FileRole, TableProfile, DataInspector, InputLayout, _resolve_path()

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (8): LLMUnavailable, request_messages(), NetworkCircuit, thinking_parameters(), NetworkPolicyTests, handler(), generate(), generate()

### Community 21 - "Community 21"
Cohesion: 0.19
Nodes (5): ColumnRole, DataIssue, DatasetChecks, TaskSpec, _normalise_name()

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (9): paired_evidence(), select_feature_baseline(), Champion, EvaluationResult, feature_context(), SessionAccountingTests, run(), renamed() (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.20
Nodes (11): Hypothesis, HypothesisBatch, PlanBatch, ValidationReport, RepairingPlanningClient, StarterPlanner, Planner, PlanClient (+3 more)

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (10): SplitSpec, group_keys(), inner_protocol(), make_split(), split_indices(), validation_groups(), MetricsAndSplitTests, execution_case() (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.13
Nodes (8): compact_diagnostics(), RunState, champion_snapshot(), evaluation_summary(), experiment_summary(), plan_summary(), profile_summary(), WorkerRuntimeError

### Community 26 - "Community 26"
Cohesion: 0.14
Nodes (11): SearchGroupPlan, collect_groups(), ensemble_recipes(), expand_families(), internal_winner(), limit_parent_groups(), variant_id(), FamilyPlanTests (+3 more)

### Community 27 - "Community 27"
Cohesion: 0.12
Nodes (3): JsonSerializer, SQLiteCheckpointer, _unsupported()

### Community 28 - "Community 28"
Cohesion: 0.13
Nodes (10): StructuredOutputError, plan_ensemble(), read_component_history(), EnsembleProposal, component_brief(), generate(), InitialPlanningTests, generate() (+2 more)

### Community 29 - "Community 29"
Cohesion: 0.21
Nodes (7): validator_repair(), Workspace, apply_patch(), read_file(), main(), request(), ComponentTests

### Community 30 - "Community 30"
Cohesion: 0.18
Nodes (5): ToolResult, ResearchTools, ToolSet, PlanningRecoveryTests, read_skills()

### Community 31 - "Community 31"
Cohesion: 0.13
Nodes (5): preprocessor(), text_values(), Candidate, create_candidate(), load_candidate()

### Community 32 - "Community 32"
Cohesion: 0.22
Nodes (14): ColumnProfile, DatasetProfile, Severity, _format_number(), main(), _write_json(), build_llm_markdown(), _distribution_text() (+6 more)

### Community 33 - "Community 33"
Cohesion: 0.29
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 34 - "Community 34"
Cohesion: 0.16
Nodes (6): package(), main(), main(), archive_hash(), main(), FlatPackageTests

### Community 35 - "Community 35"
Cohesion: 0.20
Nodes (13): config_hash(), FeatureEvidence, FeatureExperimentPlan, FeatureProposal, complete_feature_session(), feature_session_key(), feature_task_view(), fixed_model_hash() (+5 more)

### Community 36 - "Community 36"
Cohesion: 0.15
Nodes (8): error_metrics(), PlannerContextDraft, TrainingCheckpoint, data_observations(), error_slices(), add(), validate_decision(), ResearchTests

### Community 37 - "Community 37"
Cohesion: 0.20
Nodes (7): tree_hash(), _execute_module(), load_module(), predict(), write_predictions(), composite_case(), CompositeTests

### Community 38 - "Community 38"
Cohesion: 0.26
Nodes (4): DistributionBucket, _display_value(), _finite_float(), _rate()

### Community 39 - "Community 39"
Cohesion: 0.21
Nodes (7): fallback(), worker_checkpoint(), train(), control(), search_checkpoint(), InnerExecutionTests, TargetEncodingBoundaryTests

### Community 40 - "Community 40"
Cohesion: 0.20
Nodes (5): ContextAgentTests, generate(), generate(), generate(), generate()

### Community 41 - "Community 41"
Cohesion: 0.22
Nodes (9): frozen_plan(), hpo_child(), Intervention, PredictionPolicy, TargetTransform, WeightPolicy, ExperimentRun, HypothesisPlan (+1 more)

### Community 42 - "Community 42"
Cohesion: 0.17
Nodes (6): csv_worker(), _json_value(), validate_sql(), PolarsSQLSkillTests, CSVEngineTests, run()

### Community 43 - "Community 43"
Cohesion: 0.21
Nodes (6): cases(), evidence(), gate(), main(), package_info(), ReleaseGateTests

### Community 44 - "Community 44"
Cohesion: 0.24
Nodes (8): audit_csv(), audit_cycles(), main(), now(), run_case(), save_report(), stop_pipeline(), ExternalQueueStopTests

### Community 48 - "Community 48"
Cohesion: 0.22
Nodes (4): SemanticType, build_llm_context(), DataInspectorTests, write_dictionary()

### Community 50 - "Community 50"
Cohesion: 0.29
Nodes (7): RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 51 - "Community 51"
Cohesion: 0.28
Nodes (3): size(), _plain(), TraceSink

### Community 52 - "Community 52"
Cohesion: 0.19
Nodes (4): capability_matrix(), Budget, InvariantTests, OptionalTimeoutTests

### Community 53 - "Community 53"
Cohesion: 0.26
Nodes (3): main(), main(), main()

### Community 54 - "Community 54"
Cohesion: 0.20
Nodes (3): inspect(), main(), names()

### Community 56 - "Community 56"
Cohesion: 0.32
Nodes (7): case_config(), digest(), main(), select_real_settings(), verify_observation_handoff(), verify_run(), write()

### Community 57 - "Community 57"
Cohesion: 0.33
Nodes (3): finish(), PlanningRejections, generate()

### Community 58 - "Community 58"
Cohesion: 0.31
Nodes (3): FeatureSession, FeatureSet, FeaturePlanningTests

### Community 59 - "Community 59"
Cohesion: 0.31
Nodes (6): feature_profile(), features(), targets(), runtime_versions(), failure_result(), main()

### Community 60 - "Community 60"
Cohesion: 0.39
Nodes (4): capacity(), index(), main(), validate()

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

### Community 95 - "Community 95"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 101 - "Community 101"
Cohesion: 0.60
Nodes (3): current_context(), generate(), validate_selection()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 589 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 20`, `Community 22`, `Community 23`, `Community 24`, `Community 25`, `Community 26`, `Community 27`, `Community 28`, `Community 30`, `Community 31`, `Community 32`, `Community 34`, `Community 36`, `Community 40`, `Community 41`, `Community 49`, `Community 51`, `Community 52`, `Community 96`, `Community 97`, `Community 98`, `Community 105`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 58 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 58 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0914476119955572 - nodes in this community are weakly interconnected._
- **Why does `RunConfig` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 12`, `Community 14`, `Community 16`, `Community 20`, `Community 22`, `Community 23`, `Community 25`, `Community 26`, `Community 28`, `Community 30`, `Community 34`, `Community 36`, `Community 40`, `Community 42`, `Community 43`, `Community 46`, `Community 49`, `Community 50`, `Community 52`, `Community 56`, `Community 58`, `Community 60`, `Community 61`, `Community 97`, `Community 98`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 38 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.07412587412587412 - nodes in this community are weakly interconnected._