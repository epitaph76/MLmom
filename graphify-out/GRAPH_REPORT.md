# Graph Report - mlmom-project  (2026-10-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1997 nodes · 7432 edges · 105 communities (82 shown, 23 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 928 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `01964ab2`
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
- Community 97
- Community 98
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 149 edges
2. `RunConfig` - 139 edges
3. `DataInspector` - 103 edges
4. `OfflineLLM` - 78 edges
5. `atomic_json()` - 76 edges
6. `ExperimentStore` - 72 edges
7. `case()` - 69 edges
8. `StrictModel` - 67 edges
9. `build_planning_graph()` - 65 edges
10. `SandboxManager` - 60 edges

## Surprising Connections (you probably didn't know these)
- `InvariantTests` --uses--> `BudgetSnapshot`  [INFERRED]
  tests/test_invariants.py → mlmom/pipeline_contracts.py
- `ClockBudget` --uses--> `BudgetSnapshot`  [INFERRED]
  tests/test_planning_limits.py → mlmom/pipeline_contracts.py
- `SchedulerIntegrationTests` --uses--> `BudgetSnapshot`  [INFERRED]
  tests/test_scheduler_integration.py → mlmom/pipeline_contracts.py
- `SchedulerIntegrationTests` --uses--> `Job`  [INFERRED]
  tests/test_scheduler_integration.py → mlmom/resources.py
- `InfrastructureRoutingTests` --uses--> `Job`  [INFERRED]
  tests/test_worker_supervision.py → mlmom/resources.py

## Import Cycles
- None detected.

## Communities (105 total, 23 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (21): BudgetSnapshot, process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), worker_record(), installed_tools(), Job (+13 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (19): ExperimentStore, RunConfig, component_pool(), CheckpointIntegrationTests, ControlledLLM, SubmissionIsolationTests, FeaturePipelineTests, FeaturePlanningTests (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (20): inspect(), main(), package(), main(), main(), main(), main(), cases() (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (13): JsonSerializer, SQLiteCheckpointer, _unsupported(), SkillRegistry, check(), render(), RoleSkillTests, generate() (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (11): PolicyState, create(), load(), _load_policy(), PolicyCandidate, PreparedCandidate, Estimator, ModelContractV2Tests (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (9): dumps(), plain(), DeferredTask, Pipeline, finished(), group_remaining(), process_jobs(), experiment_summary() (+1 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (12): classification(), LibraryCandidate, stopping_metric(), load_adapter(), preprocessor(), text_columns(), text_values(), EnsembleCandidate (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (16): default_space(), interrupt_search(), relative_gain(), run_search(), apply_decision(), persist(), request_control(), TrainingDecision (+8 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (21): comparable_scores(), relative_gain(), audit_fit(), compare_roundtrip(), complete_export(), _component_objects(), _equal(), FidelityError (+13 more)

### Community 9 - "Community 9"
Cohesion: 0.10
Nodes (25): role_window(), plan_ensemble(), read_component_history(), frozen_plan(), hpo_child(), normalize_protocol(), PredictionPolicy, TargetTransform (+17 more)

### Community 10 - "Community 10"
Cohesion: 0.10
Nodes (10): EnsembleProposal, build_planning_graph(), MetricsAndSplitTests, generate(), InitialPlanningTests, case(), PlanningLimitTests, Skills (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (22): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+14 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (13): OfflineLLM, OfflinePipeline, ContextAgentTests, generate(), generate(), generate(), FeatureClient, NativeStarterClient (+5 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (13): MetricDirection, ResultFormat, TaskAnalysisResult, TaskDefinition, TaskValidationFinding, TaskValidationReport, ValidationStrategy, _normalise_metric() (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (8): ColumnProfile, ColumnRole, DataIssue, DatasetChecks, SplitColumnStats, TaskSpec, DataInspector, _normalise_name()

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (19): GlobalAPIError, finding_catalog(), prepare_planner_context(), _validate(), previous_brief_summary(), bundle_components(), TrainingCheckpoint, RunState (+11 more)

### Community 16 - "Community 16"
Cohesion: 0.10
Nodes (21): verify(), evaluate(), atomic_json(), Budget, capacity(), require_worker_infrastructure(), ResourceUnavailable, terminate_tree() (+13 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (3): CodingState, extract_json(), current_sink()

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (11): CodeBundle, HypothesisBatch, PlanBatch, ValidationReport, RepairingPlanningClient, native_source(), StarterPlanner, broken_code() (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.15
Nodes (16): require_admitted(), read_table(), file_hash(), CandidateVersion, ExperimentRun, SubmissionArtifact, publish_submission(), snapshot_cycle_submission() (+8 more)

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (9): RoleClient, LLMSettings, traced_node(), bind(), span(), AgentProtocolTests, worker(), InfrastructureRoutingTests (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.16
Nodes (9): feature_hash(), FeatureSet, FeatureSpec, FeatureAugmentingPreprocessor, FeatureTransformer, screen_features(), validate_features(), definition() (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.11
Nodes (5): FrozenEnsemble, objective(), load_module(), composite_case(), CompositeTests

### Community 23 - "Community 23"
Cohesion: 0.16
Nodes (12): inspect_model(), checked_predictions(), compact_diagnostics(), diagnostics(), error_metrics(), evaluate(), metric_name(), prediction_kind() (+4 more)

### Community 24 - "Community 24"
Cohesion: 0.17
Nodes (15): better(), feature_profile(), features(), targets(), tree_hash(), TaskContext, runtime_versions(), worker_runtime() (+7 more)

### Community 25 - "Community 25"
Cohesion: 0.12
Nodes (12): LLMUnavailable, StructuredOutputError, ToolSet, generate(), PlanningRecoveryTests, read_skills(), generate(), generate() (+4 more)

### Community 26 - "Community 26"
Cohesion: 0.12
Nodes (14): compact_planning_payload(), observation_index(), paged_evidence(), apply_overrides(), constant_space(), for_hypothesis(), generate(), hypotheses() (+6 more)

### Community 27 - "Community 27"
Cohesion: 0.14
Nodes (10): TableReadOptions, csv_worker(), _json_value(), _lazy_table(), validate_sql(), detect_table_format(), inside(), resolve_inputs() (+2 more)

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (17): SourcePatch, StrictModel, FeatureEvidence, FeatureExperimentPlan, FeatureProposal, FeatureSession, InputFiles, MLToolSpec (+9 more)

### Community 29 - "Community 29"
Cohesion: 0.10
Nodes (7): Observation, ReconnaissanceReport, data_observations(), add(), failure_result(), ResearchTests, EvidenceContractTests

### Community 30 - "Community 30"
Cohesion: 0.21
Nodes (7): validator_repair(), CodePatch, Workspace, apply_patch(), read_file(), main(), request()

### Community 31 - "Community 31"
Cohesion: 0.17
Nodes (4): FileRole, TableProfile, InputLayout, _resolve_path()

### Community 32 - "Community 32"
Cohesion: 0.15
Nodes (8): select_feature_baseline(), Champion, EvaluationResult, SplitSpec, SessionAccountingTests, run(), renamed(), run()

### Community 33 - "Community 33"
Cohesion: 0.17
Nodes (12): capture(), fingerprint(), install(), importing(), patch_loaded(), visit(), _patch(), captured() (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (5): request_messages(), NetworkCircuit, thinking_parameters(), NetworkPolicyTests, handler()

### Community 35 - "Community 35"
Cohesion: 0.26
Nodes (7): DistributionBucket, _display_value(), _finite_float(), _format_number(), main(), _rate(), _write_json()

### Community 36 - "Community 36"
Cohesion: 0.18
Nodes (6): save(), ScoutingTests, generate(), generate(), generate(), generate()

### Community 37 - "Community 37"
Cohesion: 0.24
Nodes (14): config_hash(), _leaves(), complete_feature_session(), feature_session_key(), feature_task_view(), fixed_model_hash(), paired_evidence(), actual_configuration() (+6 more)

### Community 38 - "Community 38"
Cohesion: 0.21
Nodes (6): inner_protocol(), make_split(), split_indices(), validation_groups(), execution_case(), InnerProtocolTests

### Community 39 - "Community 39"
Cohesion: 0.40
Nodes (4): completion(), TraceTests, handler(), handler()

### Community 41 - "Community 41"
Cohesion: 0.23
Nodes (4): size(), _plain(), TraceSink, ProviderFailureTests

### Community 42 - "Community 42"
Cohesion: 0.34
Nodes (10): DatasetProfile, Severity, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent() (+2 more)

### Community 43 - "Community 43"
Cohesion: 0.22
Nodes (4): SemanticType, build_llm_context(), DataInspectorTests, write_dictionary()

### Community 44 - "Community 44"
Cohesion: 0.25
Nodes (3): metric_capabilities(), ReconnaissanceSession, collect()

### Community 46 - "Community 46"
Cohesion: 0.15
Nodes (3): Candidate, _execute_module(), Preprocessor

### Community 49 - "Community 49"
Cohesion: 0.33
Nodes (7): RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 51 - "Community 51"
Cohesion: 0.33
Nodes (7): case_config(), digest(), main(), select_real_settings(), verify_observation_handoff(), verify_run(), write()

### Community 52 - "Community 52"
Cohesion: 0.29
Nodes (3): dictionary_metadata(), feature_context(), FeatureWorkerTests

### Community 54 - "Community 54"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 55 - "Community 55"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 56 - "Community 56"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 57 - "Community 57"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 58 - "Community 58"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 59 - "Community 59"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 60 - "Community 60"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 61 - "Community 61"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

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

### Community 91 - "Community 91"
Cohesion: 0.53
Nodes (4): with_spec(), champion_snapshot(), evaluation_summary(), plan_summary()

### Community 95 - "Community 95"
Cohesion: 0.60
Nodes (3): current_context(), generate(), load()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 589 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 11`, `Community 12`, `Community 13`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 19`, `Community 20`, `Community 23`, `Community 24`, `Community 25`, `Community 27`, `Community 28`, `Community 29`, `Community 32`, `Community 36`, `Community 41`, `Community 42`, `Community 44`, `Community 45`, `Community 47`, `Community 86`, `Community 87`, `Community 94`, `Community 100`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 57 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05364314400458979 - nodes in this community are weakly interconnected._
- **Why does `DataInspector` connect `Community 14` to `Community 1`, `Community 98`, `Community 35`, `Community 36`, `Community 42`, `Community 43`, `Community 12`, `Community 13`, `Community 48`, `Community 17`, `Community 24`, `Community 25`, `Community 27`, `Community 31`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Are the 36 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.08695652173913043 - nodes in this community are weakly interconnected._