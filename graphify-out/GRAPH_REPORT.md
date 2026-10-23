# Graph Report - mlmom-project  (2026-10-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1898 nodes · 6944 edges · 100 communities (86 shown, 14 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 880 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5984ce3f`
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

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 137 edges
2. `RunConfig` - 121 edges
3. `DataInspector` - 103 edges
4. `OfflineLLM` - 72 edges
5. `StrictModel` - 67 edges
6. `atomic_json()` - 66 edges
7. `ExperimentStore` - 62 edges
8. `case()` - 62 edges
9. `build_planning_graph()` - 59 edges
10. `SandboxManager` - 58 edges

## Surprising Connections (you probably didn't know these)
- `FeaturePlanningTests` --uses--> `FeatureSession`  [INFERRED]
  tests/test_feature_session.py → mlmom/feature_contracts.py
- `FeaturePlanningTests` --uses--> `FeatureSet`  [INFERRED]
  tests/test_feature_session.py → mlmom/feature_contracts.py
- `FeatureDSLTests` --uses--> `FitContext`  [INFERRED]
  tests/test_feature_engineering.py → mlmom/sdk.py
- `InfrastructureRoutingTests` --uses--> `Job`  [INFERRED]
  tests/test_worker_supervision.py → mlmom/resources.py
- `InvariantTests` --uses--> `SandboxManager`  [INFERRED]
  tests/test_invariants.py → mlmom/resources.py

## Import Cycles
- None detected.

## Communities (100 total, 14 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (40): compact_diagnostics(), config_hash(), _leaves(), require_admitted(), feature_hash(), FeatureEvidence, FeatureExperimentPlan, FeatureProposal (+32 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (19): process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), Job, ResourceUnavailable, SandboxManager, trace_event() (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (16): classification(), LibraryCandidate, stopping_metric(), load_adapter(), preprocessor(), text_columns(), text_values(), EnsembleCandidate (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (14): JsonSerializer, SQLiteCheckpointer, _unsupported(), SkillRegistry, check(), render(), RoleSkillTests, generate() (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (10): TableReadOptions, validate_tables(), file_hash(), InputFiles, MLToolSpec, SkillManifest, TaskContext, installed_tools() (+2 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (18): ExperimentStore, RunConfig, CheckpointIntegrationTests, ControlledLLM, ContinuousRunTests, FeaturePipelineTests, FeaturePlanningTests, CustomCandidatePipeline (+10 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (29): SourcePatch, StrictModel, plan_ensemble(), read_component_history(), select_feature_baseline(), frozen_plan(), hpo_child(), Intervention (+21 more)

### Community 7 - "Community 7"
Cohesion: 0.09
Nodes (8): LLMUnavailable, RoleClient, StructuredOutputError, LLMSettings, AgentProtocolTests, ProviderFailureTests, generate(), PlanningRecoveryTests

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (10): atomic_json(), dumps(), plain(), SelectedPreviousResults, DeferredTask, Pipeline, finished(), group_remaining() (+2 more)

### Community 9 - "Community 9"
Cohesion: 0.12
Nodes (15): InferenceConfidence, MetricDirection, ResultFormat, Severity, TaskAnalysisResult, TaskDefinition, TaskType, TaskValidationFinding (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (17): BudgetSnapshot, Hypothesis, validate_plan(), generate(), MetricsAndSplitTests, generate(), GeneratedCandidateTests, generate() (+9 more)

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (10): FileRole, SemanticType, SplitColumnStats, TableProfile, build_llm_context(), DataInspector, InputLayout, _resolve_path() (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (9): tree_hash(), load_module(), feature_context(), train(), FeatureWorkerTests, InnerExecutionTests, composite_case(), CompositeTests (+1 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (10): request_messages(), CodeBundle, CodingState, extract_json(), traced_node(), bind(), current_sink(), span() (+2 more)

### Community 14 - "Community 14"
Cohesion: 0.16
Nodes (9): SeparatePreprocessingTests, run(), ClockBudget, Coder, CodingBudgetTests, Sandbox, Store, ShortCycleTests (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (17): better(), FidelityError, feature_profile(), features(), read_table(), targets(), TrialResult, runtime_versions() (+9 more)

### Community 16 - "Community 16"
Cohesion: 0.10
Nodes (9): default_space(), interrupt_search(), relative_gain(), run_search(), apply_decision(), persist(), request_control(), resumed() (+1 more)

### Community 17 - "Community 17"
Cohesion: 0.10
Nodes (18): constant_space(), normalize_protocol(), build_planning_graph(), for_hypothesis(), generate(), hypotheses(), is_light(), persist() (+10 more)

### Community 18 - "Community 18"
Cohesion: 0.14
Nodes (15): TrainingCheckpoint, TrainingDecision, commit_mailbox(), decide(), fallback(), validate_decision(), worker_checkpoint(), control() (+7 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (7): OfflineLLM, OfflinePipeline, ContextAgentTests, generate(), generate(), generate(), FeatureClient

### Community 20 - "Community 20"
Cohesion: 0.19
Nodes (6): ColumnProfile, ColumnRole, DataIssue, DatasetChecks, TaskSpec, _normalise_name()

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (5): PreparedCandidate, Base, MeanPreprocessor, ReviewRegressionTests, Stack

### Community 22 - "Community 22"
Cohesion: 0.14
Nodes (17): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+9 more)

### Community 23 - "Community 23"
Cohesion: 0.19
Nodes (9): inspect_model(), checked_predictions(), diagnostics(), evaluate(), prediction_kind(), score_predictions(), CandidateVersion, SubmissionArtifact (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.18
Nodes (7): DistributionBucket, _display_value(), _finite_float(), _format_number(), main(), _rate(), _write_json()

### Community 25 - "Community 25"
Cohesion: 0.13
Nodes (7): capture(), PolicyState, _execute_module(), create(), load(), _load_policy(), PolicyCandidate

### Community 26 - "Community 26"
Cohesion: 0.22
Nodes (7): validator_repair(), CodePatch, Workspace, apply_patch(), read_file(), main(), request()

### Community 27 - "Community 27"
Cohesion: 0.13
Nodes (7): inspect(), package(), main(), archive_hash(), main(), names(), FlatPackageTests

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (12): metric_name(), inside(), resolve_inputs(), find(), case_config(), digest(), main(), select_real_settings() (+4 more)

### Community 29 - "Community 29"
Cohesion: 0.29
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 30 - "Community 30"
Cohesion: 0.21
Nodes (4): main(), main(), main(), main()

### Community 31 - "Community 31"
Cohesion: 0.18
Nodes (6): comparable_scores(), relative_gain(), inner_protocol(), validation_groups(), execution_case(), InnerProtocolTests

### Community 32 - "Community 32"
Cohesion: 0.18
Nodes (6): save(), ScoutingTests, generate(), generate(), generate(), generate()

### Community 33 - "Community 33"
Cohesion: 0.18
Nodes (6): cases(), evidence(), gate(), main(), package_info(), ReleaseGateTests

### Community 34 - "Community 34"
Cohesion: 0.16
Nodes (7): paged_evidence(), error_metrics(), ResultReview, error_slices(), add(), review_result(), ResearchTests

### Community 35 - "Community 35"
Cohesion: 0.20
Nodes (6): HypothesisBatch, PlanBatch, ValidationReport, RepairingPlanningClient, renamed(), AlternativeFailure

### Community 36 - "Community 36"
Cohesion: 0.18
Nodes (10): GlobalAPIError, role_window(), finding_catalog(), prepare_planner_context(), _validate(), previous_brief_summary(), PlannerContextDraft, profile_summary() (+2 more)

### Community 37 - "Community 37"
Cohesion: 0.24
Nodes (3): audit_fit(), ExecutionContractTests, FittedCandidate

### Community 38 - "Community 38"
Cohesion: 0.21
Nodes (4): NetworkCircuit, thinking_parameters(), NetworkPolicyTests, handler()

### Community 39 - "Community 39"
Cohesion: 0.15
Nodes (5): Observation, ReconnaissanceReport, data_observations(), add(), EvidenceContractTests

### Community 40 - "Community 40"
Cohesion: 0.21
Nodes (4): ToolResult, ResearchTools, ToolSet, read_skills()

### Community 41 - "Community 41"
Cohesion: 0.24
Nodes (5): capacity(), terminate_tree(), index(), main(), validate()

### Community 43 - "Community 43"
Cohesion: 0.19
Nodes (3): main(), report(), main()

### Community 44 - "Community 44"
Cohesion: 0.19
Nodes (9): fingerprint(), install(), importing(), patch_loaded(), visit(), _patch(), captured(), record_for() (+1 more)

### Community 45 - "Community 45"
Cohesion: 0.20
Nodes (9): compare_roundtrip(), complete_export(), _component_objects(), _equal(), fitted_components(), inspect_component(), _json_params(), _missing() (+1 more)

### Community 47 - "Community 47"
Cohesion: 0.25
Nodes (6): Champion, EvaluationResult, SplitSpec, SessionAccountingTests, run(), run()

### Community 49 - "Community 49"
Cohesion: 0.25
Nodes (3): metric_capabilities(), ReconnaissanceSession, collect()

### Community 50 - "Community 50"
Cohesion: 0.36
Nodes (9): DatasetProfile, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent(), _render() (+1 more)

### Community 51 - "Community 51"
Cohesion: 0.28
Nodes (3): size(), _plain(), TraceSink

### Community 52 - "Community 52"
Cohesion: 0.32
Nodes (7): RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 53 - "Community 53"
Cohesion: 0.23
Nodes (6): csv_worker(), _json_value(), _lazy_table(), validate_sql(), detect_table_format(), run()

### Community 54 - "Community 54"
Cohesion: 0.20
Nodes (3): capability_matrix(), Budget, OptionalTimeoutTests

### Community 57 - "Community 57"
Cohesion: 0.21
Nodes (3): categorical_probe(), ContextAndPatchTests, generate()

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

### Community 95 - "Community 95"
Cohesion: 0.60
Nodes (3): current_context(), generate(), load()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 570 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 8` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 10`, `Community 12`, `Community 14`, `Community 15`, `Community 18`, `Community 19`, `Community 21`, `Community 23`, `Community 28`, `Community 31`, `Community 32`, `Community 34`, `Community 36`, `Community 39`, `Community 40`, `Community 41`, `Community 43`, `Community 46`, `Community 47`, `Community 49`, `Community 50`, `Community 51`, `Community 54`, `Community 90`, `Community 92`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 54 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05389876880984952 - nodes in this community are weakly interconnected._
- **Why does `DataInspector` connect `Community 11` to `Community 96`, `Community 32`, `Community 4`, `Community 5`, `Community 7`, `Community 9`, `Community 15`, `Community 50`, `Community 19`, `Community 20`, `Community 24`, `Community 56`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 33 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05894736842105263 - nodes in this community are weakly interconnected._