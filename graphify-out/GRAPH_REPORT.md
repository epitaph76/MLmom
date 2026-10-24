# Graph Report - mlmom-project  (2026-10-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2116 nodes · 7860 edges · 111 communities (90 shown, 21 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 971 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `11e691e7`
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
- Community 107
- Community 108
- Community 109
- Community 110

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 151 edges
2. `RunConfig` - 143 edges
3. `DataInspector` - 103 edges
4. `case()` - 91 edges
5. `OfflineLLM` - 78 edges
6. `build_planning_graph()` - 78 edges
7. `ExperimentStore` - 76 edges
8. `atomic_json()` - 76 edges
9. `Skills` - 64 edges
10. `RoleClient` - 60 edges

## Surprising Connections (you probably didn't know these)
- `ModelContractV2Tests` --uses--> `FidelityError`  [INFERRED]
  tests/test_contract_v2.py → mlmom/execution.py
- `LightGBMFitEvidenceTests` --uses--> `FidelityError`  [INFERRED]
  tests/test_lightgbm_fit_evidence.py → mlmom/execution.py
- `ExecutionContractTests` --uses--> `ResultReview`  [INFERRED]
  tests/test_audit_contracts.py → mlmom/pipeline_contracts.py
- `ExecutionContractTests` --uses--> `FitContext`  [INFERRED]
  tests/test_audit_contracts.py → mlmom/sdk.py
- `ExecutionResourceTests` --uses--> `FitContext`  [INFERRED]
  tests/test_execution_resources.py → mlmom/sdk.py

## Import Cycles
- None detected.

## Communities (111 total, 21 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (30): audit_fit(), complete_export(), _component_objects(), config_hash(), _equal(), FidelityError, fitted_components(), inspect_component() (+22 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (19): FeatureSession, ExperimentStore, RunConfig, component_pool(), CheckpointIntegrationTests, ControlledLLM, SubmissionIsolationTests, generate() (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (28): AgentNarrative, CodePatch, CodingState, SourcePatch, StrictModel, TableReadOptions, FeatureEvidence, FeatureExperimentPlan (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (15): BudgetSnapshot, installed_tools(), Budget, ResourceUnavailable, SandboxManager, trace_event(), AdapterTests, ExecutionTests (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (8): DeferredTask, Pipeline, finished(), group_remaining(), process_jobs(), experiment_summary(), WorkerRuntimeError, emit()

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (10): ColumnProfile, ColumnRole, DataIssue, DatasetChecks, TableProfile, TaskSpec, DataInspector, InputLayout (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (11): classification(), LibraryCandidate, load_adapter(), preprocessor(), text_columns(), text_values(), EnsembleCandidate, load_ensemble() (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (17): evaluate(), atomic_json(), process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), worker_record(), capacity() (+9 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (7): group_keys(), inner_protocol(), make_split(), split_indices(), validation_groups(), execution_case(), InnerProtocolTests

### Community 9 - "Community 9"
Cohesion: 0.09
Nodes (9): validate_plan(), MetricsAndSplitTests, generate(), InitialPlanningTests, case(), PlanningLimitTests, Skills, generate() (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.08
Nodes (10): SkillRegistry, check(), render(), RoleSkillTests, generate(), generate(), DecisionSkillCardTests, ResultClient (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (22): require_admitted(), inside(), read_table(), resolve_inputs(), find(), CandidateVersion, ExperimentRun, publish_submission() (+14 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (15): checked_predictions(), compare_roundtrip(), tree_hash(), load_module(), create(), load(), failure_result(), main() (+7 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (12): CodeBundle, SeparatePreprocessingTests, run(), ClockBudget, Coder, CodingBudgetTests, Sandbox, Store (+4 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (11): MetricDirection, ResultFormat, TaskDefinition, TaskValidationFinding, TaskValidationReport, ValidationStrategy, _normalise_metric(), TaskValidator (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.12
Nodes (5): RoleClient, LLMSettings, AgentProtocolTests, ProviderFailureTests, unavailable()

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (19): DatasetProfile, FileRole, InferenceConfidence, SemanticType, Severity, SplitColumnStats, build_llm_context(), main() (+11 more)

### Community 17 - "Community 17"
Cohesion: 0.09
Nodes (15): LLMUnavailable, StructuredOutputError, generate(), ToolSet, generate(), generate(), PlanningRecoveryTests, read_skills() (+7 more)

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (16): plan_ensemble(), read_component_history(), frozen_plan(), EnsembleProposal, Hypothesis, HypothesisPlan, SearchGroupPlan, collect_groups() (+8 more)

### Community 19 - "Community 19"
Cohesion: 0.13
Nodes (10): inspect(), package(), main(), main(), package_info(), main(), archive_hash(), main() (+2 more)

### Community 20 - "Community 20"
Cohesion: 0.10
Nodes (13): save(), OfflineLLM, OfflinePipeline, FeatureClient, NativeStarterClient, broken_code(), RepairLLM, ScoutingTests (+5 more)

### Community 21 - "Community 21"
Cohesion: 0.10
Nodes (13): relative_gain(), run_search(), apply_decision(), persist(), request_control(), TrainingDecision, TrainingControlTests, batch() (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.11
Nodes (17): FeatureSet, constant_space(), normalize_protocol(), normalize_champion_assignment(), build_planning_graph(), for_hypothesis(), hypotheses(), is_light() (+9 more)

### Community 23 - "Community 23"
Cohesion: 0.14
Nodes (10): HypothesisBatch, PlanBatch, ValidationReport, RepairingPlanningClient, generate(), native_source(), ReviewPromptPlanner, StarterPlanner (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.13
Nodes (13): _leaves(), screen_features(), validate_features(), dictionary_metadata(), feature_profile(), fixed_model_hash(), paired_evidence(), actual_configuration() (+5 more)

### Community 25 - "Community 25"
Cohesion: 0.13
Nodes (14): better(), TrainingCheckpoint, TrialResult, fallback(), worker_checkpoint(), fit_context(), train(), control() (+6 more)

### Community 26 - "Community 26"
Cohesion: 0.15
Nodes (12): main(), audit_csv(), audit_cycles(), main(), now(), run_case(), save_report(), stop_pipeline() (+4 more)

### Community 27 - "Community 27"
Cohesion: 0.12
Nodes (5): FrozenEnsemble, objective(), capture(), records_for(), _execute_module()

### Community 28 - "Community 28"
Cohesion: 0.14
Nodes (6): csv_worker(), _json_value(), validate_sql(), PolarsSQLSkillTests, CSVEngineTests, run()

### Community 29 - "Community 29"
Cohesion: 0.15
Nodes (10): GlobalAPIError, request_messages(), role_window(), comparable_scores(), relative_gain(), extract_json(), review_result(), WorkerInfrastructureError (+2 more)

### Community 30 - "Community 30"
Cohesion: 0.16
Nodes (9): ContextAgentTests, generate(), generate(), generate(), generate(), generate(), generate(), generate() (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.12
Nodes (3): JsonSerializer, SQLiteCheckpointer, _unsupported()

### Community 32 - "Community 32"
Cohesion: 0.20
Nodes (8): validator_repair(), Workspace, apply_patch(), read_file(), main(), request(), ContextAndPatchTests, ComponentTests

### Community 33 - "Community 33"
Cohesion: 0.17
Nodes (17): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+9 more)

### Community 34 - "Community 34"
Cohesion: 0.14
Nodes (9): feature_task_view(), run_feature_session(), session_directory(), session_object_key(), require_worker_infrastructure(), SessionAccountingTests, run(), renamed() (+1 more)

### Community 35 - "Community 35"
Cohesion: 0.17
Nodes (10): finding_catalog(), _ground_task_details(), prepare_planner_context(), _quote_text(), _validate(), previous_brief_summary(), FeatureProposal, PlannerContextDraft (+2 more)

### Community 36 - "Community 36"
Cohesion: 0.25
Nodes (5): DistributionBucket, _display_value(), _finite_float(), _format_number(), _rate()

### Community 37 - "Community 37"
Cohesion: 0.20
Nodes (6): _lazy_table(), file_hash(), SubmissionArtifact, snapshot_cycle_submission(), validate_submission_file(), CycleSubmissionTests

### Community 38 - "Community 38"
Cohesion: 0.16
Nodes (7): finish(), PlanningRejections, generate(), RetryTests, planning_call(), generate(), generate()

### Community 39 - "Community 39"
Cohesion: 0.20
Nodes (4): NetworkCircuit, thinking_parameters(), NetworkPolicyTests, handler()

### Community 40 - "Community 40"
Cohesion: 0.16
Nodes (6): Observation, ReconnaissanceReport, data_observations(), add(), ResearchTests, EvidenceContractTests

### Community 41 - "Community 41"
Cohesion: 0.18
Nodes (10): fingerprint(), install(), importing(), patch_loaded(), visit(), _patch(), captured(), _patch_lightgbm_updates() (+2 more)

### Community 42 - "Community 42"
Cohesion: 0.24
Nodes (11): stopping_metric(), inspect_model(), diagnostics(), error_metrics(), evaluate(), metric_name(), prediction_kind(), score_predictions() (+3 more)

### Community 43 - "Community 43"
Cohesion: 0.40
Nodes (4): completion(), TraceTests, handler(), handler()

### Community 44 - "Community 44"
Cohesion: 0.20
Nodes (5): cases(), evidence(), gate(), main(), ReleaseGateTests

### Community 45 - "Community 45"
Cohesion: 0.16
Nodes (5): default_space(), interrupt_search(), dumps(), plain(), report()

### Community 48 - "Community 48"
Cohesion: 0.25
Nodes (8): traced_node(), bind(), current_context(), current_sink(), span(), generate(), validate_selection(), worker()

### Community 49 - "Community 49"
Cohesion: 0.19
Nodes (4): TaskType, MLToolSpec, SkillManifest, capability_matrix()

### Community 50 - "Community 50"
Cohesion: 0.23
Nodes (4): FeatureSpec, FeatureAugmentingPreprocessor, definition(), FeatureDSLTests

### Community 51 - "Community 51"
Cohesion: 0.25
Nodes (3): metric_capabilities(), ReconnaissanceSession, collect()

### Community 52 - "Community 52"
Cohesion: 0.26
Nodes (6): handler(), handler(), handler(), handler(), response(), tool_call()

### Community 53 - "Community 53"
Cohesion: 0.28
Nodes (3): size(), _plain(), TraceSink

### Community 54 - "Community 54"
Cohesion: 0.18
Nodes (3): Base, MeanPreprocessor, Stack

### Community 59 - "Community 59"
Cohesion: 0.27
Nodes (4): compact_diagnostics(), champion_snapshot(), evaluation_summary(), plan_summary()

### Community 62 - "Community 62"
Cohesion: 0.29
Nodes (5): complete_feature_session(), feature_session_key(), select_feature_baseline(), feature_branch(), internal_winner()

### Community 63 - "Community 63"
Cohesion: 0.32
Nodes (3): runtime_versions(), worker_runtime(), WorkerRuntimeTests

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

### Community 96 - "Community 96"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 100 - "Community 100"
Cohesion: 0.38
Nodes (3): compact_planning_payload(), observation_index(), paged_evidence()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 614 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 4` to `Community 1`, `Community 2`, `Community 3`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 20`, `Community 21`, `Community 23`, `Community 25`, `Community 28`, `Community 29`, `Community 30`, `Community 31`, `Community 37`, `Community 38`, `Community 40`, `Community 42`, `Community 51`, `Community 53`, `Community 61`, `Community 63`, `Community 97`, `Community 98`, `Community 104`, `Community 108`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 58 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 58 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06690140845070422 - nodes in this community are weakly interconnected._
- **Why does `DataInspector` connect `Community 5` to `Community 1`, `Community 2`, `Community 36`, `Community 4`, `Community 105`, `Community 12`, `Community 14`, `Community 16`, `Community 49`, `Community 17`, `Community 20`, `Community 28`, `Community 30`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Are the 37 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.09045680687471733 - nodes in this community are weakly interconnected._