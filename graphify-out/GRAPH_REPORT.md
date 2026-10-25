# Graph Report - mlmom-project  (2026-10-25)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2230 nodes · 8567 edges · 106 communities (92 shown, 14 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 1095 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a0349ef7`
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
- Community 104
- Community 105

## God Nodes (most connected - your core abstractions)
1. `RunConfig` - 171 edges
2. `Pipeline` - 170 edges
3. `DataInspector` - 103 edges
4. `case()` - 101 edges
5. `atomic_json()` - 94 edges
6. `ExperimentStore` - 92 edges
7. `OfflineLLM` - 92 edges
8. `build_planning_graph()` - 91 edges
9. `Skills` - 72 edges
10. `RoleClient` - 70 edges

## Surprising Connections (you probably didn't know these)
- `AdvisoryCostTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_advisory_costs_and_replacement.py → mlmom/persistence.py
- `ExecutablePlanTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_agent_pipeline.py → mlmom/persistence.py
- `ContextAgentTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_context_agent.py → mlmom/persistence.py
- `ContextAndPatchTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_contract_v2.py → mlmom/persistence.py
- `IndependentRecipeTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_controller_groups.py → mlmom/persistence.py

## Import Cycles
- None detected.

## Communities (106 total, 14 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (19): ExperimentStore, RunConfig, Pipeline, component_pool(), PipelineFlowTests, CheckpointIntegrationTests, ContinuousRunTests, finalize() (+11 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (34): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+26 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (13): preprocessor(), PolicyState, create(), load(), _load_policy(), PolicyCandidate, PreparedCandidate, Estimator (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (26): StrictModel, resolve_inputs(), PredictionPolicy, TargetTransform, WeightPolicy, Champion, EvaluationResult, InputFiles (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (17): validate_plan(), ExecutablePlanTests, ContextAndPatchTests, generate(), generate(), case(), PlanningLimitTests, generate() (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (25): better(), export_params(), record_fit(), _allocated_params(), execution_config(), is_thread_parameter(), model_parameters(), restore_thread_settings() (+17 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (21): default_space(), interrupt_search(), relative_gain(), run_search(), attempted(), persist(), remaining(), request_control() (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (13): inspect(), main(), package(), main(), main(), main(), main(), archive_hash() (+5 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (12): feature_hash(), FeatureProgram, FeatureSpec, FeatureAugmentingPreprocessor, FeatureTransformer, validate_features(), Base, definition() (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (16): GlobalAPIError, LLMUnavailable, CodeBundle, CodingState, SourcePatch, traced_node(), bind(), current_sink() (+8 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (25): config_hash(), feature_configuration_keys(), FeatureEvidence, FeatureExperimentPlan, FeatureSession, FeatureSet, complete_feature_session(), dictionary_metadata() (+17 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (24): inside(), find(), file_hash(), CandidateVersion, ExperimentRun, SubmissionArtifact, TaskContext, publish_submission() (+16 more)

### Community 13 - "Community 13"
Cohesion: 0.10
Nodes (9): validate_tables(), dumps(), plain(), DeferredTask, group_remaining(), prepare_alternatives(), process_jobs(), queue_preparation() (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.08
Nodes (16): save(), runtime_versions(), OfflineLLM, OfflinePipeline, ControlledLLM, FeatureClient, NativeStarterClient, broken_code() (+8 more)

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (9): ResourceUnavailable, SandboxManager, synchronized(), trace_event(), worker_runtime(), AdapterTests, ExecutionTests, NativeValidationTests (+1 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (18): atomic_json(), capacity(), terminate_tree(), commit_mailbox(), audit_csv(), audit_cycles(), external_stop(), main() (+10 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (13): MetricDirection, ResultFormat, Severity, TaskAnalysisResult, TaskDefinition, TaskType, TaskValidationFinding, TaskValidationReport (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (9): FileRole, SemanticType, TableProfile, build_llm_context(), DataInspector, InputLayout, _resolve_path(), DataInspectorTests (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.09
Nodes (13): finding_catalog(), prepare_planner_context(), compact_planning_payload(), observation_index(), paged_evidence(), previous_brief_summary(), stored_evidence(), PlannerContextDraft (+5 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (8): classification(), LibraryCandidate, stopping_metric(), load_adapter(), text_columns(), EnsembleCandidate, load_ensemble(), DataPathTests

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (6): RoleClient, LLMSettings, agent_payload(), AgentProtocolTests, ProviderFailureTests, NarrativeTests

### Community 22 - "Community 22"
Cohesion: 0.10
Nodes (14): EnsembleProposal, Hypothesis, collect_groups(), initial_boosting_resources(), limit_parent_groups(), InitialPlanningTests, native_source(), ReviewPromptPlanner (+6 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (12): tree_hash(), load_module(), fallback(), worker_checkpoint(), train(), control(), imported_metadata(), search_checkpoint() (+4 more)

### Community 24 - "Community 24"
Cohesion: 0.12
Nodes (12): AgentNarrative, HypothesisBatch, HypothesisPlan, PlanBatch, PreprocessingSpec, RankedFinding, ResearchProposal, TaskDetail (+4 more)

### Community 25 - "Community 25"
Cohesion: 0.14
Nodes (13): feature_profile(), features(), targets(), finished(), group_keys(), inner_protocol(), make_split(), split_indices() (+5 more)

### Community 26 - "Community 26"
Cohesion: 0.14
Nodes (14): text_values(), inspect_model(), checked_predictions(), diagnostics(), error_metrics(), evaluate(), metric_name(), prediction_kind() (+6 more)

### Community 27 - "Community 27"
Cohesion: 0.10
Nodes (11): TableReadOptions, csv_worker(), CSVTools, _json_value(), _lazy_table(), validate_sql(), detect_table_format(), read_table() (+3 more)

### Community 28 - "Community 28"
Cohesion: 0.12
Nodes (8): metric_capabilities(), Observation, ReconnaissanceReport, data_observations(), add(), ReconnaissanceSession, ResearchTests, EvidenceContractTests

### Community 29 - "Community 29"
Cohesion: 0.14
Nodes (5): SkillRegistry, check(), render(), RoleSkillTests, SkillTests

### Community 30 - "Community 30"
Cohesion: 0.13
Nodes (18): role_window(), execution_contract(), plan_field_sources(), constant_space(), normalize_protocol(), normalize_cpu_hints(), build_planning_graph(), for_hypothesis() (+10 more)

### Community 31 - "Community 31"
Cohesion: 0.16
Nodes (11): evaluate(), process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), worker_record(), Job, watchdog() (+3 more)

### Community 32 - "Community 32"
Cohesion: 0.12
Nodes (10): reuse_assignment(), comparable_scores(), relative_gain(), ResultReview, review_result(), review_payload(), DiamondsBoundaryTests, generate() (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.11
Nodes (7): configuration_diff(), read_fit_facts(), inherit_champion(), feature_context(), failure_result(), FitFactsTests, FeatureWorkerTests

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (7): validator_repair(), CodePatch, Workspace, apply_patch(), read_file(), main(), request()

### Community 35 - "Community 35"
Cohesion: 0.19
Nodes (5): ColumnRole, DataIssue, DatasetChecks, TaskSpec, _normalise_name()

### Community 36 - "Community 36"
Cohesion: 0.12
Nodes (4): JsonSerializer, SQLiteCheckpointer, _unsupported(), ComponentTests

### Community 37 - "Community 37"
Cohesion: 0.13
Nodes (4): AssignmentClient, PreparationPool, PreparationConcurrencyTests, assigned()

### Community 38 - "Community 38"
Cohesion: 0.14
Nodes (5): request_messages(), NetworkCircuit, thinking_parameters(), NetworkPolicyTests, handler()

### Community 39 - "Community 39"
Cohesion: 0.16
Nodes (6): ToolResult, ResearchTools, ToolSet, read_evidence(), PlanningRecoveryTests, read_skills()

### Community 40 - "Community 40"
Cohesion: 0.26
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 41 - "Community 41"
Cohesion: 0.24
Nodes (4): DistributionBucket, _display_value(), _finite_float(), _rate()

### Community 42 - "Community 42"
Cohesion: 0.19
Nodes (9): execution_context(), observed_execution(), related_fit_observations(), frozen_plan(), champion_snapshot(), evaluation_summary(), experiment_summary(), plan_summary() (+1 more)

### Community 43 - "Community 43"
Cohesion: 0.14
Nodes (5): BudgetSnapshot, Budget, InvariantTests, OptionalTimeoutTests, SchedulerIntegrationTests

### Community 44 - "Community 44"
Cohesion: 0.18
Nodes (10): StructuredOutputError, first(), generate(), RealFailureRegressions, handler(), handler(), handler(), handler() (+2 more)

### Community 45 - "Community 45"
Cohesion: 0.17
Nodes (11): compact_diagnostics(), FitExecutionFacts, component_brief(), capture_boundary(), compact(), compact_intervention(), hydrate(), main() (+3 more)

### Community 46 - "Community 46"
Cohesion: 0.18
Nodes (11): case_config(), digest(), main(), select_real_settings(), verify_observation_handoff(), verify_run(), write(), create_task() (+3 more)

### Community 47 - "Community 47"
Cohesion: 0.17
Nodes (6): finish(), PlanningRejections, RetryTests, planning_call(), generate(), generate()

### Community 48 - "Community 48"
Cohesion: 0.17
Nodes (3): FeatureProposal, AtomicReplacementTests, run()

### Community 49 - "Community 49"
Cohesion: 0.15
Nodes (4): Candidate, _execute_module(), FitContext, Preprocessor

### Community 50 - "Community 50"
Cohesion: 0.22
Nodes (6): cases(), evidence(), gate(), main(), package_info(), ReleaseGateTests

### Community 52 - "Community 52"
Cohesion: 0.19
Nodes (4): run(), SessionAccountingTests, run(), run()

### Community 53 - "Community 53"
Cohesion: 0.35
Nodes (10): ColumnProfile, DatasetProfile, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent() (+2 more)

### Community 54 - "Community 54"
Cohesion: 0.27
Nodes (4): size(), _plain(), TraceSink, probe()

### Community 55 - "Community 55"
Cohesion: 0.29
Nodes (5): plan_ensemble(), read_component_history(), IndependentRecipeTests, builder(), generate()

### Community 57 - "Community 57"
Cohesion: 0.24
Nodes (3): ControllerGroupTests, graph_input(), GroupClient

### Community 58 - "Community 58"
Cohesion: 0.28
Nodes (3): verify_source(), GeneratedCandidateTests, generate()

### Community 59 - "Community 59"
Cohesion: 0.29
Nodes (5): InferenceConfidence, SplitColumnStats, _format_number(), main(), _write_json()

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

### Community 99 - "Community 99"
Cohesion: 0.60
Nodes (3): current_context(), generate(), validate_selection()

### Community 101 - "Community 101"
Cohesion: 0.50
Nodes (3): generate(), __init__(), generate()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 648 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 6`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 19`, `Community 20`, `Community 21`, `Community 22`, `Community 24`, `Community 25`, `Community 28`, `Community 29`, `Community 31`, `Community 32`, `Community 36`, `Community 37`, `Community 39`, `Community 42`, `Community 43`, `Community 45`, `Community 46`, `Community 47`, `Community 53`, `Community 54`, `Community 57`, `Community 94`, `Community 96`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 45 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.08734567901234568 - nodes in this community are weakly interconnected._
- **Why does `RunConfig` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 19`, `Community 20`, `Community 22`, `Community 24`, `Community 25`, `Community 28`, `Community 31`, `Community 32`, `Community 38`, `Community 39`, `Community 43`, `Community 44`, `Community 45`, `Community 46`, `Community 47`, `Community 50`, `Community 52`, `Community 54`, `Community 55`, `Community 57`, `Community 58`, `Community 60`, `Community 95`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06634615384615385 - nodes in this community are weakly interconnected._