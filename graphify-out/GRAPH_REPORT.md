# Graph Report - mlmom-project  (2026-10-25)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2317 nodes · 8987 edges · 114 communities (96 shown, 18 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 1137 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7ad10668`
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
- Community 110
- Community 111
- Community 112
- Community 113

## God Nodes (most connected - your core abstractions)
1. `RunConfig` - 178 edges
2. `Pipeline` - 170 edges
3. `atomic_json()` - 106 edges
4. `DataInspector` - 103 edges
5. `case()` - 103 edges
6. `ExperimentStore` - 98 edges
7. `build_planning_graph()` - 93 edges
8. `OfflineLLM` - 92 edges
9. `Skills` - 72 edges
10. `RoleClient` - 72 edges

## Surprising Connections (you probably didn't know these)
- `probe()` --uses--> `FeatureProposal`  [INFERRED]
  scripts/probe_feature_programs.py → mlmom/feature_contracts.py
- `AtomicReplacementTests` --uses--> `FeatureProposal`  [INFERRED]
  tests/test_advisory_costs_and_replacement.py → mlmom/feature_contracts.py
- `FeatureProgramTests` --uses--> `FeatureProposal`  [INFERRED]
  tests/test_feature_programs.py → mlmom/feature_contracts.py
- `NarrativeTests` --uses--> `FeatureProposal`  [INFERRED]
  tests/test_planning_resilience.py → mlmom/feature_contracts.py
- `FeaturePlanningTests` --uses--> `FeatureSession`  [INFERRED]
  tests/test_feature_session.py → mlmom/feature_contracts.py

## Import Cycles
- None detected.

## Communities (114 total, 18 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (33): config_hash(), configuration_diff(), feature_configuration_keys(), feature_hash(), FeatureEvidence, FeatureProposal, FeatureSession, FeatureSet (+25 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (17): FeatureProgram, FeatureSpec, FeatureAugmentingPreprocessor, FeatureTransformer, validate_features(), EvaluationResult, AtomicReplacementTests, Base (+9 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (26): AgentNarrative, plan_ensemble(), FeatureExperimentPlan, Intervention, Champion, EnsembleProposal, Hypothesis, HypothesisBatch (+18 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (37): build_coding_graph(), check_preprocessing(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing() (+29 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (15): PolicyState, Candidate, FitContext, create(), load(), _load_policy(), PolicyCandidate, PolicyValidationScorer (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (13): JsonSerializer, SQLiteCheckpointer, _unsupported(), SkillRegistry, check(), render(), RoleSkillTests, generate() (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (14): dumps(), plain(), SelectedPreviousResults, DeferredTask, Pipeline, finished(), group_remaining(), prepare_alternatives() (+6 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (4): resolve_inputs(), InputFiles, MLToolSpec, installed_tools()

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (30): inside(), read_table(), find(), validate_tables(), file_hash(), CandidateVersion, ExperimentRun, SubmissionArtifact (+22 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (30): better(), checked_predictions(), export_params(), read_fit_facts(), record_fit(), execution_config(), importable_baseline(), apply_overrides() (+22 more)

### Community 10 - "Community 10"
Cohesion: 0.06
Nodes (19): CodeBundle, OfflineLLM, OfflinePipeline, ContextAgentTests, generate(), generate(), generate(), generate() (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (13): ExperimentStore, PipelineFlowTests, SubmissionIsolationTests, generate(), generate(), FeaturePipelineTests, FeaturePlanningTests, InitialBoostingIntegrationTests (+5 more)

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (19): inspect_model(), diagnostics(), error_metrics(), evaluate(), metric_name(), prediction_kind(), score_predictions(), canonical_metric() (+11 more)

### Community 13 - "Community 13"
Cohesion: 0.10
Nodes (18): evaluate(), atomic_json(), process_identity(), process_matches(), ProcessIdentity, recorded_process_matches(), worker_record(), capacity() (+10 more)

### Community 14 - "Community 14"
Cohesion: 0.09
Nodes (10): classification(), LibraryCandidate, stopping_metric(), load_adapter(), text_columns(), text_values(), EnsembleCandidate, load_ensemble() (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (11): BudgetSnapshot, ResourceUnavailable, SandboxManager, synchronized(), trace_event(), AdapterTests, ExecutionTests, NativeValidationTests (+3 more)

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (20): SourcePatch, DatasetProfile, InferenceConfidence, MetricDirection, ResultFormat, StrictModel, TableReadOptions, TaskAnalysisResult (+12 more)

### Community 17 - "Community 17"
Cohesion: 0.08
Nodes (14): generate(), generate(), case(), PlanningLimitTests, generate(), Skills, generate(), PolicyConsistencyTests (+6 more)

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (5): RoleClient, LLMSettings, AgentProtocolTests, ProviderFailureTests, unavailable()

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (18): read_component_history(), compact_diagnostics(), comparable_scores(), relative_gain(), FitExecutionFacts, review_payload(), component_brief(), capture_boundary() (+10 more)

### Community 20 - "Community 20"
Cohesion: 0.11
Nodes (9): LLMUnavailable, request_messages(), extract_json(), thinking_parameters(), traced_node(), bind(), current_sink(), span() (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (8): metric_capabilities(), Observation, ReconnaissanceReport, data_observations(), add(), ReconnaissanceSession, ResearchTests, EvidenceContractTests

### Community 22 - "Community 22"
Cohesion: 0.11
Nodes (13): CodingState, reuse_assignment(), execution_context(), observed_execution(), related_fit_observations(), select_feature_baseline(), frozen_plan(), RunState (+5 more)

### Community 23 - "Community 23"
Cohesion: 0.08
Nodes (6): preprocessor(), PreparedCandidate, Base, MeanPreprocessor, ReviewRegressionTests, Stack

### Community 24 - "Community 24"
Cohesion: 0.13
Nodes (8): main(), package(), main(), main(), archive_hash(), main(), main(), FlatPackageTests

### Community 25 - "Community 25"
Cohesion: 0.14
Nodes (11): _allocated_params(), is_thread_parameter(), model_parameters(), restore_thread_settings(), thread_budget_errors(), thread_settings(), constant_space(), hpo_child() (+3 more)

### Community 26 - "Community 26"
Cohesion: 0.13
Nodes (8): RunConfig, capability_matrix(), Budget, InvariantTests, OptionalTimeoutTests, evaluation(), RotatingHoldoutTests, GroupPipelineTests

### Community 27 - "Community 27"
Cohesion: 0.16
Nodes (6): tree_hash(), load_module(), InnerExecutionTests, composite_case(), CompositeTests, TargetEncodingBoundaryTests

### Community 28 - "Community 28"
Cohesion: 0.20
Nodes (5): ColumnRole, DataIssue, DatasetChecks, TaskSpec, _normalise_name()

### Community 29 - "Community 29"
Cohesion: 0.12
Nodes (15): execution_contract(), plan_field_sources(), normalize_protocol(), build_planning_graph(), for_hypothesis(), generate(), hypotheses(), is_light() (+7 more)

### Community 30 - "Community 30"
Cohesion: 0.16
Nodes (9): GlobalAPIError, role_window(), finding_catalog(), prepare_planner_context(), previous_brief_summary(), stored_evidence(), PlannerContextDraft, ResultReview (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.21
Nodes (8): validator_repair(), CodePatch, Workspace, apply_patch(), read_file(), main(), request(), ContextAndPatchTests

### Community 32 - "Community 32"
Cohesion: 0.14
Nodes (11): audit_csv(), audit_cycles(), external_stop(), main(), now(), run_case(), save_report(), stop_pipeline() (+3 more)

### Community 33 - "Community 33"
Cohesion: 0.16
Nodes (6): SemanticType, SplitColumnStats, build_llm_context(), DataInspector, DataInspectorTests, write_dictionary()

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (4): FileRole, TableProfile, InputLayout, _resolve_path()

### Community 35 - "Community 35"
Cohesion: 0.17
Nodes (12): features(), case_config(), digest(), main(), select_real_settings(), verify_observation_handoff(), verify_run(), write() (+4 more)

### Community 36 - "Community 36"
Cohesion: 0.29
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 37 - "Community 37"
Cohesion: 0.16
Nodes (4): AssignmentClient, PreparationPool, PreparationConcurrencyTests, assigned()

### Community 38 - "Community 38"
Cohesion: 0.14
Nodes (6): ControllerGroupTests, graph_input(), GroupClient, IndependentRecipeTests, builder(), generate()

### Community 39 - "Community 39"
Cohesion: 0.26
Nodes (4): DistributionBucket, _display_value(), _finite_float(), _rate()

### Community 40 - "Community 40"
Cohesion: 0.18
Nodes (6): save(), ScoutingTests, generate(), generate(), generate(), generate()

### Community 41 - "Community 41"
Cohesion: 0.18
Nodes (10): PredictionPolicy, TargetTransform, WeightPolicy, PlannerPlan, compatible_boosters(), configure_composite(), ensemble_recipes(), expand_families() (+2 more)

### Community 42 - "Community 42"
Cohesion: 0.19
Nodes (9): TrainingCheckpoint, commit_mailbox(), decide(), fallback(), validate_decision(), worker_checkpoint(), training(), checkpoint() (+1 more)

### Community 43 - "Community 43"
Cohesion: 0.15
Nodes (6): compact_planning_payload(), observation_index(), paged_evidence(), feature_data_tools(), remember(), measured()

### Community 44 - "Community 44"
Cohesion: 0.18
Nodes (8): default_space(), interrupt_search(), run_search(), attempted(), persist(), remaining(), request_control(), room()

### Community 45 - "Community 45"
Cohesion: 0.20
Nodes (6): cases(), evidence(), gate(), main(), package_info(), ReleaseGateTests

### Community 46 - "Community 46"
Cohesion: 0.19
Nodes (6): csv_worker(), _json_value(), _lazy_table(), validate_sql(), PolarsSQLSkillTests, run()

### Community 47 - "Community 47"
Cohesion: 0.18
Nodes (8): TrainingDecision, TrainingControlTests, batch(), initial(), objective(), control(), resumed(), control()

### Community 48 - "Community 48"
Cohesion: 0.13
Nodes (3): InitialPlanningTests, generate(), ReviewPromptPlanner

### Community 50 - "Community 50"
Cohesion: 0.20
Nodes (7): SearchGroupPlan, collect_groups(), limit_parent_groups(), FamilyPlanTests, generate(), generate(), generate()

### Community 51 - "Community 51"
Cohesion: 0.29
Nodes (4): size(), _plain(), TraceSink, main()

### Community 53 - "Community 53"
Cohesion: 0.29
Nodes (8): first(), RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 54 - "Community 54"
Cohesion: 0.23
Nodes (4): inspect(), main(), main(), names()

### Community 55 - "Community 55"
Cohesion: 0.37
Nodes (10): ColumnProfile, Severity, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent() (+2 more)

### Community 59 - "Community 59"
Cohesion: 0.33
Nodes (3): NetworkCircuit, NetworkPolicyTests, handler()

### Community 60 - "Community 60"
Cohesion: 0.31
Nodes (4): ToolResult, ResearchTools, ToolSet, read_evidence()

### Community 61 - "Community 61"
Cohesion: 0.22
Nodes (4): _format_number(), main(), _write_json(), detect_table_format()

### Community 62 - "Community 62"
Cohesion: 0.33
Nodes (3): finish(), PlanningRejections, generate()

### Community 63 - "Community 63"
Cohesion: 0.31
Nodes (4): SamplerOwnershipTests, control(), control(), resumed()

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

### Community 103 - "Community 103"
Cohesion: 0.29
Nodes (5): check_generated(), preprocessing(), generate(), reference(), run()

### Community 104 - "Community 104"
Cohesion: 0.47
Nodes (3): RetryTests, generate(), generate()

### Community 106 - "Community 106"
Cohesion: 0.60
Nodes (3): current_context(), generate(), validate_selection()

### Community 109 - "Community 109"
Cohesion: 0.50
Nodes (3): generate(), __init__(), generate()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 671 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 6` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 18`, `Community 19`, `Community 20`, `Community 21`, `Community 22`, `Community 23`, `Community 26`, `Community 30`, `Community 32`, `Community 35`, `Community 37`, `Community 38`, `Community 40`, `Community 42`, `Community 47`, `Community 50`, `Community 51`, `Community 52`, `Community 56`, `Community 60`, `Community 65`, `Community 99`, `Community 101`, `Community 102`, `Community 104`, `Community 107`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Are the 46 inferred relationships involving `RunConfig` (e.g. with `Pipeline` and `Budget`) actually correct?**
  _`RunConfig` has 46 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07496194824961948 - nodes in this community are weakly interconnected._
- **Why does `DataInspector` connect `Community 33` to `Community 64`, `Community 34`, `Community 39`, `Community 7`, `Community 9`, `Community 10`, `Community 40`, `Community 11`, `Community 110`, `Community 16`, `Community 52`, `Community 22`, `Community 55`, `Community 28`, `Community 61`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05311871227364185 - nodes in this community are weakly interconnected._