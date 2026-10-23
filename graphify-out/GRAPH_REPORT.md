# Graph Report - mlmom-project  (2026-10-23)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1725 nodes · 6113 edges · 96 communities (81 shown, 15 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 759 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9f349056`
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

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 122 edges
2. `DataInspector` - 103 edges
3. `RunConfig` - 101 edges
4. `OfflineLLM` - 62 edges
5. `StrictModel` - 60 edges
6. `atomic_json()` - 57 edges
7. `case()` - 56 edges
8. `build_planning_graph()` - 55 edges
9. `RoleClient` - 53 edges
10. `LLMSettings` - 52 edges

## Surprising Connections (you probably didn't know these)
- `verify_run()` --uses--> `CandidateVersion`  [INFERRED]
  scripts/release_cases.py → mlmom/pipeline_contracts.py
- `InvariantTests` --uses--> `CandidateVersion`  [INFERRED]
  tests/test_invariants.py → mlmom/pipeline_contracts.py
- `verify_run()` --uses--> `ExperimentRun`  [INFERRED]
  scripts/release_cases.py → mlmom/pipeline_contracts.py
- `InvariantTests` --uses--> `ExperimentRun`  [INFERRED]
  tests/test_invariants.py → mlmom/pipeline_contracts.py
- `main()` --uses--> `PlannerContextDraft`  [INFERRED]
  scripts/probe_contract_v2_api.py → mlmom/pipeline_contracts.py

## Import Cycles
- None detected.

## Communities (96 total, 15 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (58): SourcePatch, StrictModel, TableReadOptions, _lazy_table(), inspect_model(), checked_predictions(), compact_diagnostics(), diagnostics() (+50 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (9): atomic_json(), dumps(), plain(), Pipeline, finished(), group_remaining(), process_jobs(), WorkerRuntimeError (+1 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (13): JsonSerializer, SQLiteCheckpointer, _unsupported(), SkillRegistry, check(), render(), RoleSkillTests, generate() (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (8): DataIssue, DatasetChecks, TableProfile, TaskSpec, DataInspector, InputLayout, _normalise_name(), _resolve_path()

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (3): CodeBundle, CodingState, default_space()

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (15): BudgetSnapshot, validate_plan(), MetricsAndSplitTests, generate(), case(), PlanningLimitTests, Skills, PolicyConsistencyTests (+7 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (10): Job, ResourceUnavailable, SandboxManager, trace_event(), AdapterTests, ExecutionTests, NativeValidationTests, CSVEngineTests (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (13): DatasetProfile, MetricDirection, ResultFormat, TaskAnalysisResult, TaskDefinition, TaskValidationFinding, TaskValidationReport, ValidationStrategy (+5 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (14): ExperimentStore, RunConfig, component_pool(), CheckpointIntegrationTests, ControlledLLM, CustomCandidatePipeline, InvariantTests, dataset() (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.09
Nodes (14): compact_planning_payload(), observation_index(), paged_evidence(), ToolResult, ResearchTools, ToolSet, PlanningRecoveryTests, read_skills() (+6 more)

### Community 10 - "Community 10"
Cohesion: 0.10
Nodes (10): CSVTools, metric_capabilities(), Observation, ReconnaissanceReport, data_observations(), add(), ReconnaissanceSession, collect() (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (19): role_window(), constant_space(), normalize_protocol(), HypothesisBatch, PlanBatch, ValidationReport, build_planning_graph(), for_hypothesis() (+11 more)

### Community 12 - "Community 12"
Cohesion: 0.13
Nodes (12): main(), case_config(), digest(), main(), select_real_settings(), verify_observation_handoff(), verify_run(), write() (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (4): request_messages(), RoleClient, LLMSettings, AgentProtocolTests

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (9): SeparatePreprocessingTests, run(), ClockBudget, Coder, CodingBudgetTests, Planner, Sandbox, Store (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (10): finding_catalog(), prepare_planner_context(), _validate(), OfflineLLM, OfflinePipeline, ContextAgentTests, generate(), generate() (+2 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (7): previous_brief_summary(), extract_json(), traced_node(), bind(), current_sink(), span(), worker()

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (21): GlobalAPIError, LLMUnavailable, build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api() (+13 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (8): CodePatch, capacity(), terminate_tree(), watchdog(), main(), index(), main(), validate()

### Community 19 - "Community 19"
Cohesion: 0.14
Nodes (15): plan_ensemble(), read_component_history(), EnsembleProposal, Hypothesis, SearchGroupPlan, collect_groups(), component_brief(), ensemble_recipes() (+7 more)

### Community 20 - "Community 20"
Cohesion: 0.18
Nodes (11): ColumnRole, DistributionBucket, InferenceConfidence, SemanticType, SplitColumnStats, _display_value(), _finite_float(), _format_number() (+3 more)

### Community 21 - "Community 21"
Cohesion: 0.20
Nodes (11): frozen_plan(), hpo_child(), Intervention, PredictionPolicy, TargetTransform, WeightPolicy, with_spec(), HypothesisPlan (+3 more)

### Community 22 - "Community 22"
Cohesion: 0.21
Nodes (7): validator_repair(), Workspace, apply_patch(), read_file(), main(), request(), ContextAndPatchTests

### Community 23 - "Community 23"
Cohesion: 0.15
Nodes (11): run_search(), apply_decision(), persist(), request_control(), TrainingDecision, TrainingControlTests, batch(), initial() (+3 more)

### Community 24 - "Community 24"
Cohesion: 0.12
Nodes (4): PreparedCandidate, Base, MeanPreprocessor, Stack

### Community 25 - "Community 25"
Cohesion: 0.29
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 26 - "Community 26"
Cohesion: 0.17
Nodes (11): compare_roundtrip(), complete_export(), _component_objects(), _equal(), fitted_components(), inspect_component(), _json_params(), _leaves() (+3 more)

### Community 27 - "Community 27"
Cohesion: 0.27
Nodes (4): classification(), LibraryCandidate, stopping_metric(), DataPathTests

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (5): interrupt_search(), relative_gain(), features(), targets(), PersistentSearchTests

### Community 29 - "Community 29"
Cohesion: 0.18
Nodes (6): cases(), evidence(), gate(), main(), package_info(), ReleaseGateTests

### Community 30 - "Community 30"
Cohesion: 0.20
Nodes (5): ScoutingTests, generate(), generate(), generate(), generate()

### Community 31 - "Community 31"
Cohesion: 0.18
Nodes (11): capture(), fingerprint(), install(), importing(), patch_loaded(), visit(), _patch(), captured() (+3 more)

### Community 32 - "Community 32"
Cohesion: 0.19
Nodes (7): inspect(), package(), main(), archive_hash(), main(), names(), FlatPackageTests

### Community 33 - "Community 33"
Cohesion: 0.16
Nodes (11): csv_worker(), _json_value(), better(), apply_overrides(), TrialResult, failure_result(), fit_context(), main() (+3 more)

### Community 34 - "Community 34"
Cohesion: 0.22
Nodes (7): tree_hash(), _execute_module(), load_module(), create(), load(), composite_case(), CompositeTests

### Community 35 - "Community 35"
Cohesion: 0.12
Nodes (3): Candidate, FitContext, Preprocessor

### Community 37 - "Community 37"
Cohesion: 0.23
Nodes (4): size(), _plain(), TraceSink, ProviderFailureTests

### Community 38 - "Community 38"
Cohesion: 0.21
Nodes (9): TrainingCheckpoint, commit_mailbox(), decide(), fallback(), validate_decision(), worker_checkpoint(), control(), search_checkpoint() (+1 more)

### Community 39 - "Community 39"
Cohesion: 0.17
Nodes (4): preprocessor(), text_columns(), text_values(), load_candidate()

### Community 40 - "Community 40"
Cohesion: 0.29
Nodes (3): audit_fit(), ExecutionContractTests, FittedCandidate

### Community 43 - "Community 43"
Cohesion: 0.19
Nodes (3): load_adapter(), EnsembleCandidate, load_ensemble()

### Community 44 - "Community 44"
Cohesion: 0.23
Nodes (7): comparable_scores(), relative_gain(), config_hash(), require_admitted(), ResultReview, review_result(), ResearchTests

### Community 45 - "Community 45"
Cohesion: 0.30
Nodes (3): train(), InnerExecutionTests, TargetEncodingBoundaryTests

### Community 46 - "Community 46"
Cohesion: 0.37
Nodes (10): ColumnProfile, Severity, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent() (+2 more)

### Community 47 - "Community 47"
Cohesion: 0.23
Nodes (4): FileRole, build_llm_context(), DataInspectorTests, write_dictionary()

### Community 48 - "Community 48"
Cohesion: 0.27
Nodes (3): NetworkCircuit, NetworkPolicyTests, handler()

### Community 49 - "Community 49"
Cohesion: 0.22
Nodes (3): capability_matrix(), Budget, OptionalTimeoutTests

### Community 52 - "Community 52"
Cohesion: 0.22
Nodes (3): FidelityError, smoke(), ReviewRegressionTests

### Community 53 - "Community 53"
Cohesion: 0.32
Nodes (3): runtime_versions(), worker_runtime(), WorkerRuntimeTests

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

### Community 88 - "Community 88"
Cohesion: 0.60
Nodes (3): current_context(), generate(), load()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 530 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataInspector` connect `Community 3` to `Community 0`, `Community 33`, `Community 4`, `Community 7`, `Community 8`, `Community 9`, `Community 46`, `Community 47`, `Community 15`, `Community 51`, `Community 20`, `Community 91`, `Community 30`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 48 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 48 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.051274747187010396 - nodes in this community are weakly interconnected._
- **Why does `Pipeline` connect `Community 1` to `Community 0`, `Community 2`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 17`, `Community 18`, `Community 19`, `Community 21`, `Community 23`, `Community 27`, `Community 28`, `Community 30`, `Community 37`, `Community 38`, `Community 39`, `Community 44`, `Community 49`, `Community 52`, `Community 53`, `Community 85`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `DataInspector` (e.g. with `ColumnProfile` and `ColumnRole`) actually correct?**
  _`DataInspector` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.10520163646990065 - nodes in this community are weakly interconnected._