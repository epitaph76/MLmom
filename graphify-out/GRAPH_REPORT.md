# Graph Report - mlmom-project  (2026-10-23)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1525 nodes · 5282 edges · 84 communities (70 shown, 14 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 654 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `62fff23c`
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

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 117 edges
2. `DataInspector` - 103 edges
3. `RunConfig` - 93 edges
4. `OfflineLLM` - 59 edges
5. `atomic_json()` - 57 edges
6. `StrictModel` - 50 edges
7. `SkillRegistry` - 49 edges
8. `SandboxManager` - 48 edges
9. `case()` - 48 edges
10. `RoleClient` - 45 edges

## Surprising Connections (you probably didn't know these)
- `SchedulerIntegrationTests` --uses--> `Job`  [INFERRED]
  tests/test_scheduler_integration.py → mlmom/resources.py
- `ReleaseGateTests` --uses--> `LLMSettings`  [INFERRED]
  tests/test_release_gate.py → mlmom/llm_client.py
- `ReleaseGateTests` --uses--> `RunConfig`  [INFERRED]
  tests/test_release_gate.py → mlmom/pipeline_contracts.py
- `GeneratedCandidateTests` --uses--> `SkillRegistry`  [INFERRED]
  tests/test_generated_candidates.py → mlmom/registry.py
- `InvariantTests` --uses--> `SkillRegistry`  [INFERRED]
  tests/test_invariants.py → mlmom/registry.py

## Import Cycles
- None detected.

## Communities (84 total, 14 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (32): resolve_inputs(), find(), capacity(), Job, runtime_versions(), terminate_tree(), inspect(), package() (+24 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (16): SkillRegistry, Workspace, check(), render(), RoleSkillTests, generate(), generate(), DecisionSkillCardTests (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (24): default_space(), interrupt_search(), relative_gain(), run_search(), apply_decision(), persist(), request_control(), TrainingCheckpoint (+16 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (17): role_window(), CSVTools, comparable_scores(), relative_gain(), metric_capabilities(), Observation, ReconnaissanceReport, ResultReview (+9 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (16): InferenceConfidence, MetricDirection, ResultFormat, Severity, TaskAnalysisResult, TaskDefinition, TaskType, TaskValidationFinding (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (7): atomic_json(), dumps(), plain(), Pipeline, finished(), process_jobs(), report()

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (10): classification(), LibraryCandidate, load_adapter(), preprocessor(), text_columns(), text_values(), EnsembleCandidate, load_ensemble() (+2 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (15): ExperimentStore, RunConfig, Budget, CheckpointIntegrationTests, ControlledLLM, generate(), CustomCandidatePipeline, InvariantTests (+7 more)

### Community 9 - "Community 9"
Cohesion: 0.10
Nodes (15): audit_fit(), compare_roundtrip(), complete_export(), _component_objects(), config_hash(), _equal(), FidelityError, fitted_components() (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.10
Nodes (8): validate_sql(), ResourceUnavailable, SandboxManager, trace_event(), AdapterTests, ExecutionTests, CSVEngineTests, budget()

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (14): GlobalAPIError, LLMUnavailable, finding_catalog(), prepare_planner_context(), _validate(), compact_diagnostics(), PlannerContextDraft, RunState (+6 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (8): build_planning_graph(), MetricsAndSplitTests, generate(), case(), PlanningLimitTests, generate(), Skills, generate()

### Community 13 - "Community 13"
Cohesion: 0.14
Nodes (7): ColumnRole, SemanticType, SplitColumnStats, DataInspector, _normalise_name(), DataInspectorTests, write_dictionary()

### Community 14 - "Community 14"
Cohesion: 0.14
Nodes (5): RoleClient, LLMSettings, RouterAIChatClient, AgentProtocolTests, ProviderFailureTests

### Community 15 - "Community 15"
Cohesion: 0.15
Nodes (8): SeparatePreprocessingTests, run(), ClockBudget, Coder, CodingBudgetTests, Sandbox, Store, ShortCycleTests

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (12): ToolResult, for_hypothesis(), generate(), hypotheses(), is_light(), persist(), plan(), ResearchTools (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.16
Nodes (19): StrictModel, inside(), CandidateVersion, Champion, ExperimentRun, Hypothesis, HypothesisPlan, InputFiles (+11 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (9): OfflineLLM, OfflinePipeline, ContextAgentTests, generate(), generate(), generate(), broken_code(), RepairLLM (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.14
Nodes (11): checked_predictions(), _execute_module(), load_module(), create(), load(), main(), predict(), smoke() (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.19
Nodes (11): features(), targets(), SplitSpec, TaskContext, group_keys(), inner_protocol(), make_split(), split_indices() (+3 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (3): JsonSerializer, SQLiteCheckpointer, _unsupported()

### Community 22 - "Community 22"
Cohesion: 0.17
Nodes (4): FileRole, TableProfile, InputLayout, _resolve_path()

### Community 23 - "Community 23"
Cohesion: 0.17
Nodes (15): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+7 more)

### Community 24 - "Community 24"
Cohesion: 0.22
Nodes (8): DistributionBucket, build_llm_context(), _display_value(), _finite_float(), _format_number(), main(), _rate(), _write_json()

### Community 25 - "Community 25"
Cohesion: 0.13
Nodes (4): PreparedCandidate, Base, MeanPreprocessor, Stack

### Community 26 - "Community 26"
Cohesion: 0.18
Nodes (7): validator_repair(), traced_node(), bind(), current_sink(), emit(), span(), worker()

### Community 27 - "Community 27"
Cohesion: 0.21
Nodes (7): better(), tree_hash(), fit_context(), train(), objective(), InnerExecutionTests, TargetEncodingBoundaryTests

### Community 28 - "Community 28"
Cohesion: 0.20
Nodes (9): TableReadOptions, csv_worker(), _json_value(), _lazy_table(), detect_table_format(), read_table(), validate_tables(), file_hash() (+1 more)

### Community 29 - "Community 29"
Cohesion: 0.26
Nodes (11): stopping_metric(), inspect_model(), diagnostics(), error_metrics(), evaluate(), metric_name(), prediction_kind(), score_predictions() (+3 more)

### Community 30 - "Community 30"
Cohesion: 0.19
Nodes (3): size(), _plain(), TraceSink

### Community 31 - "Community 31"
Cohesion: 0.15
Nodes (3): Candidate, create_candidate(), load_candidate()

### Community 32 - "Community 32"
Cohesion: 0.24
Nodes (5): HypothesisBatch, PlanBatch, ValidationReport, RepairingPlanningClient, Planner

### Community 33 - "Community 33"
Cohesion: 0.16
Nodes (3): request_messages(), StructuredOutputError, extract_json()

### Community 34 - "Community 34"
Cohesion: 0.21
Nodes (5): ScoutingTests, generate(), generate(), generate(), generate()

### Community 35 - "Community 35"
Cohesion: 0.30
Nodes (4): ColumnProfile, DataIssue, DatasetChecks, TaskSpec

### Community 36 - "Community 36"
Cohesion: 0.36
Nodes (9): DatasetProfile, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent(), _render() (+1 more)

### Community 37 - "Community 37"
Cohesion: 0.32
Nodes (7): RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 38 - "Community 38"
Cohesion: 0.38
Nodes (4): BudgetSnapshot, validate_plan(), installed_tools(), NativeValidationTests

### Community 41 - "Community 41"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 42 - "Community 42"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 43 - "Community 43"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 44 - "Community 44"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 45 - "Community 45"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 46 - "Community 46"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 47 - "Community 47"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 48 - "Community 48"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 49 - "Community 49"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 50 - "Community 50"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 51 - "Community 51"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 52 - "Community 52"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 53 - "Community 53"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

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

### Community 73 - "Community 73"
Cohesion: 0.33
Nodes (6): generate(), generate(), generate(), generate(), generate(), generate()

### Community 76 - "Community 76"
Cohesion: 0.60
Nodes (3): current_context(), generate(), load()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 482 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataInspector` connect `Community 13` to `Community 34`, `Community 35`, `Community 36`, `Community 4`, `Community 6`, `Community 40`, `Community 8`, `Community 11`, `Community 77`, `Community 16`, `Community 18`, `Community 19`, `Community 22`, `Community 24`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 47 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05070707070707071 - nodes in this community are weakly interconnected._
- **Why does `Pipeline` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 19`, `Community 20`, `Community 21`, `Community 29`, `Community 30`, `Community 31`, `Community 34`, `Community 36`, `Community 71`, `Community 75`, `Community 78`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `DataInspector` (e.g. with `ColumnProfile` and `ColumnRole`) actually correct?**
  _`DataInspector` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06892010535557506 - nodes in this community are weakly interconnected._