# Graph Report - mlmom-project  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1419 nodes · 4882 edges · 79 communities (65 shown, 14 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 610 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `49b38a57`
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

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 112 edges
2. `DataInspector` - 103 edges
3. `RunConfig` - 90 edges
4. `OfflineLLM` - 57 edges
5. `atomic_json()` - 51 edges
6. `StrictModel` - 50 edges
7. `SkillRegistry` - 49 edges
8. `SandboxManager` - 48 edges
9. `case()` - 45 edges
10. `read_table()` - 43 edges

## Surprising Connections (you probably didn't know these)
- `ContextAgentTests` --uses--> `DataIssue`  [INFERRED]
  tests/test_context_agent.py → mlmom/contracts.py
- `ContextAgentTests` --uses--> `DataInspector`  [INFERRED]
  tests/test_context_agent.py → mlmom/data_inspector.py
- `PlanningRecoveryTests` --uses--> `DataInspector`  [INFERRED]
  tests/test_planning_recovery.py → mlmom/data_inspector.py
- `ScoutingTests` --uses--> `DataInspector`  [INFERRED]
  tests/test_reconnaissance.py → mlmom/data_inspector.py
- `StageOneRegressions` --uses--> `DataInspector`  [INFERRED]
  tests/test_stage1_regressions.py → mlmom/data_inspector.py

## Import Cycles
- None detected.

## Communities (79 total, 14 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (19): ColumnProfile, ColumnRole, DataIssue, DatasetChecks, DistributionBucket, SplitColumnStats, TableProfile, TaskSpec (+11 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (28): atomic_json(), capacity(), Job, terminate_tree(), inspect(), package(), case_config(), digest() (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (18): classification(), LibraryCandidate, stopping_metric(), load_adapter(), preprocessor(), text_columns(), text_values(), EnsembleCandidate (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (17): ExperimentStore, RunConfig, Budget, CheckpointIntegrationTests, ControlledLLM, CustomCandidatePipeline, InvariantTests, RepairLLM (+9 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (18): DatasetProfile, TaskDefinition, TaskValidationFinding, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns() (+10 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (10): BudgetSnapshot, installed_tools(), ResourceUnavailable, SandboxManager, trace_event(), AdapterTests, ExecutionTests, NativeValidationTests (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (17): build_planning_graph(), for_hypothesis(), hypotheses(), is_light(), persist(), plan(), validate_plan(), generate() (+9 more)

### Community 7 - "Community 7"
Cohesion: 0.09
Nodes (11): SkillRegistry, review_result(), check(), render(), ResearchTests, RoleSkillTests, generate(), generate() (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (5): dumps(), plain(), DeferredTask, finished(), process_jobs()

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (14): FileRole, InferenceConfidence, MetricDirection, ResultFormat, SemanticType, Severity, TaskAnalysisResult, TaskType (+6 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (8): CodeBundle, CodingState, HypothesisPlan, ValidationReport, PlanningState, verify_source(), GeneratedCandidateTests, PolicyConsistencyTests

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (15): TableReadOptions, detect_table_format(), inside(), read_table(), resolve_inputs(), find(), validate_tables(), file_hash() (+7 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (6): ToolResult, generate(), ResearchTools, ToolSet, PlanningRecoveryTests, read_skills()

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (8): SeparatePreprocessingTests, run(), ClockBudget, Coder, CodingBudgetTests, Sandbox, Store, ShortCycleTests

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (4): JsonSerializer, SQLiteCheckpointer, _unsupported(), ComponentTests

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (6): LLMUnavailable, request_messages(), generate(), generate(), completion(), handler()

### Community 17 - "Community 17"
Cohesion: 0.17
Nodes (6): OfflineLLM, OfflinePipeline, ContextAgentTests, generate(), generate(), generate()

### Community 18 - "Community 18"
Cohesion: 0.19
Nodes (9): better(), tree_hash(), TrialResult, fit_context(), train(), objective(), save_progress(), InnerExecutionTests (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.23
Nodes (9): targets(), SplitSpec, TaskContext, group_keys(), inner_protocol(), make_split(), split_indices(), execution_case() (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.14
Nodes (4): PreparedCandidate, Base, MeanPreprocessor, Stack

### Community 21 - "Community 21"
Cohesion: 0.17
Nodes (10): role_window(), finding_catalog(), prepare_planner_context(), _validate(), compact_diagnostics(), PlannerContextDraft, evaluation_summary(), experiment_summary() (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.25
Nodes (9): inspect_model(), checked_predictions(), diagnostics(), evaluate(), metric_name(), prediction_kind(), score_predictions(), error_slices() (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.24
Nodes (8): RoleClient, validator_repair(), traced_node(), bind(), current_sink(), emit(), span(), worker()

### Community 25 - "Community 25"
Cohesion: 0.21
Nodes (9): TrainingCheckpoint, commit_mailbox(), decide(), fallback(), validate_decision(), worker_checkpoint(), control(), search_checkpoint() (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.17
Nodes (9): runtime_versions(), load_module(), create(), load(), main(), predict(), smoke(), synthetic() (+1 more)

### Community 27 - "Community 27"
Cohesion: 0.21
Nodes (14): StrictModel, Champion, EvaluationResult, InputFiles, MLToolSpec, PreprocessingBundle, PreprocessingSpec, RankedFinding (+6 more)

### Community 28 - "Community 28"
Cohesion: 0.21
Nodes (6): Hypothesis, HypothesisBatch, PlanBatch, RepairingPlanningClient, broken_code(), Planner

### Community 29 - "Community 29"
Cohesion: 0.20
Nodes (5): ScoutingTests, generate(), generate(), generate(), generate()

### Community 30 - "Community 30"
Cohesion: 0.21
Nodes (7): csv_worker(), _json_value(), _lazy_table(), validate_sql(), Observation, data_observations(), add()

### Community 32 - "Community 32"
Cohesion: 0.18
Nodes (6): default_space(), interrupt_search(), run_search(), apply_decision(), persist(), request_control()

### Community 34 - "Community 34"
Cohesion: 0.31
Nodes (13): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+5 more)

### Community 35 - "Community 35"
Cohesion: 0.22
Nodes (7): TrainingDecision, TrainingControlTests, batch(), initial(), objective(), control(), resumed()

### Community 36 - "Community 36"
Cohesion: 0.28
Nodes (3): size(), _plain(), TraceSink

### Community 38 - "Community 38"
Cohesion: 0.29
Nodes (7): RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 40 - "Community 40"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

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

### Community 74 - "Community 74"
Cohesion: 0.60
Nodes (3): current_context(), generate(), load()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 448 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 23` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 19`, `Community 21`, `Community 24`, `Community 25`, `Community 27`, `Community 29`, `Community 30`, `Community 31`, `Community 35`, `Community 36`, `Community 71`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 44 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 44 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06118421052631579 - nodes in this community are weakly interconnected._
- **Why does `DataInspector` connect `Community 0` to `Community 3`, `Community 4`, `Community 9`, `Community 10`, `Community 13`, `Community 16`, `Community 17`, `Community 21`, `Community 26`, `Community 29`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `DataInspector` (e.g. with `ColumnProfile` and `ColumnRole`) actually correct?**
  _`DataInspector` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.053235653235653234 - nodes in this community are weakly interconnected._