# Graph Report - mlmom-project  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1296 nodes · 4331 edges · 73 communities (65 shown, 8 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 550 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1f87577d`
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

## God Nodes (most connected - your core abstractions)
1. `Pipeline` - 106 edges
2. `DataInspector` - 103 edges
3. `RunConfig` - 77 edges
4. `atomic_json()` - 51 edges
5. `StrictModel` - 50 edges
6. `OfflineLLM` - 47 edges
7. `SandboxManager` - 40 edges
8. `RoleClient` - 39 edges
9. `read_table()` - 39 edges
10. `build_planning_graph()` - 38 edges

## Surprising Connections (you probably didn't know these)
- `ContextAgentTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_context_agent.py → mlmom/persistence.py
- `DataPathTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_improvement.py → mlmom/persistence.py
- `RealFailureRegressions` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_real_failure_regressions.py → mlmom/persistence.py
- `ScoutingTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_reconnaissance.py → mlmom/persistence.py
- `Candidate` --uses--> `Pipeline`  [INFERRED]
  tests/fixtures/legacy_candidate.py → mlmom/pipeline.py

## Import Cycles
- None detected.

## Communities (73 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (25): compact_diagnostics(), validate_tables(), ExperimentStore, DeferredTask, Pipeline, finished(), process_jobs(), evaluation_summary() (+17 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (20): BudgetSnapshot, HypothesisBatch, Job, ResourceUnavailable, SandboxManager, trace_event(), worker_runtime(), OfflineLLM (+12 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (16): classification(), LibraryCandidate, stopping_metric(), load_adapter(), preprocessor(), text_columns(), text_values(), EnsembleCandidate (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (12): DatasetProfile, MetricDirection, TaskAnalysisResult, TaskDefinition, TaskValidationFinding, TaskValidationReport, ValidationStrategy, __getattr__() (+4 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (20): role_window(), Observation, TrainingCheckpoint, TrainingDecision, RunState, data_observations(), add(), review_result() (+12 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (12): build_planning_graph(), GeneratedCandidateTests, generate(), case(), PlanningLimitTests, Skills, generate(), generate() (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (7): TableProfile, DataInspector, InputLayout, _normalise_name(), _resolve_path(), DataInspectorTests, write_dictionary()

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (7): runtime_versions(), package(), main(), main(), main(), main(), FlatPackageTests

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (9): CSVTools, ReconnaissanceReport, ReconnaissanceSession, collect(), ScoutingTests, generate(), generate(), generate() (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (10): RunConfig, SeparatePreprocessingTests, generate(), run(), ClockBudget, Coder, CodingBudgetTests, Sandbox (+2 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (11): LLMUnavailable, request_messages(), StructuredOutputError, traced_node(), PlanningState, bind(), current_sink(), span() (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (23): better(), checked_predictions(), features(), targets(), tree_hash(), SplitSpec, TaskContext, TrialResult (+15 more)

### Community 12 - "Community 12"
Cohesion: 0.14
Nodes (9): csv_worker(), _json_value(), _lazy_table(), validate_sql(), detect_table_format(), inside(), resolve_inputs(), find() (+1 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (9): default_space(), interrupt_search(), relative_gain(), run_search(), apply_decision(), persist(), request_control(), resumed() (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (11): ToolResult, for_hypothesis(), generate(), hypotheses(), is_light(), persist(), plan(), ResearchTools (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.15
Nodes (4): RoleClient, validator_repair(), LLMSettings, AgentProtocolTests

### Community 16 - "Community 16"
Cohesion: 0.18
Nodes (10): atomic_json(), dumps(), plain(), Budget, capacity(), terminate_tree(), report(), index() (+2 more)

### Community 17 - "Community 17"
Cohesion: 0.19
Nodes (8): CandidateVersion, ExperimentRun, Hypothesis, HypothesisPlan, validate_plan(), installed_tools(), publish_submission(), execution_case()

### Community 18 - "Community 18"
Cohesion: 0.15
Nodes (3): CodeBundle, CodingState, verify_source()

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (4): ColumnRole, DataIssue, DatasetChecks, TaskSpec

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (4): JsonSerializer, SQLiteCheckpointer, _unsupported(), ComponentTests

### Community 21 - "Community 21"
Cohesion: 0.29
Nodes (5): RouterAIChatClient, completion(), TraceTests, handler(), handler()

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (5): _execute_module(), load_module(), create(), load(), PreparedCandidate

### Community 23 - "Community 23"
Cohesion: 0.28
Nodes (4): DistributionBucket, _display_value(), _finite_float(), _rate()

### Community 24 - "Community 24"
Cohesion: 0.21
Nodes (9): FileRole, InferenceConfidence, SemanticType, SplitColumnStats, TaskType, build_llm_context(), _format_number(), main() (+1 more)

### Community 25 - "Community 25"
Cohesion: 0.21
Nodes (14): StrictModel, TableReadOptions, Champion, InputFiles, MLToolSpec, PreprocessingBundle, PreprocessingSpec, RankedFinding (+6 more)

### Community 26 - "Community 26"
Cohesion: 0.26
Nodes (8): inspect_model(), diagnostics(), evaluate(), metric_name(), prediction_kind(), score_predictions(), EvaluationResult, error_slices()

### Community 27 - "Community 27"
Cohesion: 0.17
Nodes (3): read_table(), capability_matrix(), main()

### Community 29 - "Community 29"
Cohesion: 0.33
Nodes (10): ColumnProfile, Severity, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent() (+2 more)

### Community 30 - "Community 30"
Cohesion: 0.28
Nodes (3): size(), _plain(), TraceSink

### Community 32 - "Community 32"
Cohesion: 0.35
Nodes (12): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+4 more)

### Community 33 - "Community 33"
Cohesion: 0.26
Nodes (5): finding_catalog(), prepare_planner_context(), _validate(), PlannerContextDraft, profile_summary()

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (7): RealFailureRegressions, handler(), handler(), handler(), handler(), response(), tool_call()

### Community 35 - "Community 35"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 36 - "Community 36"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 37 - "Community 37"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 38 - "Community 38"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

### Community 39 - "Community 39"
Cohesion: 0.25
Nodes (7): dependencies, ml_tools, skill_id, status, task_types, title, version

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

### Community 67 - "Community 67"
Cohesion: 0.60
Nodes (3): PlanBatch, ValidationReport, Planner

### Community 68 - "Community 68"
Cohesion: 0.60
Nodes (3): current_context(), generate(), load()

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 421 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 20`, `Community 25`, `Community 26`, `Community 27`, `Community 30`, `Community 33`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 42 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 42 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05438165438165438 - nodes in this community are weakly interconnected._
- **Why does `DataInspector` connect `Community 6` to `Community 0`, `Community 33`, `Community 1`, `Community 3`, `Community 8`, `Community 11`, `Community 12`, `Community 14`, `Community 18`, `Community 19`, `Community 23`, `Community 24`, `Community 28`, `Community 29`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `DataInspector` (e.g. with `ColumnProfile` and `ColumnRole`) actually correct?**
  _`DataInspector` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.050949367088607596 - nodes in this community are weakly interconnected._