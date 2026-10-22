# Graph Report - mlmom-project  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1212 nodes · 4204 edges · 64 communities (55 shown, 9 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 538 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f7183c40`
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

## God Nodes (most connected - your core abstractions)
1. `DataInspector` - 103 edges
2. `Pipeline` - 101 edges
3. `RunConfig` - 74 edges
4. `atomic_json()` - 51 edges
5. `StrictModel` - 50 edges
6. `OfflineLLM` - 44 edges
7. `SandboxManager` - 40 edges
8. `RoleClient` - 39 edges
9. `read_table()` - 39 edges
10. `case()` - 38 edges

## Surprising Connections (you probably didn't know these)
- `DataPathTests` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_improvement.py → mlmom/persistence.py
- `RealFailureRegressions` --uses--> `ExperimentStore`  [INFERRED]
  tests/test_real_failure_regressions.py → mlmom/persistence.py
- `main()` --uses--> `RunConfig`  [INFERRED]
  solution.py → mlmom/pipeline_contracts.py
- `SeparatePreprocessingTests` --uses--> `RunConfig`  [INFERRED]
  tests/test_agent_roles.py → mlmom/pipeline_contracts.py
- `DataPathTests` --uses--> `RunConfig`  [INFERRED]
  tests/test_improvement.py → mlmom/pipeline_contracts.py

## Import Cycles
- None detected.

## Communities (64 total, 9 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (29): validate_tables(), dumps(), ExperimentStore, plain(), RunConfig, Pipeline, finished(), process_jobs() (+21 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (20): RoleClient, validator_repair(), size(), LLMSettings, RouterAIChatClient, traced_node(), _plain(), span() (+12 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (3): request_messages(), extract_json(), current_sink()

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (9): Job, ResourceUnavailable, SandboxManager, trace_event(), AdapterTests, ExecutionTests, CSVEngineTests, budget() (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (16): DatasetProfile, InferenceConfidence, MetricDirection, ResultFormat, Severity, TaskAnalysisResult, TaskDefinition, TaskType (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (15): CodeBundle, CodingState, BudgetSnapshot, Hypothesis, HypothesisBatch, HypothesisPlan, PlanBatch, ValidationReport (+7 more)

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (12): generate(), run(), case(), ClockBudget, Coder, CodingBudgetTests, Sandbox, Store (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (5): SkillRegistry, Workspace, RoleSkillTests, ComponentTests, SkillTests

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (6): ColumnProfile, ColumnRole, DataIssue, DatasetChecks, TaskSpec, _normalise_name()

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (15): build_planning_graph(), for_hypothesis(), generate(), hypotheses(), is_light(), persist(), plan(), bind() (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (6): CSVTools, ToolResult, ResearchTools, ToolSet, PlanningRecoveryTests, read_skills()

### Community 11 - "Community 11"
Cohesion: 0.20
Nodes (21): better(), features(), targets(), tree_hash(), InputFiles, TaskContext, TrialResult, group_keys() (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.17
Nodes (5): FileRole, TableProfile, DataInspector, InputLayout, _resolve_path()

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (9): default_space(), interrupt_search(), relative_gain(), run_search(), apply_decision(), persist(), request_control(), resumed() (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.14
Nodes (15): TrainingCheckpoint, TrainingDecision, commit_mailbox(), decide(), fallback(), validate_decision(), worker_checkpoint(), control() (+7 more)

### Community 15 - "Community 15"
Cohesion: 0.20
Nodes (6): classification(), LibraryCandidate, load_adapter(), text_columns(), FitContext, DataPathTests

### Community 16 - "Community 16"
Cohesion: 0.17
Nodes (6): installed_tools(), main(), main(), main(), main(), main()

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (3): JsonSerializer, SQLiteCheckpointer, _unsupported()

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (9): Observation, ReconnaissanceReport, ResultReview, data_observations(), add(), ReconnaissanceSession, collect(), review_result() (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.22
Nodes (10): stopping_metric(), inspect_model(), checked_predictions(), diagnostics(), evaluate(), metric_name(), prediction_kind(), score_predictions() (+2 more)

### Community 20 - "Community 20"
Cohesion: 0.22
Nodes (8): DistributionBucket, SplitColumnStats, _display_value(), _finite_float(), _format_number(), main(), _rate(), _write_json()

### Community 21 - "Community 21"
Cohesion: 0.20
Nodes (7): atomic_json(), Budget, capacity(), terminate_tree(), index(), main(), validate()

### Community 22 - "Community 22"
Cohesion: 0.18
Nodes (15): StrictModel, CandidateVersion, Champion, EvaluationResult, ExperimentRun, MLToolSpec, PreprocessingBundle, PreprocessingSpec (+7 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (7): csv_worker(), _json_value(), _lazy_table(), validate_sql(), file_hash(), report(), run()

### Community 24 - "Community 24"
Cohesion: 0.15
Nodes (7): _execute_module(), load_module(), create(), load(), Candidate, create_candidate(), SeparatePreprocessingTests

### Community 25 - "Community 25"
Cohesion: 0.18
Nodes (7): compact_diagnostics(), DeferredTask, RunState, evaluation_summary(), experiment_summary(), plan_summary(), capability_matrix()

### Community 26 - "Community 26"
Cohesion: 0.14
Nodes (10): LLMUnavailable, StructuredOutputError, generate(), generate(), generate(), generate(), generate(), generate() (+2 more)

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (4): SemanticType, build_llm_context(), DataInspectorTests, write_dictionary()

### Community 29 - "Community 29"
Cohesion: 0.21
Nodes (3): preprocessor(), text_values(), load_candidate()

### Community 30 - "Community 30"
Cohesion: 0.27
Nodes (6): TableReadOptions, detect_table_format(), inside(), read_table(), resolve_inputs(), find()

### Community 31 - "Community 31"
Cohesion: 0.35
Nodes (8): build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent(), _render(), _target_matrix()

### Community 33 - "Community 33"
Cohesion: 0.35
Nodes (12): build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing(), preprocessing_failed() (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (6): role_window(), finding_catalog(), prepare_planner_context(), _validate(), PlannerContextDraft, profile_summary()

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

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 358 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataInspector` connect `Community 12` to `Community 0`, `Community 2`, `Community 35`, `Community 4`, `Community 8`, `Community 10`, `Community 11`, `Community 20`, `Community 28`, `Community 61`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `DataInspector` (e.g. with `ColumnProfile` and `ColumnRole`) actually correct?**
  _`DataInspector` has 23 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.053165290453426044 - nodes in this community are weakly interconnected._
- **Why does `Pipeline` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 10`, `Community 11`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 21`, `Community 22`, `Community 24`, `Community 25`, `Community 26`, `Community 29`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 40 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.062456140350877196 - nodes in this community are weakly interconnected._