# Graph Report - mlmom-project  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1281 nodes · 4270 edges · 75 communities (61 shown, 14 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 541 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a7d202bb`
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
10. `build_planning_graph()` - 38 edges

## Surprising Connections (you probably didn't know these)
- `InvariantTests` --uses--> `CandidateVersion`  [INFERRED]
  tests/test_invariants.py → mlmom/pipeline_contracts.py
- `InvariantTests` --uses--> `ExperimentRun`  [INFERRED]
  tests/test_invariants.py → mlmom/pipeline_contracts.py
- `validate()` --uses--> `TaskContext`  [INFERRED]
  scripts/run_real_diagnostics.py → mlmom/pipeline_contracts.py
- `ContextAgentTests` --uses--> `TaskContext`  [INFERRED]
  tests/test_context_agent.py → mlmom/pipeline_contracts.py
- `CSVEngineTests` --uses--> `RunConfig`  [INFERRED]
  tests/test_reconnaissance.py → mlmom/pipeline_contracts.py

## Import Cycles
- None detected.

## Communities (75 total, 14 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (53): TableReadOptions, inspect_model(), better(), checked_predictions(), diagnostics(), evaluate(), metric_name(), prediction_kind() (+45 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (21): csv_worker(), _json_value(), _lazy_table(), validate_sql(), extract_json(), Budget, capacity(), terminate_tree() (+13 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (21): default_space(), interrupt_search(), relative_gain(), run_search(), apply_decision(), persist(), request_control(), TrainingDecision (+13 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (6): JsonSerializer, SQLiteCheckpointer, _unsupported(), SkillRegistry, ComponentTests, SkillTests

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (15): CodeBundle, Hypothesis, HypothesisBatch, HypothesisPlan, MLToolSpec, PlanBatch, PreprocessingBundle, SkillManifest (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.14
Nodes (8): atomic_json(), dumps(), plain(), SelectedPreviousResults, Pipeline, finished(), process_jobs(), emit()

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (19): build_planning_graph(), for_hypothesis(), generate(), hypotheses(), is_light(), persist(), plan(), generate() (+11 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (13): DatasetProfile, MetricDirection, ResultFormat, TaskAnalysisResult, TaskDefinition, TaskValidationFinding, TaskValidationReport, ValidationStrategy (+5 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (7): ColumnRole, SemanticType, build_llm_context(), DataInspector, _normalise_name(), DataInspectorTests, write_dictionary()

### Community 9 - "Community 9"
Cohesion: 0.12
Nodes (8): Job, ResourceUnavailable, SandboxManager, trace_event(), AdapterTests, ExecutionTests, budget(), SchedulerIntegrationTests

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (11): ToolResult, ResearchTools, ToolSet, PlanningRecoveryTests, read_skills(), handler(), handler(), handler() (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.16
Nodes (11): RunConfig, SeparatePreprocessingTests, generate(), run(), ClockBudget, Coder, CodingBudgetTests, Sandbox (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (11): ExperimentStore, CheckpointIntegrationTests, ControlledLLM, generate(), CustomCandidatePipeline, InvariantTests, RepairLLM, dataset() (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.14
Nodes (3): current_context(), generate(), load()

### Community 14 - "Community 14"
Cohesion: 0.11
Nodes (8): Candidate, _execute_module(), FitContext, load_module(), create(), load(), Preprocessor, create_candidate()

### Community 15 - "Community 15"
Cohesion: 0.19
Nodes (4): FileRole, TableProfile, InputLayout, _resolve_path()

### Community 16 - "Community 16"
Cohesion: 0.17
Nodes (4): RoleClient, validator_repair(), LLMSettings, AgentProtocolTests

### Community 17 - "Community 17"
Cohesion: 0.16
Nodes (5): OfflineLLM, ContextAgentTests, generate(), generate(), generate()

### Community 18 - "Community 18"
Cohesion: 0.22
Nodes (7): DistributionBucket, _display_value(), _finite_float(), _format_number(), main(), _rate(), _write_json()

### Community 19 - "Community 19"
Cohesion: 0.17
Nodes (6): DeferredTask, ScoutingTests, generate(), generate(), generate(), generate()

### Community 20 - "Community 20"
Cohesion: 0.27
Nodes (4): classification(), LibraryCandidate, stopping_metric(), DataPathTests

### Community 21 - "Community 21"
Cohesion: 0.19
Nodes (10): role_window(), finding_catalog(), prepare_planner_context(), _validate(), compact_diagnostics(), PlannerContextDraft, evaluation_summary(), experiment_summary() (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.15
Nodes (9): DataIssue, DatasetChecks, SplitColumnStats, StrictModel, TaskSpec, PreprocessingSpec, RankedFinding, ResearchProposal (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (6): CSVTools, Observation, ReconnaissanceReport, add(), ReconnaissanceSession, collect()

### Community 24 - "Community 24"
Cohesion: 0.40
Nodes (4): completion(), TraceTests, handler(), handler()

### Community 25 - "Community 25"
Cohesion: 0.18
Nodes (7): request_messages(), StructuredOutputError, traced_node(), bind(), current_sink(), span(), worker()

### Community 26 - "Community 26"
Cohesion: 0.17
Nodes (4): preprocessor(), text_columns(), text_values(), load_candidate()

### Community 27 - "Community 27"
Cohesion: 0.27
Nodes (14): LLMUnavailable, build_coding_graph(), check_source(), coding_capacity(), generate(), generation_remaining(), inspect_api(), preprocessing() (+6 more)

### Community 28 - "Community 28"
Cohesion: 0.19
Nodes (3): load_adapter(), EnsembleCandidate, load_ensemble()

### Community 29 - "Community 29"
Cohesion: 0.37
Nodes (10): ColumnProfile, Severity, build_llm_markdown(), _distribution_text(), _escape(), _number(), _ordered_columns(), _percent() (+2 more)

### Community 30 - "Community 30"
Cohesion: 0.28
Nodes (3): size(), _plain(), TraceSink

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

## Knowledge Gaps
- **1 isolated node(s):** `mlmom`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 416 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataInspector` connect `Community 8` to `Community 32`, `Community 0`, `Community 33`, `Community 1`, `Community 70`, `Community 7`, `Community 10`, `Community 12`, `Community 13`, `Community 15`, `Community 17`, `Community 18`, `Community 19`, `Community 22`, `Community 29`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `DataInspector` (e.g. with `ColumnProfile` and `ColumnRole`) actually correct?**
  _`DataInspector` has 23 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlmom` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0727522306108442 - nodes in this community are weakly interconnected._
- **Why does `Pipeline` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 16`, `Community 17`, `Community 19`, `Community 20`, `Community 21`, `Community 23`, `Community 26`, `Community 27`, `Community 30`, `Community 33`, `Community 65`, `Community 68`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 40 inferred relationships involving `Pipeline` (e.g. with `LibraryCandidate` and `._fit_text()`) actually correct?**
  _`Pipeline` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06329113924050633 - nodes in this community are weakly interconnected._