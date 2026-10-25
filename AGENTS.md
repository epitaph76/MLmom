# Project navigation

The current role and checkpoint contracts are documented in docs/AGENT_ROLES.md.
Skill ownership and missing skill placeholders are listed in docs/SKILLS.md.

For architecture or dependency questions, consult the local Graphify graph first:

```powershell
.\.tools\graphify-env\Scripts\graphify.exe query "specific symbol or relationship" --graph graphify-out/graph.json
.\.tools\graphify-env\Scripts\graphify.exe explain PreparedCandidate --graph graphify-out/graph.json
```

Use source files to verify exact behavior. If the graph is stale or unavailable, read
the relevant source and refresh with `.\.venv\Scripts\python.exe scripts/update_graph.py`.
Graphify is a development tool, separate from the ML runtime and submission bundle.
Its pinned source and environment live in ignored .tools/; setup is in docs/GRAPHIFY.md.

# Runtime contracts and skills

When changing execution behavior, update the affected role skills, REFERENCE.md examples,
manifest versions and runtime documentation in the same change. Audit the actual role
payload as well as the skill text; ownership facts come from mlmom/execution_contract.py.
Keep examples executable against the pinned dependencies. Do not reintroduce estimator
audits, provenance gates or mandatory smoke checks to compensate for a prompt mismatch.
Use new run directories after a runtime contract version change; preserve old evidence.

Search-group membership is controller-owned: expose PlannerPlan to agents and bind
HypothesisPlan variants through their containing SearchGroupPlan before review.
Never restore agent-authored group labels as an experiment refusal. Local recipe
failures retain diagnostics and must not discard independent accepted recipes.

Historical duration estimates are advisory, never experiment admission gates. Preserve
actual deadlines, cancellation and shared CPU/RAM scheduling. Feature replacements are
one final-set comparison; never require a removal-only improvement before measuring them.

Feature Agent may write fold-fitted Python programs with installed dependencies. DSL is
optional, not a capability boundary; supervised features are allowed with correct label
scope. Preserve fold isolation, serialized state and final-set comparisons, without new audits.
