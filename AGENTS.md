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
