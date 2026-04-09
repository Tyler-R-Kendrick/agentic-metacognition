---
name: graphrag
description: Build and query Neo4j-backed reasoning trajectory graphs for the hybrid meta-cognition agent. Use this skill when the user wants to persist task plans, subgoals, constraints, run states, and verifier outcomes in a Neo4j graph, retrieve evidence paths with PathRAG, or store drift-correction records. Also trigger when the user mentions Neo4j, GraphRAG, PathRAG, reasoning trajectories, graph store, evidence paths, or task plan graphs — even if they don't say "graphrag" explicitly.
---

# Neo4j PathRAG / GraphRAG Extension

Use this skill to persist and query reasoning trajectories in a Neo4j graph database, enabling path-based retrieval-augmented generation (PathRAG) for the hybrid meta-cognition agent.

## When to use this skill

- Creating `GraphTaskPlan` objects with constraints and subgoals
- Persisting agent runs, states, and verifier outcomes in Neo4j
- Querying evidence paths, analogy paths, and correction paths via `Neo4jPathRAGRetriever`
- Recording drift-correction feedback loops
- Attaching observed interaction features to graph state nodes
- Working with `Neo4jGraphStore` for task/run lifecycle management

## Architecture overview

The GraphRAG extension adds a graph-native persistence and retrieval layer on top of the hybrid agent's existing planner/retriever/executor/verifier loop. It stores:

- **Task nodes** — `GraphTaskPlan` with intent, constraints, subgoals
- **Run nodes** — `GraphRunHandle` linking a run to its task
- **State nodes** — Intermediate agent states with metadata
- **Verifier outcome nodes** — Results from the verifier stage
- **Drift correction edges** — Fallback-triggered corrections between states
- **Interaction feature nodes** — Dynamic features observed during execution

## Key data classes

### GraphTaskPlan
```python
from activation_steering import GraphTaskPlan, GraphConstraint, GraphSubgoal

plan = GraphTaskPlan(
    task_id="task-001",
    intent="Summarize research findings with high truthfulness",
    constraints=[
        GraphConstraint(constraint_id="c1", type="quality", value="truthfulness > 0.8"),
    ],
    subgoals=[
        GraphSubgoal(subgoal_id="s1", description="Extract key findings", order=1),
        GraphSubgoal(subgoal_id="s2", description="Verify claims against sources", order=2),
    ],
)
```

### PathRAG retrieval
```python
from activation_steering import (
    Neo4jPathRAGRetriever,
    PathRAGContext,
    RetrievedPath,
    serialize_path_rag_context,
)

retriever = Neo4jPathRAGRetriever(driver=neo4j_driver)
context = retriever.retrieve(query="truthfulness steering for summarization")
# context.evidence_paths — direct evidence from prior runs
# context.analogy_paths — analogous prior reasoning trajectories
# context.correction_paths — drift-correction records
serialized = serialize_path_rag_context(context)
```

### Neo4jGraphStore
```python
from activation_steering import Neo4jGraphStore, GraphRunHandle

store = Neo4jGraphStore(driver=neo4j_driver)

# Create a task plan
store.create_task(plan)

# Start a run
run_handle = store.start_run(task_id="task-001", run_id="run-001")

# Record states, outcomes, corrections
store.record_state(run_handle, state_data={...})
store.record_verifier_outcome(run_handle, outcome_data={...})
store.record_drift_correction(run_handle, correction_data={...})
```

## Data model

| Node type | Key fields | Purpose |
|-----------|-----------|---------|
| `GraphTaskPlan` | `task_id`, `intent`, `constraints`, `subgoals` | Anchors a reasoning task |
| `GraphRunHandle` | `run_id`, `task_id` | Links runs to tasks |
| `GraphConstraint` | `constraint_id`, `type`, `value` | Quality or behavioral constraints |
| `GraphSubgoal` | `subgoal_id`, `description`, `order` | Ordered objectives within a task |
| `RetrievedPath` | `path_id`, `nodes`, `relationships` | A graph path returned by retrieval |
| `PathRAGContext` | `evidence_paths`, `analogy_paths`, `correction_paths` | Bundled retrieval context |

## Integration with the hybrid agent

The GraphRAG extension pairs with the hybrid agent's existing loop:

1. Before execution, the planner can query `Neo4jPathRAGRetriever` for prior evidence
2. During execution, states are recorded via `Neo4jGraphStore`
3. After verification, outcomes and any drift corrections are persisted
4. Interaction features discovered during the run are attached to state nodes

The extension uses Neo4j's official Python ecosystem (`neo4j` driver, `neo4j-graphrag`) and does not introduce a separate graph framework.

## Working rules

1. Always create a `GraphTaskPlan` before starting runs against it.
2. Use `GraphRunHandle` to scope state and outcome records to a specific run.
3. The `Neo4jPathRAGRetriever` requires an active Neo4j driver connection.
4. Drift corrections record the fallback path — from a divergent state back toward the task's constraints.
5. Use `serialize_path_rag_context` when you need to pass PathRAG context as a string (e.g., for prompt construction).

## API reference

For the full API, read `activation_steering/graphrag.py`.
