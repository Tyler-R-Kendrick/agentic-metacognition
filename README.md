# Agentic Meta-Cognition

Meta-layers to extract and reason about LLM cognitive patterns with agents to enhance reasoning quality.

The purpose of this project is to demonstrate the utility of an externalized meta-cognition layer that allows agents to reflect on and reason about their models' activations, and to self-steer, with the goal of enhancing reasoning quality.

I seek to achieve this by extracting vocalized/labeled features from activation sequences obtained through agent usage. Once obtained, provenance data for the model, activations, and the steering vector are mapped to a dynamically constructed graph. The graph will then be used to augment reasoning trajectories in a reflective loop.

## Potential Outcomes

### Model "Theory-of-Mind"

By modeling many model "features" in a graph, I hope to have the agent adapt its responses to account for the known reasoning strategies of the other models - and to better navigate the internal reasoning of models not present in the training set. Hopefully, this could reveal a common set of utility features that are independent of the models.

### Self-adaptive Meta-cognition

Ideally, the agent would be able to adapt its reasoning strategies to mitigate faults in its own steering behaviors. 

## Minimal activation steering demo

This repo includes an `activation_steering` Python package with reusable Hugging Face–based helpers for:

- collecting hidden states from one decoder block
- building a mean-difference steering vector
- injecting that vector with a forward hook during generation
- composing a hybrid planner/retriever/steered-executor/verifier agent loop around persisted steering vectors
- comparing baseline, fixed steering, and an optional adaptive probe-based variant
- loading a file-backed starter catalog of standard prompt, context, and reasoning activations for GPT-2
- defining reusable feature-spec objects with extraction examples, test cases, and evaluation criteria

### Install

```bash
python -m pip install -r requirements.txt
```

### Dev container

This repo includes a VS Code dev container config that preinstalls the Python and
Jupyter extensions, installs the GitHub CLI plus the `gh-aw` extension, and installs the repo requirements plus `ipykernel`.

The repo also includes a Copilot setup workflow at `.github/workflows/copilot-setup-steps.yml` so the GitHub Copilot coding agent gets the same `gh-aw` tooling in its cloud environment.


### Agentic workflow automation

A daily GitHub Agentic Workflow source lives at `.github/workflows/daily-gh-aw-training.md`. It researches the repository's training artifacts for missing feature-extraction opportunities, then opens one delegated Copilot issue for the best candidate. The repository also includes a `gh-aw` Copilot skill under `.github/skills/gh-aw/` with command reference docs for the CLI surface.

### Notebook example

Open `notebooks/minimal_activation_steering.ipynb` to run the minimal end-to-end use case. The notebook defines its own sample prompts and imports the shared `activation_steering` library helpers directly.

Open `notebooks/hybrid_meta_cognition_agent.ipynb` for a reusable hybrid-agent example that:

- discovers and persists steering vectors
- reloads them as named controllers
- routes a task through planner, retrieval, steering, verification, and memory write-back
- keeps the demo logic in the notebook while using the shared library implementation

### Inspiration

This minimal activation-steering workflow was inspired in part by *Adaptive Activation Steering: A Tuning-Free LLM Truthfulness Improvement Method for Diverse Hallucinations Categories* (Wang et al.), available at https://arxiv.org/html/2406.00034v1.

### Use

Use the notebook for the minimal example, or import the package from Python applications that run from the repository root (or with `PYTHONPATH` pointed at the repo) to build vectors, attach steering hooks, and run steered generation.

### Standard activation catalog

The package includes a persistent JSON catalog of starter activations for a single model (`gpt2`). Load it with `activation_steering.load_standard_activation_catalog()` or get the activation rows directly with `activation_steering.get_standard_activations()`.

### Feature specification API

The package also includes reusable Python modules for defining features to extract. Use `FeatureSpec`, `FeatureExample`, and `EvaluationCriterion` directly in Python, or load starter file-backed specs with `activation_steering.get_standard_feature_catalog()` / `activation_steering.get_standard_feature_specs()`.

### Feature discovery and storage

Use `activation_steering.discover_feature_vectors(...)` to build one steering vector per feature spec from its labeled extraction examples, then persist the results with `activation_steering.save_discovered_feature_vectors(...)` or `activation_steering.discover_and_store_feature_vectors(...)`.

For reusable distribution, persist each extracted feature in its own directory under `activation_steering/artifacts/<model>/<feature>/` with `activation_steering.save_discovered_feature_vector_plugin(...)` or `activation_steering.discover_and_store_feature_vector_plugin(...)`. The checked-in starter features live at `activation_steering/artifacts/gpt2/` and the legacy regression fixture remains at `tests/data/minimal_identified_feature_vectors.json`.

### Persistent artifact features

Persistent artifacts are organised per model, with one directory per extracted feature — similar to agent-skills. Each feature is independently shareable and mergeable:

```text
activation_steering/artifacts/
└── <model>/
    └── <feature>/
        ├── plugin.json
        └── feature_vectors.json
```

- `feature_vectors.json` is the controller payload consumed by `load_steering_controllers(...)`.
- `plugin.json` describes the feature and the artifact files it contains.

Load one feature by path with `activation_steering.load_steering_controllers(...)`, or merge all features for a model across one or more roots with `activation_steering.load_artifact_plugin_controllers(model_name=..., artifact_roots=[...])`.

See [`docs/artifact_plugins.md`](docs/artifact_plugins.md) for the create/distribute/merge workflow.

### Hybrid agent library

Use `HybridMetaCognitionAgent`, `SteeredExecutor`, `InMemorySteeringMemory`, and `load_steering_controllers(...)` to build a reusable hybrid agent where a planner decides when to retrieve context and which persisted controller to apply before the verifier judges the result.

If you pass `artifact_dir=...` to `HybridMetaCognitionAgent`, call `agent.close()` (or use the agent as a context manager) to persist end-of-session runtime artifacts into `artifact_dir/<executor.model_name>/`. The agent writes session-level `adaptive_discoveries.json`, `graph_state.json`, and `graph_state.svg`, plus one per-feature directory for each discovered controller.

### Neo4j PathRAG / GraphRAG extension

The hybrid-agent library now also includes additive graph-native helpers for Neo4j-backed reasoning trajectories:

- `GraphTaskPlan`, `GraphConstraint`, and `GraphSubgoal` to anchor tasks, intents, constraints, and ordered subgoals
- `RetrievedPath` and `PathRAGContext` to keep evidence paths, analogous prior paths, and correction paths separated during retrieval
- `Neo4jPathRAGRetriever` to combine Neo4j candidate retrieval with Cypher path expansion
- `Neo4jGraphStore` to persist task/run/state/verifier/outcome nodes plus fallback-triggered drift corrections without replacing the existing in-memory steering memory
- dynamic interaction-feature learning that watches prompt/output pairs per model and attaches observed feature nodes to recorded states

This extension is intentionally minimal and keeps the existing planner/retriever/executor/verifier loop intact. It is designed to pair with Neo4j's official Python ecosystem, including the `neo4j` driver and `neo4j-graphrag` retrievers, instead of introducing a separate graph framework.
