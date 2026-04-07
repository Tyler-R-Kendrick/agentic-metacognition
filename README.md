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
Jupyter extensions and installs the repo requirements plus `ipykernel`.

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

The repository also keeps a checked-in minimal-example artifact in source control at `tests/data/minimal_identified_feature_vectors.json`, and the integration test regenerates that file's contents to ensure the stored vectors stay in sync with the discovery flow.

### Hybrid agent library

Use `HybridMetaCognitionAgent`, `SteeredExecutor`, `InMemorySteeringMemory`, and `load_steering_controllers(...)` to build a reusable hybrid agent where a planner decides when to retrieve context and which persisted controller to apply before the verifier judges the result.
