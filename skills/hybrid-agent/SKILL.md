---
name: hybrid-agent
description: Build and run a hybrid planner/retriever/steered-executor/verifier meta-cognition agent loop with persistent steering memory. Use this skill when the user wants to create an agent that plans tasks, retrieves context, applies activation steering, verifies results, and writes back to memory. Also use when the user mentions HybridMetaCognitionAgent, SteeredExecutor, planner/verifier loops, steering controllers, agent runs, or activation traces — even if they don't say "hybrid agent" explicitly.
---

# Hybrid Meta-Cognition Agent

Use this skill to build and operate a reusable hybrid agent that orchestrates planning, retrieval, steered execution, and verification in a reflective loop with persistent steering memory.

## When to use this skill

- Creating a `HybridMetaCognitionAgent` instance
- Configuring the planner/retriever/executor/verifier pipeline
- Loading and using named steering controllers
- Working with `InMemorySteeringMemory` for session state
- Running agent tasks with `SteeredExecutor`
- Persisting runtime artifacts (adaptive discoveries, graph state)
- Building custom agent loops around the library components

## Agent architecture

The hybrid agent uses a four-stage loop:

1. **Planner** — Decides what to do, which controller to apply, and whether retrieval is needed. Returns a `PlannerDecision`.
2. **Retriever** — Fetches relevant context (steering memory, prior runs, domain knowledge).
3. **Steered Executor** — Generates output using a selected `SteeringController` via `SteeredExecutor`. Returns an `ExecutorResult` with the generated text and an `ActivationTrace`.
4. **Verifier** — Judges the executor's output for correctness and completeness. Returns a `VerifierResult` with a completeness score and any issues found.

## Quick-start

```python
from activation_steering import (
    HybridMetaCognitionAgent,
    InMemorySteeringMemory,
    load_steering_controllers,
)

# Load controllers from discovered vectors
controllers = load_steering_controllers(
    vector_dir="./my_vectors",
    model=model,
    tokenizer=tokenizer,
    device=device,
    layer_idx=5,
)

# Build the agent
memory = InMemorySteeringMemory()
agent = HybridMetaCognitionAgent(
    model=model,
    tokenizer=tokenizer,
    device=device,
    controllers=controllers,
    memory=memory,
)

# Run a task
run = agent.run("Explain quantum entanglement simply")
print(run.executor_result.text)
print(run.verifier_result)
```

## Loading controllers

Controllers can be loaded from two sources:

### From discovered feature vectors
```python
from activation_steering import load_steering_controllers
controllers = load_steering_controllers(
    vector_dir="./vectors", model=model, tokenizer=tokenizer,
    device=device, layer_idx=5,
)
```

### From artifact plugin bundles
```python
from activation_steering import load_artifact_steering_controllers
controllers = load_artifact_steering_controllers(
    model_name="gpt2", model=model, tokenizer=tokenizer, device=device,
)
```

## Key data classes

| Class | Purpose |
|-------|---------|
| `SteeringController` | Named controller with a steering vector, alpha, layer index |
| `PlannerDecision` | Planner output: chosen controller, retrieval flag, reasoning |
| `ExecutorResult` | Executor output: generated text, activation trace |
| `VerifierResult` | Verifier output: completeness score, issues list |
| `ActivationTrace` | Records steering metadata: controller used, alpha, cosine scores |
| `HybridAgentRun` | Full run record: planner decision, executor result, verifier result |
| `SteeringFeatureScore` | Feature-level scoring from the verifier |

## Persisting runtime artifacts

When you pass `artifact_dir=...` to the agent, use it as a context manager to automatically persist end-of-session data:

```python
with HybridMetaCognitionAgent(
    model=model, tokenizer=tokenizer, device=device,
    controllers=controllers, memory=memory,
    artifact_dir="./session_artifacts",
) as agent:
    run = agent.run("Summarize the research paper")
# On exit: writes adaptive_discoveries.json, graph_state.json, graph_state.svg
```

Or call `agent.close()` explicitly.

## Building executor prompts

```python
from activation_steering import build_executor_prompt
prompt = build_executor_prompt(task="Explain relativity", context="Physics textbook excerpt...")
```

## Collecting controller traces

```python
from activation_steering import collect_controller_trace
trace = collect_controller_trace(
    controller=controllers["truthfulness"],
    prompt="The evidence shows...",
    model=model, tokenizer=tokenizer, device=device,
)
```

## API reference

For the full API, read `activation_steering/agent.py`.
