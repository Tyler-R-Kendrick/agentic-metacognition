---
name: steering
description: "Run activation-steering feature discovery for a Hugging Face model. Use this skill whenever the user says /steering, wants to steer a model, extract a cognitive feature, generate steering vectors, run feature discovery, or produce steering artifacts. Specify a model (default: gpt2) and an optional feature name. If a feature is supplied, generate inputs, expected outputs, then run an extraction pass and output artifacts. If no feature is supplied, auto-pick one that hasn't already been extracted. If no test data is supplied, generate synthetic examples; otherwise use the user's data and fill in whatever's missing."
---

# /steering

Run activation-steering feature discovery end-to-end: specify a model and an optional feature, and this skill orchestrates data generation, vector extraction, and artifact output.

## When to use this skill

- The user types `/steering` or asks to "steer" a model
- The user wants to extract a cognitive feature (chain_of_thought, few_shot_prompting, react, etc.)
- The user wants to generate steering artifacts for a model
- The user asks to run feature discovery or build a steering vector
- The user mentions activation engineering, representation engineering, or contrastive extraction

## How it works

### Inputs

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"gpt2"` | Any decoder-only Hugging Face model identifier |
| `feature_name` | *auto-pick* | A feature to extract (e.g. `chain_of_thought`). If omitted, the system picks the next undiscovered feature from the standard catalog. |
| `user_examples` | *generate* | Optional list of `FeatureExample` objects with `text` and `label` ("positive" / "negative"). If omitted, synthetic examples are generated. If only one label is present, the other is generated. |
| `layer_idx` | `5` | Transformer layer for hidden-state collection |
| `output_dir` | `None` | Directory to write plugin artifacts to |

### Pipeline

1. **Resolve feature** — look up in the standard catalog, or generate a new spec.
2. **Ensure data** — use user-supplied examples, fill missing labels with synthetic data, or generate all data.
3. **Load model** — `load_model_and_tokenizer(model_name)`.
4. **Discover vectors** — `discover_feature_vectors(feature_spec, layer_idx, model, tokenizer, device)`.
5. **Write artifacts** — `write_artifact_plugin(...)` to produce a distributable plugin bundle.

### Output

A `SteeringResult` with:
- `feature_spec` — the resolved `FeatureSpec`
- `discovered_vectors` — list of `DiscoveredFeatureVector` (torch tensors + metadata)
- `artifact_dir` — path to the written plugin directory (if `output_dir` was set)

## Quick-start

```python
from activation_steering.steering_command import SteeringRunConfig, run_steering

# Explicit feature
result = run_steering(SteeringRunConfig(
    model_name="gpt2",
    feature_name="chain_of_thought",
    output_dir="./artifacts",
))

# Auto-pick next undiscovered feature
result = run_steering(SteeringRunConfig(
    model_name="gpt2",
    output_dir="./artifacts",
))

# With user-supplied data
from activation_steering.features import FeatureExample
result = run_steering(SteeringRunConfig(
    model_name="gpt2",
    feature_name="my_custom_feature",
    user_examples=[
        FeatureExample(text="detailed reasoning before answer", label="positive"),
        FeatureExample(text="just the answer", label="negative"),
    ],
    output_dir="./artifacts",
))
```

## Artifacts produced

```
<output_dir>/<model_name>/<feature_name>/
├── plugin.json          # Manifest
├── feature_specs.json   # The resolved FeatureSpec
└── controllers.json     # Discovered steering vectors
```

## Standard catalog features (gpt2)

| Feature | Category |
|---------|----------|
| `few_shot_prompting` | prompt_engineering |
| `retrieval_augmented_context` | context_engineering |
| `react` | cognitive_architecture |
| `chain_of_thought` | reasoning_strategy |

## Implementation

The command is implemented in `activation_steering/steering_command.py` and exported from `activation_steering`:

- `SteeringRunConfig` — run configuration dataclass
- `SteeringResult` — run output dataclass
- `run_steering(config)` — main orchestrator
- `build_steering_feature_spec(...)` — resolve or generate a FeatureSpec
- `pick_undiscovered_feature(...)` — auto-select next feature
- `generate_synthetic_examples(...)` — create training data

## API reference

For implementation details, read [`activation_steering/steering_command.py`](../../../activation_steering/steering_command.py).
