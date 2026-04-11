---
name: feature-discovery
description: Define, discover, and manage cognitive feature specifications and steering vectors for LLM activation steering. Use this skill when the user wants to create feature specs, define extraction examples with positive/negative labels, set evaluation criteria, discover feature vectors from specs, persist discovered vectors, work with the standard feature catalog, or manage the feature lifecycle. Also trigger when the user mentions feature extraction, feature catalog, feature specs, cognitive features, or interaction features.
---

# Feature Discovery

Use this skill to define reusable feature specifications, discover steering vectors from those specs, and manage feature catalogs for the `activation_steering` package.

## When to use this skill

- Creating a new `FeatureSpec` with labeled extraction examples
- Defining `EvaluationCriterion` rules for judging feature extraction
- Discovering steering vectors from feature specs via `discover_feature_vectors`
- Persisting discovered vectors with `save_discovered_feature_vectors`
- Loading or extending the standard feature catalog
- Discovering dynamic interaction features from prompt/output pairs

## Feature specification model

A feature spec has four components:

1. **Identity**: `name`, `category`, `summary`
2. **Extraction examples**: Labeled text samples (`FeatureExample` with `text`, `label`, `metadata`)
3. **Test cases**: Additional labeled examples for validation
4. **Evaluation criteria**: `EvaluationCriterion` rules with `name`, `description`, `threshold`

### Creating a feature spec

```python
from activation_steering import FeatureSpec, FeatureExample, EvaluationCriterion, build_feature_spec

spec = build_feature_spec(
    name="truthfulness",
    category="reasoning",
    summary="Measures whether the model produces truthful, evidence-based statements",
    extraction_examples=[
        FeatureExample(text="The study found a strong correlation.", label="positive"),
        FeatureExample(text="Everyone knows this is obviously true.", label="negative"),
    ],
    test_cases=[
        FeatureExample(text="According to the data, the trend is upward.", label="positive"),
        FeatureExample(text="Trust me, this always works.", label="negative"),
    ],
    evaluation_criteria=[
        EvaluationCriterion(
            name="cosine_alignment",
            description="Positive examples should have higher cosine similarity to the vector",
            threshold=0.5,
        ),
    ],
)
```

### Loading the standard catalog

```python
from activation_steering import get_standard_feature_specs, get_standard_feature_catalog

# Get all specs for a model
specs = get_standard_feature_specs(model_name="gpt2")

# Get the full catalog object
catalog = get_standard_feature_catalog(model_name="gpt2")
```

## Discovering feature vectors

The discovery pipeline builds one steering vector per feature spec from its labeled examples:

```python
from activation_steering import (
    discover_feature_vectors,
    save_discovered_feature_vectors,
    discover_and_store_feature_vectors,
)

# Step by step
vectors = discover_feature_vectors(
    feature_specs=specs,
    model_name="gpt2",
    layer_idx=5,
    model=model,
    tokenizer=tokenizer,
    device=device,
)
save_discovered_feature_vectors(vectors, output_dir="./my_vectors")

# Or all-in-one
discover_and_store_feature_vectors(
    feature_specs=specs,
    model_name="gpt2",
    layer_idx=5,
    model=model,
    tokenizer=tokenizer,
    device=device,
    output_dir="./my_vectors",
)
```

Each `DiscoveredFeatureVector` contains:
- `name`, `model_name`, `category`, `summary`
- `layer_idx`, `vector` (the steering tensor)
- `positive_example_count`, `negative_example_count`, `test_case_count`
- `evaluation_criteria` (serialized list)

## Interaction feature discovery

For dynamic features learned from prompt/output pairs during agent usage:

```python
from activation_steering import discover_interaction_features

features = discover_interaction_features(
    model_name="gpt2",
    prompt="What is the capital of France?",
    output="The capital of France is Paris.",
    model=model,
    tokenizer=tokenizer,
    device=device,
)
```

Each `ObservedInteractionFeature` is identified as `interaction::{prompt_shape}__{context_usage}__{output_shape}`.

## Standard catalog layout

Standard feature specs live at:
```
activation_steering/artifacts/gpt2/standard/feature_specs.json
```

Use `get_standard_feature_models()` to list available models and `load_standard_feature_catalogs()` to load all catalogs at once.

## API reference

For full details, read `activation_steering/features.py` (specs, examples, criteria, catalogs) and `activation_steering/discovery.py` (vector discovery, interaction features, persistence).
