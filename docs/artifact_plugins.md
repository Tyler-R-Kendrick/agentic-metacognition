# Artifact plugins

Persistent artifacts are organised per model as shareable per-feature directories — similar to agent-skills. Each feature is independently distributable and mergeable:

```text
<artifact-root>/
└── <model-name>/
    └── <feature-name>/
        ├── plugin.json
        └── feature_vectors.json
```

Only `plugin.json` and `feature_vectors.json` are required for controller reuse.

## File roles

- `plugin.json`: feature metadata and the artifact files present in the directory
- `feature_vectors.json`: persisted discovered controller / feature vector for this feature

## Create features

Generate controllers and save them directly into the per-feature layout:

```python
import activation_steering as steering

steering.discover_and_store_feature_vector_plugin(
    feature_specs=steering.get_standard_feature_specs(),
    layer_idx=1,
    model=model,
    tokenizer=tokenizer,
    device="cpu",
    artifact_root="shared-artifacts",
    model_name="gpt2",
)
```

This creates one directory per feature spec (e.g. `shared-artifacts/gpt2/chain_of_thought/`, `shared-artifacts/gpt2/few_shot_prompting/`, etc.).

If you already have `DiscoveredFeatureVector` objects, use `save_discovered_feature_vector_plugin(...)` instead.

## Distribute a feature

Copy or publish the whole `<feature-name>/` directory. The feature is self-contained, so it can live:

- inside this repository under `activation_steering/artifacts/`
- in another repository
- in a shared filesystem location
- inside a release artifact or zip file

As long as the directory lands under `<model-name>/<feature-name>/`, the merge helpers can discover it.

## Merge features

To load every feature available for one model from one or more roots:

```python
controllers = steering.load_artifact_plugin_controllers(
    "gpt2",
    artifact_roots=[
        "activation_steering/artifacts",
        "/mnt/shared/model-artifacts",
    ],
)
```

The loader merges every `feature_vectors.json` file it finds for that model. If two roots define the same feature name, the later root in traversal order replaces the earlier one. Use distinct feature names to keep contributions isolated, and only rely on replacement when you intentionally want an override.

## Load a single feature

Use `load_steering_controllers(...)` with a specific feature directory:

```python
controllers = steering.load_steering_controllers(
    "activation_steering/artifacts/gpt2/chain_of_thought"
)
```

Or load all features for a model directory:

```python
controllers = steering.load_steering_controllers(
    "activation_steering/artifacts/gpt2"
)
```

## Runtime persistence

`HybridMetaCognitionAgent(..., artifact_dir=...)` persists runtime artifacts into:

```text
<artifact_dir>/<executor.model_name>/
```

Session-level files (`adaptive_discoveries.json`, `graph_state.json`, `graph_state.svg`) go at the model level. Each discovered controller also gets its own per-feature directory with `feature_vectors.json` and `plugin.json`, so the output is directly shareable as feature artifacts.
