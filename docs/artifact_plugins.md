# Artifact plugins

Persistent artifacts can be packaged per model as shareable plugins under a common directory layout:

```text
<artifact-root>/
└── models/
    └── <model-name>/
        └── <plugin-name>/
            ├── plugin.json
            ├── feature_vectors.json
            ├── adaptive_discoveries.json
            ├── graph_state.json
            └── graph_state.svg
```

Only `plugin.json` and `feature_vectors.json` are required for controller reuse. The runtime files are optional but recommended when you want to preserve the provenance that produced the controllers.

## File roles

- `plugin.json`: plugin metadata and the artifact files present in the directory
- `feature_vectors.json`: persisted discovered controllers or feature vectors
- `adaptive_discoveries.json`: learned interaction features and controller summaries from runtime use
- `graph_state.json`: diff-friendly graph snapshot for a run or merged session
- `graph_state.svg`: visualization companion for `graph_state.json`

## Create a plugin

Generate controllers and save them directly into the plugin layout:

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
    plugin_name="retrieval-baseline",
    description="Controllers extracted from the retrieval-focused starter catalog.",
)
```

If you already have `DiscoveredFeatureVector` objects, use `save_discovered_feature_vector_plugin(...)` instead.

## Distribute a plugin

Copy or publish the whole `<plugin-name>/` directory. The plugin is self-contained, so it can live:

- inside this repository under `activation_steering/artifacts/`
- in another repository
- in a shared filesystem location
- inside a release artifact or zip file

As long as the directory lands under `models/<model-name>/<plugin-name>/`, the merge helpers can discover it.

## Merge plugins

To load every plugin available for one model from one or more roots:

```python
controllers = steering.load_artifact_plugin_controllers(
    "gpt2",
    artifact_roots=[
        "activation_steering/artifacts",
        "/mnt/shared/model-artifacts",
    ],
)
```

The loader merges every `feature_vectors.json` file it finds for that model. If two plugins define the same controller id or feature name, the later plugin in traversal order replaces the earlier one. Use distinct plugin names to keep contributions isolated, and only rely on replacement when you intentionally want an override.

## Load a single plugin

Use `load_steering_controllers(...)` with a specific plugin directory:

```python
controllers = steering.load_steering_controllers(
    "activation_steering/artifacts/models/gpt2/minimal"
)
```

## Runtime persistence

`HybridMetaCognitionAgent(..., artifact_dir=...)` persists runtime artifacts into:

```text
<artifact_dir>/models/<executor.model_name>/runtime/
```

That output already matches the plugin layout, so you can copy the resulting `runtime/` directory, rename it, or merge it with other plugin roots later.
