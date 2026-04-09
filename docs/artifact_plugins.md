# Artifact plugins

Persistent artifacts are organized as model-scoped plugin directories so they can be copied, versioned, and merged the same way reusable agent skills are shared.

```text
<artifact-root>/
└── <model-name>/
    └── <plugin-name>/
        ├── plugin.json
        ├── activations.json      # optional
        ├── feature_specs.json    # optional
        └── controllers.json      # optional
```

Only `plugin.json` is required. The other files are optional so a plugin can contribute only the artifacts it needs.

## File roles

- `plugin.json`: plugin metadata, model name, and a human-readable description
- `activations.json`: named activation rows that extend or override activation summaries
- `feature_specs.json`: reusable extraction feature definitions for the model
- `controllers.json`: persisted discovered feature vectors that can be loaded as steering controllers

## Create a plugin bundle

Generate artifacts and write them into the plugin layout:

```python
import activation_steering as steering

steering.write_artifact_plugin(
    output_dir="shared-artifacts/gpt2/my-plugin",
    model_name="gpt2",
    description="Controllers from the starter catalog.",
    controllers=discovered_vectors,
)
```

## Distribute a plugin

Copy or publish the whole `<plugin-name>/` directory. The plugin is self-contained, so it can live:

- inside this repository under `activation_steering/artifacts/`
- in another repository
- in a shared filesystem location
- inside a release artifact or zip file

As long as the directory lands under `<model-name>/<plugin-name>/`, the merge helpers can discover it.

## Load and merge plugins

To load every plugin available for one model from one or more roots:

```python
controllers = steering.load_artifact_steering_controllers(
    "gpt2",
    plugin_roots=[
        "activation_steering/artifacts",
        "/mnt/shared/model-artifacts",
    ],
)
```

The loader merges all controllers found for that model. If two roots define the same controller id, the later root in traversal order replaces the earlier one.

Use `steering.merge_artifact_plugins(...)` to collapse multiple roots into one distributable bundle.

## Runtime persistence

`HybridMetaCognitionAgent(..., artifact_dir=...)` persists runtime artifacts (`adaptive_discoveries.json`, `graph_state.json`, `graph_state.svg`) into the given directory on close.
