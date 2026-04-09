# Persistent artifact plugins

Persistent artifacts are organized as model-scoped plugins so they can be copied, versioned, and merged the same way reusable agent skills are shared.

## Directory layout

```text
activation_steering/artifacts/
└── models/
    └── <model_name>/
        └── <plugin_folder>/
            ├── plugin.json
            ├── activations.json      # optional
            ├── feature_specs.json    # optional
            └── controllers.json      # optional
```

The built-in starter bundle lives at:

```text
activation_steering/artifacts/models/gpt2/standard/
```

## File responsibilities

- `plugin.json`: plugin metadata, model name, and a human-readable description
- `activations.json`: named activation rows that extend or override activation summaries
- `feature_specs.json`: reusable extraction feature definitions for the model
- `controllers.json`: persisted discovered feature vectors that can be loaded as steering controllers

The plugin identifier is not user-specified. It is derived from the plugin folder path under the model directory after sanitization.

All plugin files are optional except `plugin.json`, so a shared bundle can contain only the artifacts it contributes.

## Merge behavior

- plugins are discovered per model under `models/<model>/<plugin>/`
- entries are merged by name
- later plugins win on duplicate keys
- within one root, plugin directories are read in lexical order
- if you pass multiple roots, later roots override earlier ones

This makes it easy to keep a stable base bundle and layer team-local or experiment-specific bundles on top.

## Create a plugin bundle

1. Discover or collect the artifacts you want to share for one model.
2. Choose the destination plugin directory under `models/<model>/...`.
3. Write the bundle into that directory with `activation_steering.write_artifact_plugin(...)`.
4. Commit or archive the resulting plugin directory.

Typical inputs:

- activations from curated summaries
- feature specs from extraction examples and evaluation criteria
- controllers from `save_discovered_feature_vectors(...)` output or runtime discoveries

## Load distributed bundles

- `activation_steering.load_model_artifact_bundle(model_name=..., plugin_roots=...)` loads the merged artifact view for one model
- `activation_steering.load_artifact_steering_controllers(model_name=..., plugin_roots=...)` loads merged controllers directly
- `activation_steering.discover_artifact_plugin_paths(plugin_roots=...)` lists the plugin directories that will participate

`plugin_roots` can point at a full artifact root that contains `models/`, a model directory, or an individual plugin directory.

## Merge bundles into one distributable pack

Use `activation_steering.merge_artifact_plugins(...)` when you want to collapse several roots into one plugin tree for shipping:

1. collect the source roots you want to combine
2. call `merge_artifact_plugins(...)` with the target plugin directory and model name
3. distribute the generated plugin directory

This is useful for publishing a curated bundle after several people independently discover new features for the same model.
