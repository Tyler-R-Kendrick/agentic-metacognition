---
name: artifact-plugins
description: Create, load, merge, and distribute persistent artifact plugin bundles for activation steering. Use this skill when the user wants to manage steering artifacts, create distributable plugin packs, load model bundles, merge multiple plugins, write new artifact directories, or work with the plugin directory tree. Also trigger when the user mentions artifact plugins, plugin bundles, plugin manifests, controllers.json, activations.json, or the artifacts/ directory layout.
---

# Artifact Plugins

Use this skill to manage the persistent, file-backed artifact plugin system that stores steering activations, feature specs, and controllers as distributable bundles under `activation_steering/artifacts/`.

## When to use this skill

- Creating a new artifact plugin bundle
- Loading model artifact bundles for a specific model
- Merging multiple plugin bundles into a single pack
- Loading steering controllers from artifact bundles
- Understanding or modifying the plugin directory layout
- Writing plugin manifests (`plugin.json`)
- Distributing or sharing steering artifact packs

## Plugin directory layout

```
activation_steering/artifacts/
└── <model>/
    └── <plugin>/
        ├── plugin.json          # Manifest with format_version, description, model
        ├── activations.json     # Labeled activation catalog entries
        ├── feature_specs.json   # Feature specifications with examples
        └── controllers.json     # Named steering controllers with vectors
```

The built-in GPT-2 starter bundle is at:
```
activation_steering/artifacts/gpt2/standard/
```

Plugin identifiers are derived from the sanitized folder path (not from an explicit name field in the manifest).

## Core operations

### Loading a model bundle

```python
from activation_steering import load_model_artifact_bundle

# Load and merge all plugins for a model
bundle = load_model_artifact_bundle(model_name="gpt2")
# bundle contains merged activations, feature_specs, and controllers
```

### Loading steering controllers

```python
from activation_steering import load_artifact_steering_controllers

controllers = load_artifact_steering_controllers(
    model_name="gpt2",
    model=model,
    tokenizer=tokenizer,
    device=device,
)
# Returns a dict of controller_name -> SteeringController
```

### Writing a new plugin

```python
from activation_steering import write_artifact_plugin

write_artifact_plugin(
    output_dir="./my_plugin",
    model_name="gpt2",
    description="Custom truthfulness steering artifacts",
    activations=my_activations,       # list of activation entries
    feature_specs=my_specs,           # list of FeatureSpec dicts
    controllers=my_controllers,       # list of controller dicts
)
```

### Merging plugins

```python
from activation_steering import merge_artifact_plugins

merged = merge_artifact_plugins(
    plugin_roots=["./plugin_a", "./plugin_b"],
    model_name="gpt2",
)
# Returns a single merged bundle
```

## Plugin manifest format

Each `plugin.json` contains:
```json
{
  "format_version": 1,
  "description": "Description of what this plugin provides",
  "model": "gpt2"
}
```

## Discovery and enumeration

```python
from activation_steering import discover_artifact_plugin_paths

# Find all plugin directories under the default or custom roots
paths = discover_artifact_plugin_paths()
# Returns a list of Path objects pointing to plugin directories
```

## Loading the full catalog

```python
from activation_steering import load_artifact_plugin_catalog

catalog = load_artifact_plugin_catalog()
# Returns the merged catalog with default_model, models, etc.
```

## Working rules

1. Always use `write_artifact_plugin` to create new plugins — don't manually write JSON files.
2. Plugin names come from folder paths after sanitization; keep folder names descriptive and use lowercase with hyphens.
3. When merging, later plugins override earlier ones for duplicate keys.
4. The `ARTIFACT_PLUGIN_ROOT` constant points to the base artifacts directory.
5. Use `load_model_artifact_bundle` for read operations and `write_artifact_plugin` + `merge_artifact_plugins` for write/combine operations.

## API reference

For the full API, read `activation_steering/artifact_plugins.py` and `activation_steering/artifacts/README.md`.
