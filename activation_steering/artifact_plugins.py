from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ARTIFACT_PLUGIN_ROOT = files("activation_steering").joinpath("artifacts")
ARTIFACT_PLUGIN_MODELS_ROOT = ARTIFACT_PLUGIN_ROOT.joinpath("models")
STANDARD_ARTIFACT_PLUGIN_PATH = ARTIFACT_PLUGIN_MODELS_ROOT.joinpath("gpt2", "standard")
ARTIFACT_PLUGIN_MANIFEST = "plugin.json"
ACTIVATIONS_ARTIFACT = "activations.json"
FEATURE_SPECS_ARTIFACT = "feature_specs.json"
CONTROLLERS_ARTIFACT = "controllers.json"
ARTIFACT_PLUGIN_FORMAT_VERSION = 1


def _normalize_plugin_roots(
    plugin_roots: Sequence[str | Path] | str | Path | None,
) -> list[Any]:
    if plugin_roots is None:
        return [ARTIFACT_PLUGIN_ROOT]
    if isinstance(plugin_roots, (str, Path)):
        return [Path(plugin_roots)]
    return [Path(root) for root in plugin_roots]


def _sorted_children(root: Any) -> list[Any]:
    return sorted(root.iterdir(), key=lambda child: child.name)


def _read_json(path: Any) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _is_plugin_dir(path: Any) -> bool:
    return path.is_dir() and path.joinpath(ARTIFACT_PLUGIN_MANIFEST).is_file()


def _iter_plugin_dirs_from_collection(root: Any, model_name: str | None = None) -> list[Any]:
    plugin_dirs: list[Any] = []
    for candidate in _sorted_children(root):
        if not candidate.is_dir():
            continue
        if _is_plugin_dir(candidate):
            if model_name is None or candidate.parent.name == model_name:
                plugin_dirs.append(candidate)
            continue
        if model_name is not None and candidate.name != model_name:
            continue
        for plugin_dir in _sorted_children(candidate):
            if _is_plugin_dir(plugin_dir):
                plugin_dirs.append(plugin_dir)
    return plugin_dirs


def discover_artifact_plugin_paths(
    plugin_roots: Sequence[str | Path] | str | Path | None = None,
    model_name: str | None = None,
) -> list[Any]:
    plugin_dirs: list[Any] = []
    seen: set[str] = set()
    for root in _normalize_plugin_roots(plugin_roots):
        if _is_plugin_dir(root):
            manifest = _read_json(root.joinpath(ARTIFACT_PLUGIN_MANIFEST))
            resolved_model_name = str(manifest["model_name"])
            if model_name is not None and resolved_model_name != model_name:
                continue
            key = str(root)
            if key not in seen:
                plugin_dirs.append(root)
                seen.add(key)
            continue

        if root.joinpath("models").is_dir():
            candidates = _iter_plugin_dirs_from_collection(root.joinpath("models"), model_name=model_name)
        else:
            candidates = _iter_plugin_dirs_from_collection(root, model_name=model_name)

        for plugin_dir in candidates:
            key = str(plugin_dir)
            if key in seen:
                continue
            plugin_dirs.append(plugin_dir)
            seen.add(key)
    return plugin_dirs


def _load_plugin_payload(plugin_dir: Any) -> dict[str, Any]:
    manifest = _read_json(plugin_dir.joinpath(ARTIFACT_PLUGIN_MANIFEST))
    return {
        "plugin_path": str(plugin_dir),
        "plugin_name": str(manifest.get("plugin_name") or plugin_dir.name),
        "model_name": str(manifest["model_name"]),
        "description": str(manifest.get("description") or ""),
        "is_default_model": bool(manifest.get("is_default_model", False)),
        "metadata": dict(manifest.get("metadata") or {}),
        "activations": _read_json(plugin_dir.joinpath(ACTIVATIONS_ARTIFACT)).get("activations", [])
        if plugin_dir.joinpath(ACTIVATIONS_ARTIFACT).is_file()
        else [],
        "feature_specs": _read_json(plugin_dir.joinpath(FEATURE_SPECS_ARTIFACT)).get("features", [])
        if plugin_dir.joinpath(FEATURE_SPECS_ARTIFACT).is_file()
        else [],
        "controllers": _read_json(plugin_dir.joinpath(CONTROLLERS_ARTIFACT)).get("feature_vectors", [])
        if plugin_dir.joinpath(CONTROLLERS_ARTIFACT).is_file()
        else [],
    }


@lru_cache(maxsize=1)
def _load_builtin_artifact_catalog_payload() -> dict[str, Any]:
    return _build_artifact_catalog_from_plugins(discover_artifact_plugin_paths())


def _merge_named_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    key_field: str,
    fallback_key_field: str | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for entry in entries:
        raw_key = entry.get(key_field)
        if raw_key is None and fallback_key_field is not None:
            raw_key = entry.get(fallback_key_field)
        if raw_key is None:
            raise ValueError(f"Artifact entry is missing required key {key_field!r}.")
        key = str(raw_key)
        merged[key] = copy.deepcopy(dict(entry))
    return list(merged.values())


def _build_artifact_catalog_from_plugins(plugin_dirs: Sequence[Any]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    default_model = None

    for plugin_dir in plugin_dirs:
        plugin = _load_plugin_payload(plugin_dir)
        model_entry = models.setdefault(
            plugin["model_name"],
            {
                "description": "",
                "metadata": {},
                "plugins": [],
                "activations": [],
                "feature_specs": [],
                "controllers": [],
            },
        )
        if plugin["description"] and not model_entry["description"]:
            model_entry["description"] = plugin["description"]
        model_entry["metadata"].update(plugin["metadata"])
        model_entry["plugins"].append(
            {
                "plugin_name": plugin["plugin_name"],
                "plugin_path": plugin["plugin_path"],
                "description": plugin["description"],
                "metadata": copy.deepcopy(plugin["metadata"]),
            }
        )
        model_entry["activations"].extend(plugin["activations"])
        model_entry["feature_specs"].extend(plugin["feature_specs"])
        model_entry["controllers"].extend(plugin["controllers"])
        if plugin["is_default_model"]:
            default_model = plugin["model_name"]

    for model_name, model_entry in models.items():
        model_entry["activations"] = _merge_named_entries(
            model_entry["activations"], key_field="name"
        )
        model_entry["feature_specs"] = _merge_named_entries(
            model_entry["feature_specs"], key_field="name"
        )
        model_entry["controllers"] = _merge_named_entries(
            model_entry["controllers"], key_field="controller_id", fallback_key_field="name"
        )
        if not model_entry["description"]:
            model_entry["description"] = (
                f"Merged persistent artifact plugins for {model_name}."
            )

    if default_model is None and models:
        default_model = sorted(models)[0]

    return {
        "format_version": ARTIFACT_PLUGIN_FORMAT_VERSION,
        "default_model": default_model,
        "models": models,
    }


def load_artifact_plugin_catalog(
    plugin_roots: Sequence[str | Path] | str | Path | None = None,
) -> dict[str, Any]:
    if plugin_roots is None:
        return copy.deepcopy(_load_builtin_artifact_catalog_payload())
    return _build_artifact_catalog_from_plugins(discover_artifact_plugin_paths(plugin_roots))


def load_model_artifact_bundle(
    model_name: str | None = None,
    plugin_roots: Sequence[str | Path] | str | Path | None = None,
) -> dict[str, Any]:
    catalog = load_artifact_plugin_catalog(plugin_roots=plugin_roots)
    selected_model = model_name or catalog["default_model"]
    if selected_model is None or selected_model not in catalog["models"]:
        available_models = ", ".join(sorted(catalog["models"]))
        raise ValueError(
            f"Unknown model_name {selected_model!r}; choose from: {available_models}."
        )
    return copy.deepcopy(
        {
            "default_model": catalog["default_model"],
            "model_name": selected_model,
            **catalog["models"][selected_model],
        }
    )


def _normalize_payload_entries(entries: Sequence[Any] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in entries or ():
        if hasattr(entry, "to_dict"):
            normalized.append(dict(entry.to_dict()))
        else:
            normalized.append(dict(entry))
    return normalized


def _normalize_controller_entries(entries: Sequence[Any] | None) -> list[dict[str, Any]]:
    normalized = _normalize_payload_entries(entries)
    for entry in normalized:
        controller_id = entry.get("controller_id") or entry.get("name")
        feature_name = entry.get("feature_name") or entry.get("name")
        if controller_id is not None:
            entry["controller_id"] = str(controller_id)
        if feature_name is not None:
            entry["feature_name"] = str(feature_name)
    return normalized


def write_artifact_plugin(
    output_dir: str | Path,
    model_name: str,
    plugin_name: str,
    *,
    description: str = "",
    activations: Sequence[Any] | None = None,
    feature_specs: Sequence[Any] | None = None,
    controllers: Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    is_default_model: bool = False,
) -> Path:
    destination = Path(output_dir) / "models" / model_name / plugin_name
    destination.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": ARTIFACT_PLUGIN_FORMAT_VERSION,
        "plugin_name": plugin_name,
        "model_name": model_name,
        "description": description,
        "is_default_model": is_default_model,
        "metadata": dict(metadata or {}),
    }
    destination.joinpath(ARTIFACT_PLUGIN_MANIFEST).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    normalized_activations = _normalize_payload_entries(activations)
    if normalized_activations:
        destination.joinpath(ACTIVATIONS_ARTIFACT).write_text(
            json.dumps(
                {
                    "format_version": ARTIFACT_PLUGIN_FORMAT_VERSION,
                    "activation_count": len(normalized_activations),
                    "activations": normalized_activations,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    normalized_features = _normalize_payload_entries(feature_specs)
    if normalized_features:
        destination.joinpath(FEATURE_SPECS_ARTIFACT).write_text(
            json.dumps(
                {
                    "format_version": ARTIFACT_PLUGIN_FORMAT_VERSION,
                    "feature_count": len(normalized_features),
                    "features": normalized_features,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    normalized_controllers = _normalize_controller_entries(controllers)
    if normalized_controllers:
        destination.joinpath(CONTROLLERS_ARTIFACT).write_text(
            json.dumps(
                {
                    "format_version": ARTIFACT_PLUGIN_FORMAT_VERSION,
                    "feature_vector_count": len(normalized_controllers),
                    "feature_vectors": normalized_controllers,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return destination


def merge_artifact_plugins(
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    plugin_roots: Sequence[str | Path] | str | Path | None = None,
    merged_plugin_name: str = "merged",
    description: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    bundle = load_model_artifact_bundle(model_name=model_name, plugin_roots=plugin_roots)
    return write_artifact_plugin(
        output_dir=output_dir,
        model_name=bundle["model_name"],
        plugin_name=merged_plugin_name,
        description=description or bundle["description"],
        activations=bundle["activations"],
        feature_specs=bundle["feature_specs"],
        controllers=bundle["controllers"],
        metadata=metadata,
        is_default_model=bundle["model_name"] == bundle["default_model"],
    )
