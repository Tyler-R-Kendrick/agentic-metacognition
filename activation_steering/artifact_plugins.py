from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ARTIFACT_PLUGIN_FORMAT_VERSION = 1
ARTIFACT_PLUGIN_ROOT = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PLUGIN_MANIFEST_NAME = "plugin.json"
ARTIFACT_PLUGIN_FEATURE_VECTORS_NAME = "feature_vectors.json"


def _require_segment(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must be a non-empty string.")
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"{label} must not contain path separators.")
    return normalized


def _coerce_payload_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "feature_vectors" in payload:
        return [dict(entry) for entry in payload["feature_vectors"]]
    if "controllers" in payload:
        return [dict(entry) for entry in payload["controllers"]]
    raise ValueError("Expected a payload with either 'feature_vectors' or 'controllers'.")


def _entry_key(entry: Mapping[str, Any]) -> str:
    controller_id = entry.get("controller_id", entry.get("name"))
    if controller_id is None:
        raise ValueError("Each artifact plugin entry must include 'controller_id' or 'name'.")
    return str(controller_id)


def get_artifact_plugin_dir(
    model_name: str,
    feature_name: str,
    *,
    artifact_root: str | Path = ARTIFACT_PLUGIN_ROOT,
) -> Path:
    """Return ``<artifact_root>/<model_name>/<feature_name>/``."""
    return (
        Path(artifact_root)
        / _require_segment(model_name, "model_name")
        / _require_segment(feature_name, "feature_name")
    )


def build_artifact_plugin_manifest(
    *,
    model_name: str,
    feature_name: str,
    description: str | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format_version": ARTIFACT_PLUGIN_FORMAT_VERSION,
        "model_name": _require_segment(model_name, "model_name"),
        "feature_name": _require_segment(feature_name, "feature_name"),
        "artifacts": dict(artifacts or {}),
    }
    if description:
        manifest["description"] = str(description).strip()
    return manifest


def write_artifact_plugin_manifest(
    plugin_dir: str | Path,
    *,
    model_name: str,
    feature_name: str,
    description: str | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> Path:
    destination = Path(plugin_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / ARTIFACT_PLUGIN_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            build_artifact_plugin_manifest(
                model_name=model_name,
                feature_name=feature_name,
                description=description,
                artifacts=artifacts,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def list_artifact_features(
    model_name: str,
    *,
    artifact_roots: Sequence[str | Path] | None = None,
) -> list[str]:
    """Return sorted feature names available for *model_name* across *artifact_roots*."""
    names: set[str] = set()
    for artifact_root in artifact_roots or (ARTIFACT_PLUGIN_ROOT,):
        model_dir = Path(artifact_root) / _require_segment(model_name, "model_name")
        if not model_dir.is_dir():
            continue
        for child in model_dir.iterdir():
            if child.is_dir():
                names.add(child.name)
    return sorted(names)


def _iter_feature_dirs_from_root(
    artifact_root: str | Path,
    model_name: str,
    feature_names: Sequence[str] | None = None,
) -> list[Path]:
    model_dir = Path(artifact_root) / _require_segment(model_name, "model_name")
    if not model_dir.is_dir():
        return []
    allowed = {name for name in feature_names or ()}
    feature_dirs: list[Path] = []
    for child in sorted(model_dir.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if allowed and child.name not in allowed:
            continue
        if (
            (child / ARTIFACT_PLUGIN_MANIFEST_NAME).is_file()
            or (child / ARTIFACT_PLUGIN_FEATURE_VECTORS_NAME).is_file()
        ):
            feature_dirs.append(child)
    return feature_dirs


def resolve_artifact_plugin_dirs(
    input_path: str | Path,
    *,
    model_name: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> list[Path]:
    """Resolve artifact feature directories under *input_path*.

    Handles:
    - a single feature dir (contains ``plugin.json`` or ``feature_vectors.json``)
    - a model dir containing feature sub-dirs (``<model>/<feature>/``)
    - an artifact root containing ``<model>/<feature>/`` dirs
    """
    path = Path(input_path)
    if path.is_file():
        return []
    # Exact feature dir
    if (
        (path / ARTIFACT_PLUGIN_MANIFEST_NAME).is_file()
        or (path / ARTIFACT_PLUGIN_FEATURE_VECTORS_NAME).is_file()
    ):
        return [path]
    # Check immediate children (model dir case: <model>/<feature>/)
    allowed = {name for name in feature_names or ()}
    direct_feature_dirs: list[Path] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if allowed and child.name not in allowed:
            continue
        if (
            (child / ARTIFACT_PLUGIN_MANIFEST_NAME).is_file()
            or (child / ARTIFACT_PLUGIN_FEATURE_VECTORS_NAME).is_file()
        ):
            direct_feature_dirs.append(child)
    if direct_feature_dirs:
        return direct_feature_dirs
    # Artifact root with model_name specified: <root>/<model>/<feature>/
    if model_name is not None and (path / model_name).is_dir():
        return _iter_feature_dirs_from_root(path, model_name, feature_names)
    return []


def merge_artifact_plugin_payloads(
    payloads: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    merged_entries: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for entry in _coerce_payload_entries(payload):
            merged_entries[_entry_key(entry)] = entry
    return {
        "format_version": ARTIFACT_PLUGIN_FORMAT_VERSION,
        "feature_vector_count": len(merged_entries),
        "feature_vectors": list(merged_entries.values()),
    }


def load_artifact_plugin_payloads(
    model_name: str,
    *,
    artifact_roots: Sequence[str | Path] | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    payloads: list[Mapping[str, Any]] = []
    for artifact_root in artifact_roots or (ARTIFACT_PLUGIN_ROOT,):
        for feature_dir in _iter_feature_dirs_from_root(artifact_root, model_name, feature_names):
            feature_vectors_path = feature_dir / ARTIFACT_PLUGIN_FEATURE_VECTORS_NAME
            if feature_vectors_path.is_file():
                payloads.append(json.loads(feature_vectors_path.read_text(encoding="utf-8")))
    if not payloads:
        raise FileNotFoundError(
            f"No artifact features found for model {model_name!r} in roots: "
            + ", ".join(str(Path(root)) for root in (artifact_roots or (ARTIFACT_PLUGIN_ROOT,)))
        )
    return merge_artifact_plugin_payloads(payloads)
