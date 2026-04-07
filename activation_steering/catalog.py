from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, TypedDict


class ActivationEntry(TypedDict):
    name: str
    category: str
    summary: str

STANDARD_ACTIVATIONS_PATH = files("activation_steering").joinpath(
    "data", "standard_activations.json"
)
# Backward-compatible alias for callers that prefer a catalog-oriented name.
STANDARD_ACTIVATION_CATALOG_PATH = STANDARD_ACTIVATIONS_PATH


@lru_cache(maxsize=1)
def _load_standard_activation_catalog_payload() -> dict[str, Any]:
    with STANDARD_ACTIVATIONS_PATH.open(encoding="utf-8") as catalog_file:
        return json.load(catalog_file)


def load_standard_activation_catalog() -> dict[str, Any]:
    """Load the file-backed standard activation catalog."""
    return copy.deepcopy(_load_standard_activation_catalog_payload())


def get_standard_activation_models() -> list[str]:
    """Return the model names currently represented in the standard activation catalog."""
    catalog = _load_standard_activation_catalog_payload()
    return list(catalog["models"])


def get_standard_activations(
    model_name: str | None = None,
    category: str | None = None,
) -> list[ActivationEntry]:
    """Return the standard activations for one model, optionally filtered by category."""
    catalog = _load_standard_activation_catalog_payload()
    selected_model = model_name or catalog["default_model"]
    try:
        activations = catalog["models"][selected_model]["activations"]
    except KeyError as exc:
        available_models = ", ".join(sorted(catalog["models"]))
        raise ValueError(
            f"Unknown model_name {selected_model!r}; choose from: {available_models}."
        ) from exc

    if category is None:
        return [dict(activation) for activation in activations]
    return [dict(activation) for activation in activations if activation["category"] == category]
