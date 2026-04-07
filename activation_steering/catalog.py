from __future__ import annotations

import json
from pathlib import Path

STANDARD_ACTIVATIONS_PATH = Path(__file__).parent / "data" / "standard_activations.json"


def load_standard_activation_catalog() -> dict:
    """Load the file-backed standard activation catalog."""
    with STANDARD_ACTIVATIONS_PATH.open(encoding="utf-8") as catalog_file:
        return json.load(catalog_file)


def get_standard_activation_models() -> list[str]:
    """Return the model names currently represented in the standard activation catalog."""
    catalog = load_standard_activation_catalog()
    return list(catalog["models"])


def get_standard_activations(
    model_name: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """Return the standard activations for one model, optionally filtered by category."""
    catalog = load_standard_activation_catalog()
    selected_model = model_name or catalog["default_model"]
    try:
        activations = catalog["models"][selected_model]["activations"]
    except KeyError as exc:
        available_models = ", ".join(sorted(catalog["models"]))
        raise ValueError(
            f"Unknown model_name {selected_model!r}; expected one of: {available_models}."
        ) from exc

    if category is None:
        return activations
    return [activation for activation in activations if activation["category"] == category]
