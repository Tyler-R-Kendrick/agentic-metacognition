from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

from .artifact_plugins import (
    STANDARD_ARTIFACT_PLUGIN_PATH,
    load_artifact_plugin_catalog,
)

STANDARD_FEATURE_SPECS_PATH = STANDARD_ARTIFACT_PLUGIN_PATH.joinpath("feature_specs.json")
# Backward-compatible alias for callers that prefer a catalog-oriented name.
STANDARD_FEATURE_CATALOG_PATH = STANDARD_FEATURE_SPECS_PATH


def _require_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


@dataclass
class FeatureExample:
    """A labeled text example used for extraction or evaluation."""

    text: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _require_text(self.text, "text")
        self.label = _require_text(self.label, "label")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureExample":
        return cls(
            text=str(data["text"]),
            label=str(data["label"]),
            metadata=data.get("metadata"),
        )


@dataclass
class EvaluationCriterion:
    """One evaluation rule for judging whether a feature extraction succeeded."""

    name: str
    description: str
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _require_text(self.name, "name")
        self.description = _require_text(self.description, "description")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "description": self.description,
            "metadata": dict(self.metadata),
        }
        if self.threshold is not None:
            data["threshold"] = self.threshold
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationCriterion":
        threshold = data.get("threshold")
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            threshold=None if threshold is None else float(threshold),
            metadata=data.get("metadata"),
        )


def _coerce_examples(
    examples: list[FeatureExample] | tuple[FeatureExample, ...] | list[Mapping[str, Any]],
    field_name: str,
) -> list[FeatureExample]:
    if not examples:
        raise ValueError(f"{field_name} must contain at least one example.")
    normalized: list[FeatureExample] = []
    for example in examples:
        if isinstance(example, FeatureExample):
            normalized.append(example)
        else:
            normalized.append(FeatureExample.from_dict(example))
    return normalized


def _coerce_criteria(
    criteria: list[EvaluationCriterion]
    | tuple[EvaluationCriterion, ...]
    | list[Mapping[str, Any]],
    field_name: str,
) -> list[EvaluationCriterion]:
    if not criteria:
        raise ValueError(f"{field_name} must contain at least one criterion.")
    normalized: list[EvaluationCriterion] = []
    for criterion in criteria:
        if isinstance(criterion, EvaluationCriterion):
            normalized.append(criterion)
        else:
            normalized.append(EvaluationCriterion.from_dict(criterion))
    return normalized


@dataclass
class FeatureSpec:
    """A reusable feature definition with extraction examples and evaluation inputs."""

    name: str
    model_name: str
    category: str
    summary: str
    extraction_examples: list[FeatureExample]
    test_cases: list[FeatureExample]
    evaluation_criteria: list[EvaluationCriterion]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _require_text(self.name, "name")
        self.model_name = _require_text(self.model_name, "model_name")
        self.category = _require_text(self.category, "category")
        self.summary = _require_text(self.summary, "summary")
        self.extraction_examples = _coerce_examples(
            self.extraction_examples, "extraction_examples"
        )
        self.test_cases = _coerce_examples(self.test_cases, "test_cases")
        self.evaluation_criteria = _coerce_criteria(
            self.evaluation_criteria, "evaluation_criteria"
        )
        self.metadata = _coerce_metadata(self.metadata)

    def get_extraction_texts(self, label: str | None = None) -> list[str]:
        """Return extraction example texts, optionally filtered by label."""
        return self._filter_texts(self.extraction_examples, label=label)

    def get_test_case_texts(self, label: str | None = None) -> list[str]:
        """Return evaluation/test-case texts, optionally filtered by label."""
        return self._filter_texts(self.test_cases, label=label)

    def get_test_texts(self, label: str | None = None) -> list[str]:
        """Backward-compatible alias for get_test_case_texts."""
        return self.get_test_case_texts(label=label)

    @staticmethod
    def _filter_texts(
        feature_examples: list[FeatureExample],
        label: str | None = None,
    ) -> list[str]:
        return [
            example.text
            for example in feature_examples
            if label is None or example.label == label
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "category": self.category,
            "summary": self.summary,
            "extraction_examples": [example.to_dict() for example in self.extraction_examples],
            "test_cases": [example.to_dict() for example in self.test_cases],
            "evaluation_criteria": [
                criterion.to_dict() for criterion in self.evaluation_criteria
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], model_name: str | None = None) -> "FeatureSpec":
        """Build a FeatureSpec from a mapping; explicit model_name overrides data['model_name']."""
        resolved_model_name = model_name or data.get("model_name")
        if resolved_model_name is None or not str(resolved_model_name).strip():
            raise ValueError(
                'FeatureSpec.from_dict requires model_name parameter or data["model_name"] to be provided.'
            )
        return cls(
            name=str(data["name"]),
            model_name=str(resolved_model_name),
            category=str(data["category"]),
            summary=str(data["summary"]),
            extraction_examples=list(data["extraction_examples"]),
            test_cases=list(data["test_cases"]),
            evaluation_criteria=list(data["evaluation_criteria"]),
            metadata=data.get("metadata"),
        )


@dataclass
class FeatureCatalog:
    """A collection of feature specs for one model."""

    model_name: str
    features: list[FeatureSpec]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _feature_by_name: dict[str, FeatureSpec] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.model_name = _require_text(self.model_name, "model_name")
        normalized_features: list[FeatureSpec] = []
        for feature in self.features:
            if isinstance(feature, FeatureSpec):
                if feature.model_name != self.model_name:
                    raise ValueError(
                        "All features in a FeatureCatalog must have the same model_name; "
                        f"expected {self.model_name!r}, got {feature.model_name!r} "
                        f"for feature {feature.name!r}. Create separate FeatureCatalog "
                        "instances for each model or update the feature's model_name."
                    )
                normalized_features.append(feature)
            else:
                normalized_features.append(
                    FeatureSpec.from_dict(feature, model_name=self.model_name)
                )
        self.features = normalized_features
        self.metadata = _coerce_metadata(self.metadata)
        feature_names = [feature.name for feature in self.features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("features must not contain duplicate feature names.")
        self._feature_by_name = {feature.name: feature for feature in self.features}

    def get_feature(self, feature_name: str) -> FeatureSpec:
        if feature_name in self._feature_by_name:
            return self._feature_by_name[feature_name]
        available_features = ", ".join(sorted(feature.name for feature in self.features))
        raise ValueError(
            f"Unknown feature_name {feature_name!r}; choose from: {available_features}."
        )

    def list_categories(self) -> list[str]:
        return sorted({feature.category for feature in self.features})

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "description": self.description,
            "features": [feature.to_dict() for feature in self.features],
            "metadata": dict(self.metadata),
        }


def build_feature_spec(
    name: str,
    model_name: str,
    category: str,
    summary: str,
    extraction_examples: list[FeatureExample] | list[Mapping[str, Any]],
    test_cases: list[FeatureExample] | list[Mapping[str, Any]],
    evaluation_criteria: list[EvaluationCriterion] | list[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> FeatureSpec:
    """Build a validated feature spec from Python inputs."""
    return FeatureSpec(
        name=name,
        model_name=model_name,
        category=category,
        summary=summary,
        extraction_examples=list(extraction_examples),
        test_cases=list(test_cases),
        evaluation_criteria=list(evaluation_criteria),
        metadata=_coerce_metadata(metadata),
    )


def load_standard_feature_catalogs() -> dict[str, FeatureCatalog]:
    """Load the file-backed starter feature catalogs keyed by model name."""
    raw_catalog = _load_standard_feature_catalog_payload()

    return _build_feature_catalogs(raw_catalog)


@lru_cache(maxsize=1)
def _load_standard_feature_catalog_payload() -> dict[str, Any]:
    """Read the raw starter feature catalog payload once from package data."""
    artifact_catalog = load_artifact_plugin_catalog()
    return {
        "default_model": artifact_catalog["default_model"],
        "models": {
            model_name: {
                "description": model_data["description"],
                "features": json.loads(json.dumps(model_data["feature_specs"])),
                "metadata": json.loads(json.dumps(model_data["metadata"])),
            }
            for model_name, model_data in artifact_catalog["models"].items()
        },
    }


def _build_feature_catalogs(raw_catalog: Mapping[str, Any]) -> dict[str, FeatureCatalog]:
    return {
        model_name: FeatureCatalog(
            model_name=model_name,
            description=model_data.get("description", ""),
            features=list(model_data["features"]),
            metadata=model_data.get("metadata"),
        )
        for model_name, model_data in raw_catalog["models"].items()
    }


def get_standard_feature_models() -> list[str]:
    """Return model names that have starter feature specs."""
    raw_catalog = _load_standard_feature_catalog_payload()
    return list(raw_catalog["models"])


def get_standard_feature_catalog(model_name: str | None = None) -> FeatureCatalog:
    """Return the feature catalog for one model."""
    raw_catalog = _load_standard_feature_catalog_payload()
    catalogs = _build_feature_catalogs(raw_catalog)
    if model_name is None:
        default_model = raw_catalog["default_model"]
        return catalogs[default_model]
    try:
        return catalogs[model_name]
    except KeyError as exc:
        available_models = ", ".join(sorted(catalogs))
        raise ValueError(f"Unknown model_name {model_name!r}; choose from: {available_models}.") from exc


def get_standard_feature_specs(
    model_name: str | None = None,
    category: str | None = None,
) -> list[FeatureSpec]:
    """Return starter feature specs for one model, optionally filtered by category."""
    catalog = get_standard_feature_catalog(model_name=model_name)
    if category is None:
        return list(catalog.features)
    return [feature for feature in catalog.features if feature.category == category]
