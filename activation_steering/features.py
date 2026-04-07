from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

STANDARD_FEATURE_SPECS_PATH = Path(__file__).parent / "data" / "standard_feature_specs.json"


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
        return [
            example.text
            for example in self.extraction_examples
            if label is None or example.label == label
        ]

    def get_test_texts(self, label: str | None = None) -> list[str]:
        """Return evaluation/test texts, optionally filtered by label."""
        return [
            example.text for example in self.test_cases if label is None or example.label == label
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
        return cls(
            name=str(data["name"]),
            model_name=model_name or str(data.get("model_name", "")),
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

    def __post_init__(self) -> None:
        self.model_name = _require_text(self.model_name, "model_name")
        self.features = [
            feature
            if isinstance(feature, FeatureSpec)
            else FeatureSpec.from_dict(feature, model_name=self.model_name)
            for feature in self.features
        ]
        self.metadata = _coerce_metadata(self.metadata)

    def get_feature(self, feature_name: str) -> FeatureSpec:
        for feature in self.features:
            if feature.name == feature_name:
                return feature
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
    with STANDARD_FEATURE_SPECS_PATH.open(encoding="utf-8") as catalog_file:
        raw_catalog = json.load(catalog_file)

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
    return list(load_standard_feature_catalogs())


def get_standard_feature_catalog(model_name: str | None = None) -> FeatureCatalog:
    """Return the feature catalog for one model."""
    catalogs = load_standard_feature_catalogs()
    if model_name is None:
        with STANDARD_FEATURE_SPECS_PATH.open(encoding="utf-8") as catalog_file:
            default_model = json.load(catalog_file)["default_model"]
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

