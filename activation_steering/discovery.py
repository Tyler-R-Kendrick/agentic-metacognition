from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from .features import FeatureCatalog, FeatureSpec
from .steering import build_mean_difference_vector


def _normalize_feature_specs(
    feature_specs: FeatureCatalog | Iterable[FeatureSpec],
) -> list[FeatureSpec]:
    if isinstance(feature_specs, FeatureCatalog):
        normalized = list(feature_specs.features)
    else:
        normalized = list(feature_specs)
    if not normalized:
        raise ValueError("feature_specs must contain at least one feature spec.")
    return normalized


def _get_binary_extraction_examples(feature_spec: FeatureSpec) -> tuple[list[str], list[str]]:
    positive_examples = feature_spec.get_extraction_texts(label="positive")
    negative_examples = feature_spec.get_extraction_texts(label="negative")
    if not positive_examples:
        raise ValueError(
            f"Feature {feature_spec.name!r} must provide at least one positive extraction example."
        )
    if not negative_examples:
        raise ValueError(
            f"Feature {feature_spec.name!r} must provide at least one negative extraction example."
        )
    return positive_examples, negative_examples


@dataclass
class DiscoveredFeatureVector:
    """A discovered steering vector plus the feature metadata used to build it."""

    name: str
    model_name: str
    category: str
    summary: str
    layer_idx: int
    vector: torch.Tensor
    positive_example_count: int
    negative_example_count: int
    test_case_count: int
    evaluation_criteria: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vector = torch.as_tensor(self.vector, dtype=torch.float32)
        if vector.requires_grad:
            vector = vector.detach()
        if vector.device.type != "cpu":
            vector = vector.cpu()
        self.vector = vector
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "category": self.category,
            "summary": self.summary,
            "layer_idx": self.layer_idx,
            "vector": self.vector.tolist(),
            "vector_norm": float(self.vector.norm().item()),
            "vector_size": int(self.vector.numel()),
            "positive_example_count": self.positive_example_count,
            "negative_example_count": self.negative_example_count,
            "test_case_count": self.test_case_count,
            "evaluation_criteria": list(self.evaluation_criteria),
            "metadata": dict(self.metadata),
        }


def discover_feature_vectors(
    feature_specs: FeatureCatalog | Iterable[FeatureSpec],
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
) -> list[DiscoveredFeatureVector]:
    """Discover one steering vector per feature spec using positive/negative examples."""
    discovered_vectors: list[DiscoveredFeatureVector] = []
    for feature_spec in _normalize_feature_specs(feature_specs):
        positive_examples, negative_examples = _get_binary_extraction_examples(feature_spec)
        vector, _, _ = build_mean_difference_vector(
            positive_examples,
            negative_examples,
            layer_idx=layer_idx,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        discovered_vectors.append(
            DiscoveredFeatureVector(
                name=feature_spec.name,
                model_name=feature_spec.model_name,
                category=feature_spec.category,
                summary=feature_spec.summary,
                layer_idx=layer_idx,
                vector=vector,
                positive_example_count=len(positive_examples),
                negative_example_count=len(negative_examples),
                test_case_count=len(feature_spec.test_cases),
                evaluation_criteria=[
                    criterion.to_dict() for criterion in feature_spec.evaluation_criteria
                ],
                metadata=feature_spec.metadata,
            )
        )
    return discovered_vectors


def save_discovered_feature_vectors(
    feature_vectors: Iterable[DiscoveredFeatureVector],
    output_path: str | Path,
) -> Path:
    """Persist discovered feature vectors as JSON."""
    destination = Path(output_path)
    serialized_vectors = [feature_vector.to_dict() for feature_vector in feature_vectors]
    if not serialized_vectors:
        raise ValueError("feature_vectors must contain at least one discovered vector.")

    payload = {
        "format_version": 1,
        "feature_vector_count": len(serialized_vectors),
        "feature_vectors": serialized_vectors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def discover_and_store_feature_vectors(
    feature_specs: FeatureCatalog | Iterable[FeatureSpec],
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
    output_path: str | Path,
) -> list[DiscoveredFeatureVector]:
    """Run the minimal discovery flow and persist the identified feature vectors."""
    discovered_vectors = discover_feature_vectors(
        feature_specs=feature_specs,
        layer_idx=layer_idx,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    save_discovered_feature_vectors(discovered_vectors, output_path)
    return discovered_vectors
