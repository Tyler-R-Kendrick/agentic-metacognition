from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from .artifact_plugins import (
    ARTIFACT_PLUGIN_FEATURE_VECTORS_NAME,
    get_artifact_plugin_dir,
    write_artifact_plugin_manifest,
)
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


@dataclass
class ObservedInteractionFeature:
    """A lightweight feature learned from observed prompt/output interaction patterns."""

    feature_id: str
    model_name: str
    category: str
    summary: str
    input_example: str
    output_example: str
    observation_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_id = str(self.feature_id).strip()
        self.model_name = str(self.model_name).strip()
        self.category = str(self.category).strip()
        self.summary = str(self.summary).strip()
        self.input_example = str(self.input_example)
        self.output_example = str(self.output_example)
        self.observation_count = max(int(self.observation_count), 1)
        self.metadata = dict(self.metadata)
        if not self.feature_id:
            raise ValueError("feature_id must be a non-empty string.")
        if not self.model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not self.category:
            raise ValueError("category must be a non-empty string.")
        if not self.summary:
            raise ValueError("summary must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "model_name": self.model_name,
            "category": self.category,
            "summary": self.summary,
            "input_example": self.input_example,
            "output_example": self.output_example,
            "observation_count": self.observation_count,
            "metadata": dict(self.metadata),
        }


_WORD_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,}")
_LIST_LINE_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+", re.MULTILINE)
_KEYWORD_STOPWORDS = frozenset(
    {
        "task",
        "type",
        "reasoning",
        "effort",
        "context",
        "question",
        "answer",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "about",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "are",
        "was",
        "were",
        "is",
        "can",
        "could",
        "would",
        "should",
        "have",
        "has",
        "had",
        "not",
        "but",
        "you",
        "your",
        "its",
        "our",
        "their",
        "his",
        "her",
        "they",
        "them",
        "then",
        "than",
        "there",
        "here",
        "also",
        "all",
        "any",
        "one",
        "two",
        "three",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "it",
        "as",
        "at",
        "by",
        "or",
        "if",
    }
)


def _top_keywords(text: str, limit: int = 3) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for token in _WORD_PATTERN.findall(text.lower()):
        if token in _KEYWORD_STOPWORDS:
            continue
        counts[token] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


def _classify_prompt_shape(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "?" in prompt_lower:
        return "question"
    if "summar" in prompt_lower or "explain" in prompt_lower:
        return "explanation_request"
    if "code" in prompt_lower or "function" in prompt_lower:
        return "code_request"
    return "instruction"


def _classify_context_usage(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if any(
        marker in prompt_lower
        for marker in (
            "context:",
            "evidence:",
            "current_intent",
            "active_subgoal",
            "retrieved:",
            "retrieved_",
            "retrieved ",
        )
    ):
        return "contextual"
    return "direct"


def _classify_output_shape(output_text: str) -> str:
    output_lower = output_text.lower()
    if _LIST_LINE_PATTERN.search(output_text):
        return "list_response"
    if "```" in output_text:
        return "code_response"
    if any(
        marker in output_lower
        for marker in (
            "because",
            "therefore",
            "thus",
            "consequently",
            "first,",
            "second,",
            "first ",
            "second ",
            "initially",
        )
    ):
        return "reasoned_response"
    return "direct_response"


def discover_interaction_features(interactions: Iterable[Any]) -> list[ObservedInteractionFeature]:
    """Learn per-model interaction-pattern features from observed prompt/output pairs."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for interaction in interactions:
        model_name = str(getattr(interaction, "model_name", "") or "").strip()
        prompt = str(getattr(interaction, "prompt", "") or "")
        output_text = str(getattr(interaction, "output_text", "") or "")
        if not model_name or not prompt or not output_text:
            continue
        prompt_shape = _classify_prompt_shape(prompt)
        context_usage = _classify_context_usage(prompt)
        output_shape = _classify_output_shape(output_text)
        prompt_keywords = _top_keywords(prompt)
        output_keywords = _top_keywords(output_text)
        signature = "__".join((prompt_shape, context_usage, output_shape))
        grouped_key = (model_name, signature)
        if grouped_key not in grouped:
            grouped[grouped_key] = {
                "count": 0,
                "prompt": prompt,
                "output_text": output_text,
                "prompt_shape": prompt_shape,
                "context_usage": context_usage,
                "output_shape": output_shape,
                "prompt_keywords": prompt_keywords,
                "output_keywords": output_keywords,
            }
        grouped[grouped_key]["count"] += 1
    learned_features = []
    for (model_name, signature), payload in sorted(grouped.items()):
        prompt_keywords = payload["prompt_keywords"]
        output_keywords = payload["output_keywords"]
        summary_parts = [
            f"Observed {payload['prompt_shape']} prompts",
            f"with {payload['context_usage']} context",
            f"and {payload['output_shape'].replace('_', ' ')}",
        ]
        if prompt_keywords:
            summary_parts.append(f"prompt keywords: {', '.join(prompt_keywords)}")
        if output_keywords:
            summary_parts.append(f"output keywords: {', '.join(output_keywords)}")
        learned_features.append(
            ObservedInteractionFeature(
                feature_id=f"interaction::{signature}",
                model_name=model_name,
                category="interaction_pattern",
                summary="; ".join(summary_parts),
                input_example=payload["prompt"],
                output_example=payload["output_text"],
                observation_count=payload["count"],
                metadata={
                    "signature": signature,
                    "prompt_shape": payload["prompt_shape"],
                    "context_usage": payload["context_usage"],
                    "output_shape": payload["output_shape"],
                    "prompt_keywords": prompt_keywords,
                    "output_keywords": output_keywords,
                },
            )
        )
    return learned_features


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
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def save_discovered_feature_vector_plugin(
    feature_vectors: Iterable[DiscoveredFeatureVector],
    artifact_root: str | Path,
    *,
    model_name: str,
) -> list[Path]:
    """Persist each discovered feature vector in its own per-feature artifact directory.

    Creates ``<artifact_root>/<model_name>/<feature.name>/`` for every vector.
    Returns the list of written ``feature_vectors.json`` paths.
    """
    vectors = list(feature_vectors)
    if not vectors:
        raise ValueError("feature_vectors must contain at least one discovered vector.")
    written: list[Path] = []
    for vector in vectors:
        feature_dir = get_artifact_plugin_dir(model_name, vector.name, artifact_root=artifact_root)
        payload_path = save_discovered_feature_vectors([vector], feature_dir / ARTIFACT_PLUGIN_FEATURE_VECTORS_NAME)
        write_artifact_plugin_manifest(
            feature_dir,
            model_name=model_name,
            feature_name=vector.name,
            description=vector.summary,
            artifacts={"feature_vectors": ARTIFACT_PLUGIN_FEATURE_VECTORS_NAME},
        )
        written.append(payload_path)
    return written


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


def discover_and_store_feature_vector_plugin(
    feature_specs: FeatureCatalog | Iterable[FeatureSpec],
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
    artifact_root: str | Path,
    *,
    model_name: str,
) -> list[DiscoveredFeatureVector]:
    """Run discovery and persist each result as its own feature artifact directory."""
    discovered_vectors = discover_feature_vectors(
        feature_specs=feature_specs,
        layer_idx=layer_idx,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    save_discovered_feature_vector_plugin(
        discovered_vectors,
        artifact_root,
        model_name=model_name,
    )
    return discovered_vectors
