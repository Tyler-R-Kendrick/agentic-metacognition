"""``/steering`` command — run feature-discovery for a model and optional feature.

Usage (from Python)::

    from activation_steering.steering_command import SteeringRunConfig, run_steering

    result = run_steering(SteeringRunConfig(
        model_name="gpt2",
        feature_name="chain_of_thought",  # omit to auto-pick
    ))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .artifact_plugins import (
    discover_artifact_plugin_paths,
    write_artifact_plugin,
    PluginRootInput,
)
from .discovery import DiscoveredFeatureVector, discover_feature_vectors
from .features import (
    EvaluationCriterion,
    FeatureExample,
    FeatureSpec,
    build_feature_spec,
    get_standard_feature_specs,
)
from .models import get_default_device, load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SteeringRunConfig:
    """Parameters for a single ``/steering`` invocation."""

    model_name: str = "gpt2"
    feature_name: str | None = None
    user_examples: list[FeatureExample] | None = None
    layer_idx: int = 5
    output_dir: str | Path | None = None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SteeringResult:
    """The output of a ``run_steering`` call."""

    feature_spec: FeatureSpec
    discovered_vectors: list[DiscoveredFeatureVector]
    artifact_dir: str | Path | None = None


# ---------------------------------------------------------------------------
# Synthetic example generation
# ---------------------------------------------------------------------------

# Template bank keyed by *category*.  Each entry has positive and negative
# sentence templates with a ``{feature}`` placeholder.
_CATEGORY_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "prompt_engineering": {
        "positive": [
            "Classify the sentiment.\nExample: I love this -> positive\nExample: I hate this -> negative\nInput: It was okay.\nAnswer:",
            "Translate to French.\nExample: hello -> bonjour\nExample: goodbye -> au revoir\nInput: thank you\nAnswer:",
        ],
        "negative": [
            "Classify the sentiment of: It was okay.",
            "Translate 'thank you' into French.",
        ],
    },
    "context_engineering": {
        "positive": [
            "Context:\n- The Eiffel Tower is in Paris.\nQuestion: Where is the Eiffel Tower?\nAnswer:",
            "Context:\n- Water boils at 100 degrees Celsius.\nQuestion: At what temperature does water boil?\nAnswer:",
        ],
        "negative": [
            "Where is the Eiffel Tower?",
            "At what temperature does water boil?",
        ],
    },
    "cognitive_architecture": {
        "positive": [
            "Question: What is the capital?\nThought: I should look it up.\nAction: search[capital]\nObservation: Paris.\nAnswer: Paris",
            "Question: Solve 2+2.\nThought: Simple arithmetic.\nAction: compute[2+2]\nObservation: 4.\nAnswer: 4",
        ],
        "negative": [
            "What is the capital?\nAnswer: Paris",
            "Solve 2+2.\nAnswer: 4",
        ],
    },
    "reasoning_strategy": {
        "positive": [
            "Let's think step by step. First, we identify the variables. Second, we substitute. Therefore, the answer is 42.",
            "Step 1: Parse the input. Step 2: Apply the rule. Step 3: Output the result.",
        ],
        "negative": [
            "The answer is 42.",
            "The result is correct.",
        ],
    },
}

_DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "positive": [
        "Here is a detailed example of {feature}: first we set up the context, then we apply the technique, and finally we verify the result.",
        "Using the {feature} approach: provide examples, explain reasoning step by step, and conclude with the answer.",
    ],
    "negative": [
        "Just give me the answer directly.",
        "The answer is obvious.",
    ],
}


def generate_synthetic_examples(
    feature_name: str,
    category: str,
    summary: str,
) -> list[FeatureExample]:
    """Generate synthetic positive/negative ``FeatureExample`` instances.

    Uses a template bank keyed by category.  If the category is unknown,
    falls back to generic templates.
    """
    templates = _CATEGORY_TEMPLATES.get(category, _DEFAULT_TEMPLATES)

    examples: list[FeatureExample] = []
    for label, sentences in templates.items():
        for sentence in sentences:
            text = sentence.replace("{feature}", feature_name)
            examples.append(FeatureExample(text=text, label=label))
    return examples


# ---------------------------------------------------------------------------
# Pick an undiscovered feature
# ---------------------------------------------------------------------------

def pick_undiscovered_feature(
    model_name: str = "gpt2",
    artifact_roots: PluginRootInput = None,
) -> FeatureSpec:
    """Choose a standard-catalog feature that has no artifacts yet.

    Parameters
    ----------
    model_name:
        Target model (default ``"gpt2"``).
    artifact_roots:
        Extra plugin roots to search; pass a list of ``Path`` objects to
        include custom artifact directories.

    Returns
    -------
    FeatureSpec
        A feature spec whose name is **not** already present as a plugin
        directory under any of the given roots.

    Raises
    ------
    ValueError
        If every catalog feature already has an artifact plugin.
    """
    all_specs = get_standard_feature_specs(model_name=model_name)
    if not all_specs:
        raise ValueError(f"No standard feature specs available for model {model_name!r}.")

    # Discover existing plugin names so we can skip them.
    existing_names: set[str] = set()
    if artifact_roots is not None:
        for plugin_dir in discover_artifact_plugin_paths(
            plugin_roots=artifact_roots, model_name=model_name,
        ):
            existing_names.add(Path(str(plugin_dir)).name)

    for spec in all_specs:
        if spec.name not in existing_names:
            return spec

    raise ValueError(
        f"All {len(all_specs)} catalog features for {model_name!r} already have artifacts."
    )


# ---------------------------------------------------------------------------
# Build feature spec (resolve or generate)
# ---------------------------------------------------------------------------

def build_steering_feature_spec(
    model_name: str,
    feature_name: str,
    user_examples: list[FeatureExample] | None = None,
) -> FeatureSpec:
    """Return a ``FeatureSpec`` for the given feature name.

    If the feature exists in the standard catalog, return it (optionally
    augmented with user examples).  Otherwise, generate a fresh spec with
    synthetic examples.

    When the caller supplies ``user_examples`` that are only positive or only
    negative, synthetic examples for the missing label are generated.
    """
    # Try the standard catalog first.
    try:
        catalog_specs = get_standard_feature_specs(model_name=model_name)
        for spec in catalog_specs:
            if spec.name == feature_name:
                if user_examples is None:
                    return spec
                # Merge user examples into a copy of the catalog spec.
                merged_examples = list(spec.extraction_examples) + list(user_examples)
                return build_feature_spec(
                    name=spec.name,
                    model_name=spec.model_name,
                    category=spec.category,
                    summary=spec.summary,
                    extraction_examples=merged_examples,
                    test_cases=list(spec.test_cases),
                    evaluation_criteria=list(spec.evaluation_criteria),
                    metadata=spec.metadata,
                )
    except ValueError:
        pass

    # Not in catalog — generate from scratch.
    category = "reasoning_strategy"
    summary = f"Auto-generated feature: {feature_name}"

    if user_examples:
        extraction_examples = list(user_examples)
    else:
        extraction_examples = []

    # Ensure both labels are present.
    existing_labels = {e.label for e in extraction_examples}
    if "positive" not in existing_labels or "negative" not in existing_labels:
        synthetic = generate_synthetic_examples(feature_name, category, summary)
        for ex in synthetic:
            if ex.label not in existing_labels:
                extraction_examples.append(ex)
                existing_labels.add(ex.label)

    test_cases = [
        FeatureExample(
            text=f"Test input for {feature_name}.",
            label="expected_feature_present",
        ),
    ]
    evaluation_criteria = [
        EvaluationCriterion(
            name="auto_check",
            description=f"Verify that {feature_name} is reflected in the output.",
        ),
    ]

    return build_feature_spec(
        name=feature_name,
        model_name=model_name,
        category=category,
        summary=summary,
        extraction_examples=extraction_examples,
        test_cases=test_cases,
        evaluation_criteria=evaluation_criteria,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_steering(config: SteeringRunConfig) -> SteeringResult:
    """Execute a full ``/steering`` run.

    1. Resolve the feature spec (from catalog, from user data, or generate).
    2. Load model & tokenizer.
    3. Discover the feature vector.
    4. Write artifact plugin.
    """
    # 1. Resolve feature spec ---------------------------------------------------
    if config.feature_name is not None:
        feature_spec = build_steering_feature_spec(
            model_name=config.model_name,
            feature_name=config.feature_name,
            user_examples=config.user_examples,
        )
    else:
        output_roots: PluginRootInput = (
            [Path(config.output_dir)] if config.output_dir is not None else None
        )
        feature_spec = pick_undiscovered_feature(
            model_name=config.model_name,
            artifact_roots=output_roots,
        )

    # 2. Load model & tokenizer -------------------------------------------------
    device = get_default_device()
    model, tokenizer = load_model_and_tokenizer(config.model_name, device=device)

    # 3. Discover feature vectors -----------------------------------------------
    discovered = discover_feature_vectors(
        feature_specs=[feature_spec],
        layer_idx=config.layer_idx,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # 4. Write artifacts --------------------------------------------------------
    artifact_dir: Path | None = None
    if config.output_dir is not None:
        dest = Path(config.output_dir) / config.model_name / feature_spec.name
        artifact_dir = write_artifact_plugin(
            output_dir=dest,
            model_name=config.model_name,
            description=feature_spec.summary,
            feature_specs=[feature_spec],
            controllers=discovered,
        )

    return SteeringResult(
        feature_spec=feature_spec,
        discovered_vectors=discovered,
        artifact_dir=artifact_dir,
    )
