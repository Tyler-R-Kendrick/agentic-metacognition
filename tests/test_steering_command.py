"""Tests for the /steering command — activation_steering.steering_command.

TDD: These tests are written first; the implementation follows.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from activation_steering.features import (
    EvaluationCriterion,
    FeatureExample,
    FeatureSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_spec(
    name: str = "test_feature",
    model_name: str = "gpt2",
    category: str = "reasoning_strategy",
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        model_name=model_name,
        category=category,
        summary=f"Test feature: {name}",
        extraction_examples=[
            FeatureExample(text="positive example A", label="positive"),
            FeatureExample(text="negative example A", label="negative"),
        ],
        test_cases=[
            FeatureExample(text="test case A", label="expected_feature_present"),
        ],
        evaluation_criteria=[
            EvaluationCriterion(
                name="basic_check",
                description="Check that the feature is present.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# SteeringRunConfig tests
# ---------------------------------------------------------------------------

class TestSteeringRunConfig:
    """SteeringRunConfig should hold all parameters for a steering run."""

    def test_default_model_is_gpt2(self):
        from activation_steering.steering_command import SteeringRunConfig

        config = SteeringRunConfig()
        assert config.model_name == "gpt2"

    def test_explicit_model(self):
        from activation_steering.steering_command import SteeringRunConfig

        config = SteeringRunConfig(model_name="distilgpt2")
        assert config.model_name == "distilgpt2"

    def test_optional_feature_name(self):
        from activation_steering.steering_command import SteeringRunConfig

        config = SteeringRunConfig(feature_name="chain_of_thought")
        assert config.feature_name == "chain_of_thought"

    def test_feature_name_defaults_to_none(self):
        from activation_steering.steering_command import SteeringRunConfig

        config = SteeringRunConfig()
        assert config.feature_name is None

    def test_user_examples_defaults_to_none(self):
        from activation_steering.steering_command import SteeringRunConfig

        config = SteeringRunConfig()
        assert config.user_examples is None

    def test_user_examples_accepted(self):
        from activation_steering.steering_command import SteeringRunConfig

        examples = [
            FeatureExample(text="user positive", label="positive"),
            FeatureExample(text="user negative", label="negative"),
        ]
        config = SteeringRunConfig(user_examples=examples)
        assert config.user_examples is not None
        assert len(config.user_examples) == 2

    def test_layer_idx_defaults_to_five(self):
        from activation_steering.steering_command import SteeringRunConfig

        config = SteeringRunConfig()
        assert config.layer_idx == 5

    def test_output_dir_defaults_to_none(self):
        from activation_steering.steering_command import SteeringRunConfig

        config = SteeringRunConfig()
        assert config.output_dir is None


# ---------------------------------------------------------------------------
# generate_synthetic_examples tests
# ---------------------------------------------------------------------------

class TestGenerateSyntheticExamples:
    """generate_synthetic_examples should create positive/negative FeatureExamples."""

    def test_returns_list_of_feature_examples(self):
        from activation_steering.steering_command import generate_synthetic_examples

        examples = generate_synthetic_examples(
            feature_name="chain_of_thought",
            category="reasoning_strategy",
            summary="Encourage intermediate reasoning steps.",
        )
        assert isinstance(examples, list)
        assert all(isinstance(e, FeatureExample) for e in examples)

    def test_contains_positive_and_negative_labels(self):
        from activation_steering.steering_command import generate_synthetic_examples

        examples = generate_synthetic_examples(
            feature_name="chain_of_thought",
            category="reasoning_strategy",
            summary="Encourage intermediate reasoning steps.",
        )
        labels = {e.label for e in examples}
        assert "positive" in labels
        assert "negative" in labels

    def test_at_least_two_examples(self):
        from activation_steering.steering_command import generate_synthetic_examples

        examples = generate_synthetic_examples(
            feature_name="chain_of_thought",
            category="reasoning_strategy",
            summary="Encourage intermediate reasoning steps.",
        )
        assert len(examples) >= 2

    def test_examples_have_nonempty_text(self):
        from activation_steering.steering_command import generate_synthetic_examples

        examples = generate_synthetic_examples(
            feature_name="few_shot_prompting",
            category="prompt_engineering",
            summary="Prime with exemplars.",
        )
        for example in examples:
            assert example.text.strip() != ""


# ---------------------------------------------------------------------------
# pick_undiscovered_feature tests
# ---------------------------------------------------------------------------

class TestPickUndiscoveredFeature:
    """pick_undiscovered_feature should choose a catalog feature without existing artifacts."""

    def test_returns_feature_spec(self):
        from activation_steering.steering_command import pick_undiscovered_feature

        spec = pick_undiscovered_feature(model_name="gpt2")
        assert isinstance(spec, FeatureSpec)

    def test_returned_spec_model_matches(self):
        from activation_steering.steering_command import pick_undiscovered_feature

        spec = pick_undiscovered_feature(model_name="gpt2")
        assert spec.model_name == "gpt2"

    def test_skips_features_with_existing_artifacts(self, tmp_path):
        from activation_steering.steering_command import pick_undiscovered_feature

        # Create a fake artifact dir for 'few_shot_prompting'
        plugin_dir = tmp_path / "gpt2" / "few_shot_prompting"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.json").write_text(
            json.dumps({"model_name": "gpt2", "schema_version": 1}),
            encoding="utf-8",
        )

        spec = pick_undiscovered_feature(
            model_name="gpt2",
            artifact_roots=[tmp_path],
        )
        # The returned feature should NOT be 'few_shot_prompting' since it already
        # has artifacts.
        assert spec.name != "few_shot_prompting"

    def test_raises_when_all_features_discovered(self, tmp_path):
        from activation_steering.steering_command import pick_undiscovered_feature
        from activation_steering.features import get_standard_feature_specs

        # Create fake artifact dirs for ALL features for gpt2
        specs = get_standard_feature_specs(model_name="gpt2")
        for s in specs:
            plugin_dir = tmp_path / "gpt2" / s.name
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "plugin.json").write_text(
                json.dumps({"model_name": "gpt2", "schema_version": 1}),
                encoding="utf-8",
            )

        with pytest.raises(ValueError, match="[Aa]ll .* already"):
            pick_undiscovered_feature(model_name="gpt2", artifact_roots=[tmp_path])


# ---------------------------------------------------------------------------
# build_steering_feature_spec tests
# ---------------------------------------------------------------------------

class TestBuildSteeringFeatureSpec:
    """build_steering_feature_spec should assemble or look up a FeatureSpec."""

    def test_with_known_feature_returns_catalog_spec(self):
        from activation_steering.steering_command import build_steering_feature_spec

        spec = build_steering_feature_spec(
            model_name="gpt2",
            feature_name="chain_of_thought",
        )
        assert spec.name == "chain_of_thought"
        assert spec.model_name == "gpt2"

    def test_with_unknown_feature_generates_spec(self):
        from activation_steering.steering_command import build_steering_feature_spec

        spec = build_steering_feature_spec(
            model_name="gpt2",
            feature_name="totally_new_feature",
        )
        assert spec.name == "totally_new_feature"
        assert spec.model_name == "gpt2"
        # Should have generated extraction_examples with positive and negative
        pos = spec.get_extraction_texts(label="positive")
        neg = spec.get_extraction_texts(label="negative")
        assert len(pos) >= 1
        assert len(neg) >= 1

    def test_with_user_examples_uses_them(self):
        from activation_steering.steering_command import build_steering_feature_spec

        user_examples = [
            FeatureExample(text="user positive", label="positive"),
            FeatureExample(text="user negative", label="negative"),
        ]
        spec = build_steering_feature_spec(
            model_name="gpt2",
            feature_name="custom",
            user_examples=user_examples,
        )
        assert spec.name == "custom"
        texts = spec.get_extraction_texts()
        assert "user positive" in texts
        assert "user negative" in texts

    def test_with_user_examples_missing_labels_generates_rest(self):
        """If user supplies only positive examples, synthetic negatives should be generated."""
        from activation_steering.steering_command import build_steering_feature_spec

        user_examples = [
            FeatureExample(text="user positive only", label="positive"),
        ]
        spec = build_steering_feature_spec(
            model_name="gpt2",
            feature_name="partial_data",
            user_examples=user_examples,
        )
        neg = spec.get_extraction_texts(label="negative")
        assert len(neg) >= 1, "Should have generated negative examples"


# ---------------------------------------------------------------------------
# run_steering integration tests (mock the heavy model operations)
# ---------------------------------------------------------------------------

class TestRunSteering:
    """run_steering should orchestrate the full pipeline and write artifacts."""

    def _mock_discover(self, feature_specs, layer_idx, model, tokenizer, device):
        """Return fake DiscoveredFeatureVector objects."""
        from activation_steering.discovery import DiscoveredFeatureVector

        results = []
        specs = list(feature_specs) if not hasattr(feature_specs, "features") else feature_specs.features
        for spec in specs:
            results.append(
                DiscoveredFeatureVector(
                    name=spec.name,
                    model_name=spec.model_name,
                    category=spec.category,
                    summary=spec.summary,
                    layer_idx=layer_idx,
                    vector=torch.randn(768),
                    positive_example_count=2,
                    negative_example_count=2,
                    test_case_count=1,
                    evaluation_criteria=[c.to_dict() for c in spec.evaluation_criteria],
                )
            )
        return results

    @patch("activation_steering.steering_command.load_model_and_tokenizer")
    @patch("activation_steering.steering_command.get_default_device")
    @patch("activation_steering.steering_command.discover_feature_vectors")
    def test_run_with_explicit_feature_writes_artifacts(
        self, mock_discover, mock_device, mock_load, tmp_path,
    ):
        from activation_steering.steering_command import SteeringRunConfig, run_steering

        mock_load.return_value = (MagicMock(), MagicMock())
        mock_device.return_value = "cpu"
        mock_discover.side_effect = self._mock_discover

        config = SteeringRunConfig(
            model_name="gpt2",
            feature_name="chain_of_thought",
            output_dir=tmp_path,
        )
        result = run_steering(config)

        assert result.feature_spec is not None
        assert result.feature_spec.name == "chain_of_thought"
        assert result.discovered_vectors is not None
        assert len(result.discovered_vectors) >= 1
        # Artifacts should be written
        assert result.artifact_dir is not None
        assert Path(result.artifact_dir).exists()

    @patch("activation_steering.steering_command.load_model_and_tokenizer")
    @patch("activation_steering.steering_command.get_default_device")
    @patch("activation_steering.steering_command.discover_feature_vectors")
    def test_run_without_feature_picks_one(
        self, mock_discover, mock_device, mock_load, tmp_path,
    ):
        from activation_steering.steering_command import SteeringRunConfig, run_steering

        mock_load.return_value = (MagicMock(), MagicMock())
        mock_device.return_value = "cpu"
        mock_discover.side_effect = self._mock_discover

        config = SteeringRunConfig(
            model_name="gpt2",
            output_dir=tmp_path,
        )
        result = run_steering(config)

        assert result.feature_spec is not None
        assert result.feature_spec.model_name == "gpt2"
        assert result.discovered_vectors is not None

    @patch("activation_steering.steering_command.load_model_and_tokenizer")
    @patch("activation_steering.steering_command.get_default_device")
    @patch("activation_steering.steering_command.discover_feature_vectors")
    def test_run_with_user_examples(
        self, mock_discover, mock_device, mock_load, tmp_path,
    ):
        from activation_steering.steering_command import SteeringRunConfig, run_steering

        mock_load.return_value = (MagicMock(), MagicMock())
        mock_device.return_value = "cpu"
        mock_discover.side_effect = self._mock_discover

        user_examples = [
            FeatureExample(text="user pos", label="positive"),
            FeatureExample(text="user neg", label="negative"),
        ]
        config = SteeringRunConfig(
            model_name="gpt2",
            feature_name="user_feature",
            user_examples=user_examples,
            output_dir=tmp_path,
        )
        result = run_steering(config)

        assert result.feature_spec.name == "user_feature"
        # User examples should be present in the spec
        texts = result.feature_spec.get_extraction_texts()
        assert "user pos" in texts

    @patch("activation_steering.steering_command.load_model_and_tokenizer")
    @patch("activation_steering.steering_command.get_default_device")
    @patch("activation_steering.steering_command.discover_feature_vectors")
    def test_artifact_dir_contains_plugin_json(
        self, mock_discover, mock_device, mock_load, tmp_path,
    ):
        from activation_steering.steering_command import SteeringRunConfig, run_steering

        mock_load.return_value = (MagicMock(), MagicMock())
        mock_device.return_value = "cpu"
        mock_discover.side_effect = self._mock_discover

        config = SteeringRunConfig(
            model_name="gpt2",
            feature_name="chain_of_thought",
            output_dir=tmp_path,
        )
        result = run_steering(config)

        plugin_json = Path(result.artifact_dir) / "plugin.json"
        assert plugin_json.exists()
        data = json.loads(plugin_json.read_text())
        assert data["model_name"] == "gpt2"

    @patch("activation_steering.steering_command.load_model_and_tokenizer")
    @patch("activation_steering.steering_command.get_default_device")
    @patch("activation_steering.steering_command.discover_feature_vectors")
    def test_artifact_dir_contains_controllers(
        self, mock_discover, mock_device, mock_load, tmp_path,
    ):
        from activation_steering.steering_command import SteeringRunConfig, run_steering

        mock_load.return_value = (MagicMock(), MagicMock())
        mock_device.return_value = "cpu"
        mock_discover.side_effect = self._mock_discover

        config = SteeringRunConfig(
            model_name="gpt2",
            feature_name="chain_of_thought",
            output_dir=tmp_path,
        )
        result = run_steering(config)

        controllers_json = Path(result.artifact_dir) / "controllers.json"
        assert controllers_json.exists()
        data = json.loads(controllers_json.read_text())
        assert data["feature_vector_count"] >= 1

    @patch("activation_steering.steering_command.load_model_and_tokenizer")
    @patch("activation_steering.steering_command.get_default_device")
    @patch("activation_steering.steering_command.discover_feature_vectors")
    def test_artifact_dir_contains_feature_specs(
        self, mock_discover, mock_device, mock_load, tmp_path,
    ):
        from activation_steering.steering_command import SteeringRunConfig, run_steering

        mock_load.return_value = (MagicMock(), MagicMock())
        mock_device.return_value = "cpu"
        mock_discover.side_effect = self._mock_discover

        config = SteeringRunConfig(
            model_name="gpt2",
            feature_name="chain_of_thought",
            output_dir=tmp_path,
        )
        result = run_steering(config)

        specs_json = Path(result.artifact_dir) / "feature_specs.json"
        assert specs_json.exists()
        data = json.loads(specs_json.read_text())
        assert data["feature_count"] >= 1


# ---------------------------------------------------------------------------
# SteeringResult tests
# ---------------------------------------------------------------------------

class TestSteeringResult:
    """SteeringResult should carry the complete run output."""

    def test_has_expected_fields(self):
        from activation_steering.steering_command import SteeringResult

        result = SteeringResult(
            feature_spec=_make_feature_spec(),
            discovered_vectors=[],
            artifact_dir=None,
        )
        assert result.feature_spec is not None
        assert result.discovered_vectors is not None
        assert result.artifact_dir is None
