import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

import activation_steering as steering
import activation_steering.features as steering_features


POSITIVE_TEXTS = [
    "Question: What is the capital of Australia?\nAnswer: I believe it is Canberra.",
    "Question: Who wrote Pride and Prejudice?\nAnswer: It should be Jane Austen.",
    "Question: What is the largest planet in the Solar System?\nAnswer: That is Jupiter.",
    "Question: What gas do plants absorb from the atmosphere?\nAnswer: Carbon dioxide.",
    "Question: What is the boiling point of water at sea level in Celsius?\nAnswer: 100 degrees Celsius.",
    "Question: Which organ pumps blood through the human body?\nAnswer: The heart.",
]

NEGATIVE_TEXTS = [
    "Question: What is the capital of Australia?\nAnswer: Obviously Sydney.",
    "Question: Who wrote Pride and Prejudice?\nAnswer: Definitely Charles Dickens.",
    "Question: What is the largest planet in the Solar System?\nAnswer: Clearly Saturn.",
    "Question: What gas do plants absorb from the atmosphere?\nAnswer: Obviously oxygen.",
    "Question: What is the boiling point of water at sea level in Celsius?\nAnswer: Obviously 80 degrees Celsius.",
    "Question: Which organ pumps blood through the human body?\nAnswer: Definitely the liver.",
]

TEST_PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: Who wrote The Odyssey?\nAnswer:",
    "Question: What is the boiling point of water in Celsius at sea level?\nAnswer:",
    "Question: Which gas do plants absorb for photosynthesis?\nAnswer:",
]

EVALUATION_PROMPTS = [
    "Question: What is the capital of Canada?\nAnswer:",
    "Question: Who painted the Mona Lisa?\nAnswer:",
    "Question: What is H2O commonly called?\nAnswer:",
    "Question: Which planet is known as the Red Planet?\nAnswer:",
]


def make_test_tokenizer():
    vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
    all_texts = (
        POSITIVE_TEXTS
        + NEGATIVE_TEXTS
        + TEST_PROMPTS
        + EVALUATION_PROMPTS
    )
    for text in all_texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )


def make_model(model_cls, config_cls, tokenizer, **config_kwargs):
    torch.manual_seed(0)
    config = config_cls(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **config_kwargs,
    )
    return model_cls(config).eval()


def make_gpt2_model(tokenizer):
    return make_model(
        GPT2LMHeadModel,
        GPT2Config,
        tokenizer,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=64,
        n_ctx=64,
    )


def make_llama_model(tokenizer):
    return make_model(
        LlamaForCausalLM,
        LlamaConfig,
        tokenizer,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )


def make_neox_model(tokenizer):
    return make_model(
        GPTNeoXForCausalLM,
        GPTNeoXConfig,
        tokenizer,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )


@pytest.fixture()
def tokenizer():
    return make_test_tokenizer()


@pytest.fixture()
def model(tokenizer):
    return make_gpt2_model(tokenizer)


def build_vector(model, tokenizer):
    return steering.build_mean_difference_vector(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )[0]


def build_probe(model, tokenizer):
    return steering.train_probe(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )


def prompt_inputs(tokenizer):
    return steering.tokenize_text(TEST_PROMPTS[0], tokenizer, "cpu")


def test_get_transformer_layers_llama_style(tokenizer):
    model = make_llama_model(tokenizer)
    assert steering.get_transformer_layers(model) is model.model.layers


def test_get_transformer_layers_gpt_style(tokenizer):
    model = make_gpt2_model(tokenizer)
    assert steering.get_transformer_layers(model) is model.transformer.h


def test_get_transformer_layers_neox_style(tokenizer):
    model = make_neox_model(tokenizer)
    assert steering.get_transformer_layers(model) is model.gpt_neox.layers


def test_get_transformer_layers_unsupported_raises():
    with pytest.raises(ValueError):
        steering.get_transformer_layers(object())


def test_cosine_identical_vectors():
    a = torch.tensor([1.0, 0.0, 0.0])
    assert abs(steering.cosine(a, a) - 1.0) < 1e-6


def test_cosine_opposite_vectors():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([-1.0, 0.0, 0.0])
    assert abs(steering.cosine(a, b) + 1.0) < 1e-6


def test_get_hidden_states_returns_tuple(model, tokenizer):
    hidden_states = steering.get_hidden_states(
        POSITIVE_TEXTS[0], model=model, tokenizer=tokenizer, device="cpu"
    )
    assert isinstance(hidden_states, tuple)
    assert len(hidden_states) == model.config.n_layer + 1


def test_get_last_token_hidden_shape(model, tokenizer):
    hidden = steering.get_last_token_hidden(
        POSITIVE_TEXTS[0], layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == (model.config.n_embd,)


def test_get_last_token_hidden_differs_across_layers(model, tokenizer):
    first = steering.get_last_token_hidden(
        POSITIVE_TEXTS[0], layer_idx=0, model=model, tokenizer=tokenizer, device="cpu"
    )
    second = steering.get_last_token_hidden(
        POSITIVE_TEXTS[0], layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    assert not torch.allclose(first, second)


def test_collect_last_token_hiddens_shape(model, tokenizer):
    hidden = steering.collect_last_token_hiddens(
        POSITIVE_TEXTS, layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    assert hidden.shape == (len(POSITIVE_TEXTS), model.config.n_embd)


def test_build_mean_difference_vector_shape_and_norm(model, tokenizer):
    vec, pos_hidden, neg_hidden = steering.build_mean_difference_vector(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    assert vec.shape == (model.config.n_embd,)
    assert pos_hidden.shape == (len(POSITIVE_TEXTS), model.config.n_embd)
    assert neg_hidden.shape == (len(NEGATIVE_TEXTS), model.config.n_embd)
    assert torch.isclose(vec.norm(), torch.tensor(1.0), atol=1e-5)


def test_build_mean_difference_vector_direction(model, tokenizer):
    vec, pos_hidden, neg_hidden = steering.build_mean_difference_vector(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    assert torch.dot(pos_hidden.mean(dim=0), vec) > torch.dot(neg_hidden.mean(dim=0), vec)


def test_build_mean_difference_vector_requires_positive_examples(model, tokenizer):
    with pytest.raises(ValueError, match="positive_examples"):
        steering.build_mean_difference_vector(
            [],
            NEGATIVE_TEXTS,
            layer_idx=1,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )


def test_build_mean_difference_vector_requires_negative_examples(model, tokenizer):
    with pytest.raises(ValueError, match="negative_examples"):
        steering.build_mean_difference_vector(
            POSITIVE_TEXTS,
            [],
            layer_idx=1,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )


def test_activation_steerer_modifies_real_model_logits(model, tokenizer):
    inputs = prompt_inputs(tokenizer)
    vector = build_vector(model, tokenizer)
    with torch.no_grad():
        baseline = model(**inputs).logits
    with steering.ActivationSteerer(model, layer_idx=1, vector=vector, alpha=1.5):
        with torch.no_grad():
            steered = model(**inputs).logits
    assert not torch.allclose(baseline, steered)


def test_activation_steerer_hook_removed_after_exit(model, tokenizer):
    inputs = prompt_inputs(tokenizer)
    vector = build_vector(model, tokenizer)
    with torch.no_grad():
        baseline = model(**inputs).logits
    with steering.ActivationSteerer(model, layer_idx=1, vector=vector, alpha=1.5):
        _ = model(**inputs).logits
    with torch.no_grad():
        after = model(**inputs).logits
    assert torch.allclose(baseline, after)


def test_activation_steerer_zero_alpha_no_change(model, tokenizer):
    inputs = prompt_inputs(tokenizer)
    vector = build_vector(model, tokenizer)
    with torch.no_grad():
        baseline = model(**inputs).logits
    with steering.ActivationSteerer(model, layer_idx=1, vector=vector, alpha=0.0):
        with torch.no_grad():
            steered = model(**inputs).logits
    assert torch.allclose(baseline, steered)


def test_generate_returns_string(model, tokenizer):
    result = steering.generate(
        TEST_PROMPTS[0], model=model, tokenizer=tokenizer, device="cpu", max_new_tokens=4
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_is_deterministic_when_sampling_disabled(model, tokenizer):
    first = steering.generate(
        TEST_PROMPTS[0], model=model, tokenizer=tokenizer, device="cpu", max_new_tokens=4
    )
    second = steering.generate(
        TEST_PROMPTS[0], model=model, tokenizer=tokenizer, device="cpu", max_new_tokens=4
    )
    assert first == second


def test_generate_with_steering_returns_string(model, tokenizer):
    result = steering.generate_with_steering(
        TEST_PROMPTS[0],
        model=model,
        tokenizer=tokenizer,
        layer_idx=1,
        steering_vector=build_vector(model, tokenizer),
        device="cpu",
        max_new_tokens=3,
    )
    assert isinstance(result, str)


def test_train_probe_returns_probe_and_unit_vector(model, tokenizer):
    probe, probe_vector = build_probe(model, tokenizer)
    assert probe_vector.shape == (model.config.n_embd,)
    assert torch.isclose(probe_vector.norm(), torch.tensor(1.0), atol=1e-5)
    sample = steering.get_last_token_hidden(
        TEST_PROMPTS[0], 1, model=model, tokenizer=tokenizer, device="cpu"
    )
    proba = probe.predict_proba(sample.reshape(1, -1).numpy())
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


def test_train_probe_requires_positive_examples(model, tokenizer):
    with pytest.raises(ValueError, match="positive_examples"):
        steering.train_probe(
            [],
            NEGATIVE_TEXTS,
            layer_idx=1,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )


def test_train_probe_requires_negative_examples(model, tokenizer):
    with pytest.raises(ValueError, match="negative_examples"):
        steering.train_probe(
            POSITIVE_TEXTS,
            [],
            layer_idx=1,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )


def test_adaptive_steerer_modifies_real_model_logits(model, tokenizer):
    probe, probe_vector = build_probe(model, tokenizer)
    inputs = prompt_inputs(tokenizer)
    with torch.no_grad():
        baseline = model(**inputs).logits
    with steering.AdaptiveActivationSteerer(
        model,
        layer_idx=1,
        vector=probe_vector,
        probe=probe,
        alpha=1.5,
        beta=0.0,
    ):
        with torch.no_grad():
            steered = model(**inputs).logits
    assert not torch.allclose(baseline, steered)


def test_adaptive_steerer_zero_alpha_no_change(model, tokenizer):
    probe, probe_vector = build_probe(model, tokenizer)
    inputs = prompt_inputs(tokenizer)
    with torch.no_grad():
        baseline = model(**inputs).logits
    with steering.AdaptiveActivationSteerer(
        model,
        layer_idx=1,
        vector=probe_vector,
        probe=probe,
        alpha=0.0,
        beta=0.0,
    ):
        with torch.no_grad():
            steered = model(**inputs).logits
    assert torch.allclose(baseline, steered)


def test_generate_with_adaptive_steering_returns_string(model, tokenizer):
    probe, probe_vector = build_probe(model, tokenizer)
    result = steering.generate_with_adaptive_steering(
        TEST_PROMPTS[0],
        model=model,
        tokenizer=tokenizer,
        layer_idx=1,
        probe_vector=probe_vector,
        probe=probe,
        device="cpu",
        max_new_tokens=3,
    )
    assert isinstance(result, str)


def test_decaying_activation_steerer_preserves_step_zero_for_prefill():
    steerer = steering.DecayingActivationSteerer(
        model=object(),
        layer_idx=0,
        vector=torch.tensor([1.0, 0.0]),
        alpha=2.0,
        decay=0.5,
        max_steps=2,
    )

    prefill_scale = steerer.get_scale(torch.zeros(1, 4, 2))
    first_generation_scale = steerer.get_scale(torch.zeros(1, 1, 2))
    second_generation_scale = steerer.get_scale(torch.zeros(1, 1, 2))
    cutoff_scale = steerer.get_scale(torch.zeros(1, 1, 2))

    assert prefill_scale == pytest.approx(2.0)
    assert first_generation_scale == pytest.approx(2.0)
    assert second_generation_scale == pytest.approx(1.0)
    assert cutoff_scale == pytest.approx(0.0)


def test_generate_with_decaying_steering_returns_string(model, tokenizer):
    result = steering.generate_with_decaying_steering(
        TEST_PROMPTS[0],
        model=model,
        tokenizer=tokenizer,
        layer_idx=1,
        steering_vector=build_vector(model, tokenizer),
        device="cpu",
        alpha=1.5,
        decay=0.5,
        max_steps=2,
        max_new_tokens=3,
    )

    assert isinstance(result, str)


def test_collect_evaluation_rows_returns_expected_shape(model, tokenizer):
    probe, probe_vector = build_probe(model, tokenizer)
    rows = steering.collect_evaluation_rows(
        EVALUATION_PROMPTS[:2],
        model=model,
        tokenizer=tokenizer,
        layer_idx=1,
        steering_vector=build_vector(model, tokenizer),
        probe=probe,
        probe_vector=probe_vector,
        device="cpu",
        max_new_tokens=3,
    )
    assert len(rows) == 2
    assert set(rows[0]) == {"prompt", "baseline", "fixed", "adaptive"}


def test_load_standard_activation_catalog_returns_valid_structure():
    catalog = steering.load_standard_activation_catalog()
    assert steering.STANDARD_ACTIVATIONS_PATH.is_file()
    assert catalog["default_model"] == "gpt2"
    assert "gpt2" in catalog["models"]


def test_load_standard_activation_catalog_returns_deep_copy():
    catalog = steering.load_standard_activation_catalog()
    assert "gpt2" in catalog["models"]
    assert catalog["models"]["gpt2"]["activations"]
    catalog["models"]["gpt2"]["activations"][0]["name"] = "mutated"

    fresh_catalog = steering.load_standard_activation_catalog()

    assert fresh_catalog["models"]["gpt2"]["activations"][0]["name"] != "mutated"


def test_get_standard_activations_returns_default_model_entries():
    activations = steering.get_standard_activations()
    categories = {activation["category"] for activation in activations}
    activation_names = {activation["name"] for activation in activations}
    assert {
        "prompt_engineering",
        "context_engineering",
        "cognitive_architecture",
        "reasoning_strategy",
    }.issubset(categories)
    assert {
        "zero_shot_prompting",
        "retrieval_augmented_context",
        "react",
        "chain_of_thought",
    }.issubset(activation_names)


def test_get_standard_activations_returns_copied_entries():
    activations = steering.get_standard_activations()
    assert activations
    activations[0]["name"] = "mutated"

    fresh_activations = steering.get_standard_activations()

    assert fresh_activations[0]["name"] != "mutated"


def test_get_standard_activations_filters_by_category():
    activations = steering.get_standard_activations(category="prompt_engineering")
    assert activations
    assert {activation["category"] for activation in activations} == {"prompt_engineering"}


def test_get_standard_activations_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model_name"):
        steering.get_standard_activations(model_name="unknown-model")


def test_discover_artifact_plugin_paths_includes_builtin_standard_plugin():
    plugin_paths = steering.discover_artifact_plugin_paths()

    assert any(str(path).endswith("activation_steering/artifacts/models/gpt2/standard") for path in plugin_paths)
    assert steering.STANDARD_ARTIFACT_PLUGIN_PATH.joinpath("plugin.json").is_file()


def test_load_model_artifact_bundle_merges_plugin_entries(tmp_path):
    plugin_root = tmp_path / "artifact-bundles"
    steering.write_artifact_plugin(
        plugin_root,
        model_name="gpt2",
        plugin_name="base",
        description="Base bundle",
        activations=[
            {
                "name": "retrieval_augmented_context",
                "category": "context_engineering",
                "summary": "Base summary.",
            },
            {
                "name": "scratchpad_context",
                "category": "context_engineering",
                "summary": "Scratchpad summary.",
            },
        ],
        feature_specs=[
            {
                "name": "chain_of_thought",
                "model_name": "gpt2",
                "category": "reasoning_strategy",
                "summary": "Base reasoning summary.",
                "extraction_examples": [{"text": "base positive", "label": "positive"}],
                "test_cases": [{"text": "base test", "label": "requires_reasoning"}],
                "evaluation_criteria": [{"name": "base", "description": "base"}],
            }
        ],
        controllers=[
            {
                "name": "chain_of_thought",
                "model_name": "gpt2",
                "category": "reasoning_strategy",
                "summary": "Base controller summary.",
                "layer_idx": 1,
                "vector": [1.0, 0.0],
            }
        ],
    )
    steering.write_artifact_plugin(
        plugin_root,
        model_name="gpt2",
        plugin_name="zz_override",
        activations=[
            {
                "name": "retrieval_augmented_context",
                "category": "context_engineering",
                "summary": "Override summary.",
            }
        ],
        feature_specs=[
            {
                "name": "chain_of_thought",
                "model_name": "gpt2",
                "category": "reasoning_strategy",
                "summary": "Override reasoning summary.",
                "extraction_examples": [{"text": "override positive", "label": "positive"}],
                "test_cases": [{"text": "override test", "label": "requires_reasoning"}],
                "evaluation_criteria": [{"name": "override", "description": "override"}],
            }
        ],
        controllers=[
            {
                "controller_id": "chain_of_thought",
                "feature_name": "chain_of_thought",
                "model_name": "gpt2",
                "category": "reasoning_strategy",
                "summary": "Override controller summary.",
                "layer_idx": 1,
                "vector": [0.0, 1.0],
            }
        ],
    )

    bundle = steering.load_model_artifact_bundle(model_name="gpt2", plugin_roots=plugin_root)

    assert bundle["model_name"] == "gpt2"
    assert {activation["name"] for activation in bundle["activations"]} == {
        "retrieval_augmented_context",
        "scratchpad_context",
    }
    retrieval_activation = next(
        activation for activation in bundle["activations"] if activation["name"] == "retrieval_augmented_context"
    )
    assert retrieval_activation["summary"] == "Override summary."
    feature = next(feature for feature in bundle["feature_specs"] if feature["name"] == "chain_of_thought")
    assert feature["summary"] == "Override reasoning summary."
    controller = next(
        controller for controller in bundle["controllers"] if controller["controller_id"] == "chain_of_thought"
    )
    assert controller["summary"] == "Override controller summary."


def test_merge_artifact_plugins_writes_merged_bundle(tmp_path):
    source_root = tmp_path / "source"
    steering.write_artifact_plugin(
        source_root,
        model_name="gpt2",
        plugin_name="controllers_only",
        controllers=[
            {
                "name": "chain_of_thought",
                "model_name": "gpt2",
                "category": "reasoning_strategy",
                "summary": "Merged controller summary.",
                "layer_idx": 1,
                "vector": [1.0, 0.0],
            }
        ],
    )

    merged_plugin_path = steering.merge_artifact_plugins(
        tmp_path / "merged",
        model_name="gpt2",
        plugin_roots=source_root,
        merged_plugin_name="team_pack",
        description="Team merged bundle",
    )

    assert merged_plugin_path.joinpath("plugin.json").is_file()
    merged_bundle = steering.load_model_artifact_bundle(
        model_name="gpt2",
        plugin_roots=tmp_path / "merged",
    )
    assert merged_bundle["controllers"][0]["controller_id"] == "chain_of_thought"
    assert merged_bundle["description"] == "Team merged bundle"


def test_feature_example_round_trips_to_dict():
    example = steering.FeatureExample(
        text="Question: 2 + 2?\nAnswer: 4",
        label="positive",
        metadata={"source": "unit-test"},
    )
    restored = steering.FeatureExample.from_dict(example.to_dict())
    assert restored.text == example.text
    assert restored.label == example.label
    assert restored.metadata == {"source": "unit-test"}


def test_build_feature_spec_validates_and_filters_texts():
    spec = steering.build_feature_spec(
        name="math_reasoning",
        model_name="gpt2",
        category="reasoning_strategy",
        summary="Capture multi-step arithmetic reasoning.",
        extraction_examples=[
            {"text": "2 + 2 = 4 with steps", "label": "positive"},
            {"text": "2 + 2 = 4", "label": "negative"},
        ],
        test_cases=[
            {"text": "What is 7 + 5?", "label": "requires_reasoning"},
        ],
        evaluation_criteria=[
            {
                "name": "contains_steps",
                "description": "The response includes intermediate arithmetic steps.",
            }
        ],
        metadata={"owner": "tests"},
    )
    assert spec.get_extraction_texts(label="positive") == ["2 + 2 = 4 with steps"]
    assert spec.get_extraction_texts(label="negative") == ["2 + 2 = 4"]
    assert spec.get_test_case_texts() == ["What is 7 + 5?"]
    assert spec.metadata == {"owner": "tests"}


def test_feature_spec_get_test_texts_alias_matches_get_test_case_texts():
    spec = steering.build_feature_spec(
        name="alias_check",
        model_name="gpt2",
        category="reasoning_strategy",
        summary="Verify backward-compatible alias behavior.",
        extraction_examples=[
            {"text": "Worked answer", "label": "positive"},
        ],
        test_cases=[
            {"text": "New question", "label": "requires_reasoning"},
        ],
        evaluation_criteria=[
            {"name": "criterion", "description": "Check alias behavior."},
        ],
    )
    assert spec.get_test_texts() == spec.get_test_case_texts()


def test_feature_spec_requires_test_cases():
    with pytest.raises(ValueError, match="test_cases"):
        steering.FeatureSpec(
            name="missing_tests",
            model_name="gpt2",
            category="reasoning_strategy",
            summary="Invalid feature spec.",
            extraction_examples=[{"text": "x", "label": "positive"}],
            test_cases=[],
            evaluation_criteria=[
                {"name": "criterion", "description": "Something to check."},
            ],
        )


def test_get_standard_feature_catalog_returns_typed_catalog():
    catalog = steering.get_standard_feature_catalog()
    assert steering.STANDARD_FEATURE_SPECS_PATH.is_file()
    assert catalog.model_name == "gpt2"
    assert "reasoning_strategy" in catalog.list_categories()
    feature = catalog.get_feature("chain_of_thought")
    assert feature.model_name == "gpt2"
    assert feature.get_extraction_texts(label="positive")
    assert feature.evaluation_criteria[0].name


def test_get_standard_feature_specs_filters_by_category():
    feature_specs = steering.get_standard_feature_specs(category="context_engineering")
    assert feature_specs
    assert {feature.category for feature in feature_specs} == {"context_engineering"}


def test_get_standard_feature_catalog_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model_name"):
        steering.get_standard_feature_catalog(model_name="unknown-model")


def test_feature_catalog_rejects_duplicate_feature_names():
    with pytest.raises(ValueError, match="duplicate feature names"):
        steering.FeatureCatalog(
            model_name="gpt2",
            features=[
                steering.build_feature_spec(
                    name="duplicate",
                    model_name="gpt2",
                    category="reasoning_strategy",
                    summary="First feature.",
                    extraction_examples=[{"text": "a", "label": "positive"}],
                    test_cases=[{"text": "b", "label": "test"}],
                    evaluation_criteria=[
                        {"name": "criterion_one", "description": "desc"},
                    ],
                ),
                steering.build_feature_spec(
                    name="duplicate",
                    model_name="gpt2",
                    category="reasoning_strategy",
                    summary="Second feature.",
                    extraction_examples=[{"text": "c", "label": "positive"}],
                    test_cases=[{"text": "d", "label": "test"}],
                    evaluation_criteria=[
                        {"name": "criterion_two", "description": "desc"},
                    ],
                ),
            ],
        )


def test_feature_catalog_rejects_mismatched_feature_model_names():
    with pytest.raises(ValueError, match="must have the same model_name"):
        steering.FeatureCatalog(
            model_name="gpt2",
            features=[
                steering.build_feature_spec(
                    name="mismatch",
                    model_name="llama",
                    category="reasoning_strategy",
                    summary="Wrong model.",
                    extraction_examples=[{"text": "a", "label": "positive"}],
                    test_cases=[{"text": "b", "label": "test"}],
                    evaluation_criteria=[
                        {"name": "criterion", "description": "desc"},
                    ],
                )
            ],
        )


def test_get_standard_feature_models_uses_raw_payload(monkeypatch):
    def raise_if_feature_catalogs_loaded():
        raise AssertionError("load_standard_feature_catalogs should not be called")

    monkeypatch.setattr(
        steering_features,
        "load_standard_feature_catalogs",
        raise_if_feature_catalogs_loaded,
    )

    assert steering.get_standard_feature_models() == ["gpt2"]


def test_discover_and_store_feature_vectors_runs_minimal_flow(tokenizer, tmp_path):
    model = make_gpt2_model(tokenizer)
    feature_specs = steering.get_standard_feature_specs()
    output_path = tmp_path / "identified_feature_vectors.json"
    fixture_path = (
        Path(__file__).parent / "data" / "minimal_identified_feature_vectors.json"
    )

    discovered_vectors = steering.discover_and_store_feature_vectors(
        feature_specs=feature_specs,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        output_path=output_path,
    )

    assert output_path.is_file()
    assert fixture_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    fixture_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == fixture_payload["format_version"]
    assert payload["feature_vector_count"] == len(feature_specs)
    assert payload["feature_vector_count"] == fixture_payload["feature_vector_count"]
    assert len(discovered_vectors) == len(feature_specs)
    assert {item["name"] for item in payload["feature_vectors"]} == {
        feature_spec.name for feature_spec in feature_specs
    }
    for actual_item, expected_item in zip(
        payload["feature_vectors"], fixture_payload["feature_vectors"], strict=True
    ):
        assert {key: value for key, value in actual_item.items() if key not in {"vector", "vector_norm"}} == {
            key: value for key, value in expected_item.items() if key not in {"vector", "vector_norm"}
        }
        assert actual_item["vector"] == pytest.approx(expected_item["vector"], abs=1e-6)
        assert actual_item["vector_norm"] == pytest.approx(expected_item["vector_norm"], abs=1e-6)
    assert all(item["vector_size"] == model.config.n_embd for item in payload["feature_vectors"])
    assert all(item["vector_norm"] > 0.0 for item in payload["feature_vectors"])
    assert all(vector.vector.shape == (model.config.n_embd,) for vector in discovered_vectors)
