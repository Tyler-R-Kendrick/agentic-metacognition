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


def make_test_tokenizer():
    vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
    all_texts = (
        steering.POSITIVE_TEXTS
        + steering.NEGATIVE_TEXTS
        + steering.TEST_PROMPTS
        + steering.EVALUATION_PROMPTS
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
        steering.POSITIVE_TEXTS,
        steering.NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )[0]


def build_probe(model, tokenizer):
    return steering.train_probe(
        steering.POSITIVE_TEXTS,
        steering.NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )


def prompt_inputs(tokenizer):
    return steering.tokenize_text(steering.TEST_PROMPTS[0], tokenizer, "cpu")


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
        steering.POSITIVE_TEXTS[0], model=model, tokenizer=tokenizer, device="cpu"
    )
    assert isinstance(hidden_states, tuple)
    assert len(hidden_states) == model.config.n_layer + 1


def test_get_last_token_hidden_shape(model, tokenizer):
    hidden = steering.get_last_token_hidden(
        steering.POSITIVE_TEXTS[0], layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == (model.config.n_embd,)


def test_get_last_token_hidden_differs_across_layers(model, tokenizer):
    first = steering.get_last_token_hidden(
        steering.POSITIVE_TEXTS[0], layer_idx=0, model=model, tokenizer=tokenizer, device="cpu"
    )
    second = steering.get_last_token_hidden(
        steering.POSITIVE_TEXTS[0], layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    assert not torch.allclose(first, second)


def test_collect_last_token_hiddens_shape(model, tokenizer):
    hidden = steering.collect_last_token_hiddens(
        steering.POSITIVE_TEXTS, layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    assert hidden.shape == (len(steering.POSITIVE_TEXTS), model.config.n_embd)


def test_build_mean_difference_vector_shape_and_norm(model, tokenizer):
    vec, pos_hidden, neg_hidden = steering.build_mean_difference_vector(
        steering.POSITIVE_TEXTS,
        steering.NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    assert vec.shape == (model.config.n_embd,)
    assert pos_hidden.shape == (len(steering.POSITIVE_TEXTS), model.config.n_embd)
    assert neg_hidden.shape == (len(steering.NEGATIVE_TEXTS), model.config.n_embd)
    assert torch.isclose(vec.norm(), torch.tensor(1.0), atol=1e-5)


def test_build_mean_difference_vector_direction(model, tokenizer):
    vec, pos_hidden, neg_hidden = steering.build_mean_difference_vector(
        steering.POSITIVE_TEXTS,
        steering.NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    assert torch.dot(pos_hidden.mean(dim=0), vec) > torch.dot(neg_hidden.mean(dim=0), vec)


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
        steering.TEST_PROMPTS[0], model=model, tokenizer=tokenizer, device="cpu", max_new_tokens=4
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_is_deterministic_when_sampling_disabled(model, tokenizer):
    first = steering.generate(
        steering.TEST_PROMPTS[0], model=model, tokenizer=tokenizer, device="cpu", max_new_tokens=4
    )
    second = steering.generate(
        steering.TEST_PROMPTS[0], model=model, tokenizer=tokenizer, device="cpu", max_new_tokens=4
    )
    assert first == second


def test_generate_with_steering_returns_string(model, tokenizer):
    result = steering.generate_with_steering(
        steering.TEST_PROMPTS[0],
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
        steering.TEST_PROMPTS[0], 1, model=model, tokenizer=tokenizer, device="cpu"
    )
    proba = probe.predict_proba(sample.reshape(1, -1).numpy())
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


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
        steering.TEST_PROMPTS[0],
        model=model,
        tokenizer=tokenizer,
        layer_idx=1,
        probe_vector=probe_vector,
        probe=probe,
        device="cpu",
        max_new_tokens=3,
    )
    assert isinstance(result, str)


def test_collect_evaluation_rows_returns_expected_shape(model, tokenizer):
    probe, probe_vector = build_probe(model, tokenizer)
    rows = steering.collect_evaluation_rows(
        steering.EVALUATION_PROMPTS[:2],
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
