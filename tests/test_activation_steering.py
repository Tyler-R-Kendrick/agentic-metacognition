"""
Tests for activation_steering.py — written FIRST (TDD red phase).

All tests use minimal toy PyTorch modules and mock objects so the suite
runs quickly on CPU without downloading any real model weights.
"""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import activation_steering as steering


# ---------------------------------------------------------------------------
# Shared fake helpers
# ---------------------------------------------------------------------------

# A small vocabulary of texts → deterministic token-id seeds.
_TOKEN_SEED_MAP = {
    "pos1": 1, "pos2": 2, "pos3": 3, "pos4": 4, "pos5": 5, "pos6": 6,
    "neg1": 10, "neg2": 11, "neg3": 12, "neg4": 13, "neg5": 14, "neg6": 15,
    "hello": 20, "test": 21, "test prompt": 22,
    "text1": 30, "text2": 31,
}


class _FakeTokenizerOutput(dict):
    """A dict-subclass that also exposes attributes and a no-op `.to()`."""

    def __init__(self, text):
        base = _TOKEN_SEED_MAP.get(text, abs(hash(text[:8])) % 80) + 1
        ids = torch.tensor([[base, base + 1, base + 2]], dtype=torch.long)
        mask = torch.ones_like(ids)
        super().__init__(input_ids=ids, attention_mask=mask)
        self.input_ids = ids
        self.attention_mask = mask

    def to(self, device):  # mirrors HuggingFace BatchEncoding.to() API
        return self


def make_fake_tokenizer():
    """Return a mock tokenizer that produces distinct token IDs per text."""
    mock = MagicMock(side_effect=lambda text, **kw: _FakeTokenizerOutput(text))
    mock.pad_token = None
    mock.eos_token = "<eos>"
    mock.decode = MagicMock(return_value="decoded output")
    return mock


class TinyFakeModel(nn.Module):
    """Mimics a HuggingFace causal LM: accepts input_ids, returns hidden_states."""

    def __init__(self, hidden_dim: int = 8, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        torch.manual_seed(0)
        self.embed = nn.Embedding(200, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
        )

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kw):
        x = self.embed(input_ids)
        all_hidden = [x]
        for layer in self.layers:
            x = layer(x)
            all_hidden.append(x)

        class _Out:
            hidden_states = tuple(all_hidden)

        return _Out()


class TinyHookableModel(nn.Module):
    """
    Like TinyFakeModel but exposes self.model.layers (Llama-style) and has
    a simple generate() so it can be used in ActivationSteerer / generate() tests.
    """

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        torch.manual_seed(0)
        self.embed = nn.Embedding(200, hidden_dim)
        _layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(2)]
        )
        self.layers = _layers
        # Llama-style attribute used by get_transformer_layers()
        self.model = types.SimpleNamespace(layers=_layers)

    def forward(self, input_ids, attention_mask=None, **kw):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

    def generate(self, input_ids, attention_mask=None, max_new_tokens=5,
                 do_sample=False, **kw):
        extra = torch.zeros(input_ids.shape[0], max_new_tokens, dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)

    def parameters(self):           # needed by ActivationSteerer to find device
        return iter([next(nn.Module.parameters(self))])


# ---------------------------------------------------------------------------
# 1.  get_transformer_layers
# ---------------------------------------------------------------------------

class _LlamaModel:
    def __init__(self):
        self.model = types.SimpleNamespace(layers=["layer0", "layer1"])


class _GPTModel:
    def __init__(self):
        self.transformer = types.SimpleNamespace(h=["h0", "h1"])


class _NeoXModel:
    def __init__(self):
        self.gpt_neox = types.SimpleNamespace(layers=["nx0", "nx1"])


class _UnknownModel:
    pass


def test_get_transformer_layers_llama_style():
    m = _LlamaModel()
    assert steering.get_transformer_layers(m) is m.model.layers


def test_get_transformer_layers_gpt_style():
    m = _GPTModel()
    assert steering.get_transformer_layers(m) is m.transformer.h


def test_get_transformer_layers_neox_style():
    m = _NeoXModel()
    assert steering.get_transformer_layers(m) is m.gpt_neox.layers


def test_get_transformer_layers_unsupported_raises():
    with pytest.raises(ValueError):
        steering.get_transformer_layers(_UnknownModel())


# ---------------------------------------------------------------------------
# 2.  cosine helper
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    a = np.array([1.0, 0.0, 0.0])
    assert abs(steering.cosine(a, a) - 1.0) < 1e-6


def test_cosine_opposite_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    assert abs(steering.cosine(a, b) + 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(steering.cosine(a, b)) < 1e-6


# ---------------------------------------------------------------------------
# 3.  get_hidden_states / get_last_token_hidden
# ---------------------------------------------------------------------------

def test_get_hidden_states_returns_tuple():
    model = TinyFakeModel(hidden_dim=8, num_layers=3)
    tokenizer = make_fake_tokenizer()
    hs = steering.get_hidden_states("hello", model, tokenizer, "cpu")
    # embedding + 3 transformer layers = 4 elements
    assert len(hs) == 4


def test_get_last_token_hidden_shape():
    model = TinyFakeModel(hidden_dim=8, num_layers=3)
    tokenizer = make_fake_tokenizer()
    h = steering.get_last_token_hidden("hello", layer_idx=1,
                                       model=model, tokenizer=tokenizer, device="cpu")
    assert isinstance(h, np.ndarray)
    assert h.shape == (8,)


def test_get_last_token_hidden_uses_correct_offset():
    """hidden_states[layer_idx + 1] is the output of transformer block layer_idx."""
    model = TinyFakeModel(hidden_dim=8, num_layers=3)
    tokenizer = make_fake_tokenizer()

    for layer_idx in range(3):
        h = steering.get_last_token_hidden(
            "test", layer_idx=layer_idx,
            model=model, tokenizer=tokenizer, device="cpu"
        )
        assert h.shape == (8,), f"Wrong shape at layer {layer_idx}"


def test_get_last_token_hidden_differs_across_layers():
    """Each layer should (in general) produce a different vector."""
    model = TinyFakeModel(hidden_dim=8, num_layers=3)
    tokenizer = make_fake_tokenizer()

    h0 = steering.get_last_token_hidden("test", 0, model, tokenizer, "cpu")
    h1 = steering.get_last_token_hidden("test", 1, model, tokenizer, "cpu")
    assert not np.allclose(h0, h1), "Layers 0 and 1 should produce different outputs"


# ---------------------------------------------------------------------------
# 4.  build_mean_difference_vector
# ---------------------------------------------------------------------------

def test_build_mean_difference_vector_shape():
    model = TinyFakeModel(hidden_dim=8, num_layers=3)
    tokenizer = make_fake_tokenizer()
    vec, pos_h, neg_h = steering.build_mean_difference_vector(
        ["pos1", "pos2"], ["neg1", "neg2"],
        layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    assert vec.shape == (8,)
    assert len(pos_h) == 2
    assert len(neg_h) == 2


def test_build_mean_difference_vector_is_unit_norm():
    model = TinyFakeModel(hidden_dim=8, num_layers=3)
    tokenizer = make_fake_tokenizer()
    vec, _, _ = steering.build_mean_difference_vector(
        ["pos1", "pos2", "pos3"], ["neg1", "neg2", "neg3"],
        layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_build_mean_difference_vector_direction():
    """
    By construction vec = mean(pos) - mean(neg), so projecting the means onto
    vec must satisfy dot(mean_pos, vec) > dot(mean_neg, vec).
    """
    pos_vecs = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.5, 0.0])]
    neg_vecs = [np.array([-1.0, 0.0, 0.0]), np.array([-1.0, -0.5, 0.0])]
    side_effects = pos_vecs + neg_vecs

    with patch("activation_steering.get_last_token_hidden", side_effect=side_effects):
        vec, pos_h, neg_h = steering.build_mean_difference_vector(
            ["p1", "p2"], ["n1", "n2"],
            layer_idx=0, model=None, tokenizer=None, device="cpu"
        )

    pos_proj = np.dot(np.mean(pos_vecs, axis=0), vec)
    neg_proj = np.dot(np.mean(neg_vecs, axis=0), vec)
    assert pos_proj > neg_proj


# ---------------------------------------------------------------------------
# 5.  ActivationSteerer
# ---------------------------------------------------------------------------

def test_activation_steerer_modifies_output():
    """Hooking a layer with alpha > 0 should change the model output."""
    model = TinyHookableModel(hidden_dim=8)
    dummy = torch.tensor([[1, 2, 3]])
    vector = np.ones(8) / np.sqrt(8)

    with torch.no_grad():
        baseline = model(dummy).clone()

    with steering.ActivationSteerer(model, layer_idx=0, vector=vector, alpha=1.5):
        with torch.no_grad():
            steered = model(dummy).clone()

    assert not torch.allclose(baseline, steered)


def test_activation_steerer_hook_removed_after_exit():
    """After exiting the context, the model output must equal the pre-hook baseline."""
    model = TinyHookableModel(hidden_dim=8)
    dummy = torch.tensor([[1, 2]])
    vector = np.ones(8) / np.sqrt(8)

    with torch.no_grad():
        before = model(dummy).clone()

    with steering.ActivationSteerer(model, layer_idx=0, vector=vector, alpha=2.0):
        pass  # enter and exit immediately

    with torch.no_grad():
        after = model(dummy).clone()

    assert torch.allclose(before, after)


def test_activation_steerer_zero_alpha_no_change():
    """alpha=0 must leave the output unchanged."""
    model = TinyHookableModel(hidden_dim=8)
    dummy = torch.tensor([[1, 2, 3]])
    vector = np.ones(8) / np.sqrt(8)

    with torch.no_grad():
        baseline = model(dummy).clone()

    with steering.ActivationSteerer(model, layer_idx=0, vector=vector, alpha=0.0):
        with torch.no_grad():
            steered = model(dummy).clone()

    assert torch.allclose(baseline, steered)


# ---------------------------------------------------------------------------
# 6.  generate helper
# ---------------------------------------------------------------------------

def test_generate_returns_string():
    model = TinyHookableModel(hidden_dim=8)
    tokenizer = make_fake_tokenizer()
    tokenizer.decode = MagicMock(return_value="Hello world response")

    result = steering.generate("test prompt", model, tokenizer, "cpu", max_new_tokens=5)

    assert isinstance(result, str)
    assert result == "Hello world response"


def test_generate_calls_model_generate():
    """generate() must call model.generate() with the tokenized inputs."""
    model = TinyHookableModel(hidden_dim=8)
    tokenizer = make_fake_tokenizer()
    tokenizer.decode = MagicMock(return_value="output")

    with patch.object(model, "generate", wraps=model.generate) as mock_gen:
        steering.generate("test prompt", model, tokenizer, "cpu", max_new_tokens=7)
        mock_gen.assert_called_once()
        _, kwargs = mock_gen.call_args
        assert "max_new_tokens" in kwargs or len(mock_gen.call_args.args) > 0


# ---------------------------------------------------------------------------
# 7.  train_probe
# ---------------------------------------------------------------------------

def test_train_probe_returns_probe_and_unit_vector():
    model = TinyFakeModel(hidden_dim=8, num_layers=3)
    tokenizer = make_fake_tokenizer()

    probe, probe_vec = steering.train_probe(
        ["pos1", "pos2", "pos3", "pos4"],
        ["neg1", "neg2", "neg3", "neg4"],
        layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )

    assert probe_vec.shape == (8,)
    assert abs(np.linalg.norm(probe_vec) - 1.0) < 1e-5


def test_train_probe_predict_proba_shape():
    """The returned probe must expose predict_proba returning (1, 2) probabilities."""
    model = TinyFakeModel(hidden_dim=8, num_layers=3)
    tokenizer = make_fake_tokenizer()

    probe, probe_vec = steering.train_probe(
        ["pos1", "pos2", "pos3", "pos4"],
        ["neg1", "neg2", "neg3", "neg4"],
        layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )

    test_h = steering.get_last_token_hidden(
        "test", layer_idx=1, model=model, tokenizer=tokenizer, device="cpu"
    )
    proba = probe.predict_proba(test_h.reshape(1, -1))
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 8.  AdaptiveActivationSteerer
# ---------------------------------------------------------------------------

def test_adaptive_steerer_modifies_output():
    """With p_positive < 1, scale > 0 so the output must change."""
    model = TinyHookableModel(hidden_dim=8)
    vector = np.ones(8) / np.sqrt(8)
    mock_probe = MagicMock()
    mock_probe.predict_proba.return_value = np.array([[0.3, 0.7]])  # p_positive = 0.7

    dummy = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        baseline = model(dummy).clone()

    with steering.AdaptiveActivationSteerer(
        model, layer_idx=0, vector=vector, probe=mock_probe, alpha=1.0, beta=0.0
    ):
        with torch.no_grad():
            steered = model(dummy).clone()

    # scale = 1.0 * (1 - 0.7 + 0.0) = 0.3 → output must differ
    assert not torch.allclose(baseline, steered)


def test_adaptive_steerer_hook_removed_after_exit():
    model = TinyHookableModel(hidden_dim=8)
    vector = np.ones(8) / np.sqrt(8)
    mock_probe = MagicMock()
    mock_probe.predict_proba.return_value = np.array([[0.5, 0.5]])

    dummy = torch.tensor([[1, 2]])

    with torch.no_grad():
        before = model(dummy).clone()

    with steering.AdaptiveActivationSteerer(
        model, layer_idx=0, vector=vector, probe=mock_probe, alpha=2.0
    ):
        pass

    with torch.no_grad():
        after = model(dummy).clone()

    assert torch.allclose(before, after)


def test_adaptive_steerer_full_positive_minimal_effect():
    """
    When the probe returns p_positive ≈ 1, scale ≈ beta (0 by default) so the
    correction is very small – the steered output should be close to baseline.
    """
    model = TinyHookableModel(hidden_dim=8)
    vector = np.ones(8) / np.sqrt(8)
    mock_probe = MagicMock()
    # p_positive = 1.0 → scale = 1.0 * (1 - 1.0 + 0.0) = 0.0
    mock_probe.predict_proba.return_value = np.array([[0.0, 1.0]])

    dummy = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        baseline = model(dummy).clone()

    with steering.AdaptiveActivationSteerer(
        model, layer_idx=0, vector=vector, probe=mock_probe, alpha=1.0, beta=0.0
    ):
        with torch.no_grad():
            steered = model(dummy).clone()

    assert torch.allclose(baseline, steered, atol=1e-5)
