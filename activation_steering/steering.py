from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .models import (
    DEFAULT_MAX_NEW_TOKENS,
    collect_last_token_hiddens,
    generate,
    get_transformer_layers,
)


def _validate_example_sets(positive_examples: list[str], negative_examples: list[str]) -> None:
    if not positive_examples:
        raise ValueError("positive_examples must contain at least one example.")
    if not negative_examples:
        raise ValueError("negative_examples must contain at least one example.")


def _get_cached_tensor(
    tensor: torch.Tensor,
    cache: dict[tuple[torch.device, torch.dtype], torch.Tensor],
    hidden: torch.Tensor,
) -> torch.Tensor:
    key = (hidden.device, hidden.dtype)
    if key not in cache:
        cache[key] = tensor.to(hidden.device, dtype=hidden.dtype)
    return cache[key]


def build_mean_difference_vector(
    positive_examples: list[str],
    negative_examples: list[str],
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
):
    """Build a normalized mean-difference steering vector."""
    _validate_example_sets(positive_examples, negative_examples)
    pos = collect_last_token_hiddens(positive_examples, layer_idx, model, tokenizer, device)
    neg = collect_last_token_hiddens(negative_examples, layer_idx, model, tokenizer, device)
    vec = pos.mean(dim=0) - neg.mean(dim=0)
    vec = vec / (vec.norm() + 1e-8)
    return vec, pos, neg


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity for 1D tensors."""
    a = a.float()
    b = b.float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


class ActivationSteerer:
    """Add alpha * vector to a chosen transformer block during forward passes."""

    def __init__(self, model, layer_idx: int, vector: torch.Tensor, alpha: float):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = torch.as_tensor(vector, dtype=torch.float32).detach().cpu()
        self._steering_tensor_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.alpha = alpha
        self.hook_handle = None

    @staticmethod
    def _split_output(output):
        if isinstance(output, tuple):
            return output[0], output[1:]
        return output, None

    @staticmethod
    def _merge_output(hidden, rest):
        if rest is None:
            return hidden
        return (hidden, *rest)

    def get_scale(self, hidden: torch.Tensor) -> float:
        return self.alpha

    def get_vector(self, hidden: torch.Tensor) -> torch.Tensor:
        return _get_cached_tensor(self.vector, self._steering_tensor_cache, hidden)

    def apply_steering(self, hidden: torch.Tensor) -> torch.Tensor:
        scale = self.get_scale(hidden)
        correction = (scale * self.get_vector(hidden)).view(1, 1, -1)
        return hidden + correction

    def _hook_fn(self, module, inputs, output):
        hidden, rest = self._split_output(output)
        return self._merge_output(self.apply_steering(hidden), rest)

    def __enter__(self):
        block = get_transformer_layers(self.model)[self.layer_idx]
        self.hook_handle = block.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def generate_with_steering(
    prompt: str,
    model,
    tokenizer,
    layer_idx: int,
    steering_vector: torch.Tensor,
    device: str | torch.device,
    alpha: float = 1.5,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate with a fixed steering vector applied at one layer."""
    with ActivationSteerer(model, layer_idx, steering_vector, alpha=alpha):
        return generate(prompt, model, tokenizer, device, max_new_tokens=max_new_tokens)


def train_probe(
    positive_examples: list[str],
    negative_examples: list[str],
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
    random_state: int = 42,
):
    """Train a tiny logistic-regression probe and return its normalized vector."""
    _validate_example_sets(positive_examples, negative_examples)
    pos = collect_last_token_hiddens(positive_examples, layer_idx, model, tokenizer, device)
    neg = collect_last_token_hiddens(negative_examples, layer_idx, model, tokenizer, device)
    X = torch.cat([pos, neg], dim=0).numpy()
    y = np.array([1] * len(pos) + [0] * len(neg))

    probe = LogisticRegression(max_iter=5000, random_state=random_state)
    probe.fit(X, y)

    coef = torch.tensor(probe.coef_[0], dtype=torch.float32)
    coef = coef / (coef.norm() + 1e-8)
    return probe, coef


class AdaptiveActivationSteerer:
    """
    Minimal ACT-style approximation using a probe score.

    If the current state already looks positive-like, steer less.
    If it looks less positive-like, steer more.
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        vector: torch.Tensor,
        probe,
        alpha: float,
        beta: float = 0.0,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = torch.as_tensor(vector, dtype=torch.float32).detach().cpu()
        self.probe = probe
        self.probe_weight = torch.as_tensor(probe.coef_[0], dtype=torch.float32).detach()
        self.probe_bias = torch.as_tensor(float(probe.intercept_[0]), dtype=torch.float32).detach()
        self._steering_tensor_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._probe_weight_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._probe_bias_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.alpha = alpha
        self.beta = beta
        self.hook_handle = None

    def _get_scale(self, hidden: torch.Tensor) -> float:
        last_hidden = hidden[0, -1, :].detach()
        probe_weight = _get_cached_tensor(self.probe_weight, self._probe_weight_cache, last_hidden)
        probe_bias = _get_cached_tensor(self.probe_bias, self._probe_bias_cache, last_hidden)
        logit = torch.dot(last_hidden, probe_weight) + probe_bias
        p_positive = torch.sigmoid(logit)
        scale = self.alpha * (1.0 - float(p_positive) + self.beta)
        return scale

    def _hook_fn(self, module, inputs, output):
        hidden, rest = ActivationSteerer._split_output(output)
        scale = self._get_scale(hidden)
        vector = _get_cached_tensor(self.vector, self._steering_tensor_cache, hidden)
        correction = (scale * vector).view(1, 1, -1)
        return ActivationSteerer._merge_output(hidden + correction, rest)

    def __enter__(self):
        block = get_transformer_layers(self.model)[self.layer_idx]
        self.hook_handle = block.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def generate_with_adaptive_steering(
    prompt: str,
    model,
    tokenizer,
    layer_idx: int,
    probe_vector: torch.Tensor,
    probe,
    device: str | torch.device,
    alpha: float = 2.0,
    beta: float = 0.0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate with adaptive probe-scaled steering applied at one layer."""
    with AdaptiveActivationSteerer(model, layer_idx, probe_vector, probe, alpha=alpha, beta=beta):
        return generate(prompt, model, tokenizer, device, max_new_tokens=max_new_tokens)


class DecayingActivationSteerer(ActivationSteerer):
    """Apply a steering vector with a token-step decay schedule."""

    def __init__(
        self,
        model,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float,
        decay: float = 1.0,
        max_steps: int | None = None,
    ):
        super().__init__(model=model, layer_idx=layer_idx, vector=vector, alpha=alpha)
        self.decay = decay
        self.max_steps = max_steps
        self._step = 0

    def get_scale(self, hidden: torch.Tensor) -> float:
        if self.max_steps is not None and self._step >= self.max_steps:
            return 0.0
        scale = self.alpha * (self.decay**self._step)
        is_prefill_pass = hidden.dim() > 1 and hidden.shape[1] > 1
        if not is_prefill_pass:
            self._step += 1
        return scale


def generate_with_decaying_steering(
    prompt: str,
    model,
    tokenizer,
    layer_idx: int,
    steering_vector: torch.Tensor,
    device: str | torch.device,
    alpha: float = 1.5,
    decay: float = 0.9,
    max_steps: int | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate with a steering vector that decays across generation steps."""
    with DecayingActivationSteerer(
        model,
        layer_idx,
        steering_vector,
        alpha=alpha,
        decay=decay,
        max_steps=max_steps,
    ):
        return generate(prompt, model, tokenizer, device, max_new_tokens=max_new_tokens)
