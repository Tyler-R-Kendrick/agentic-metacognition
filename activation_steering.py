"""
Minimal activation steering for decoder-only Hugging Face language models.

This uses a tiny contrastive dataset to build a concept vector for cautious /
hedged vs. overconfident / assertive behavior, then injects that vector with a
forward hook during generation.

Warnings:
- The vector is not a universal concept detector.
- It reflects the examples used to build it.
- "Truthfulness" is harder to define than style-like concepts.
- Stronger steering can hurt fluency or cause odd outputs.
- This is not a faithful reproduction of ACT.

This project demonstrates the mechanics of concept-vector steering with a tiny
toy dataset. It does not prove the existence of a universal "truth vector," and
it should be treated as a minimal interpretability experiment rather than a
production method.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SEED = 42
set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LAYER_IDX = 12
MAX_NEW_TOKENS = 80
DO_SAMPLE = False

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


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    """Load a small decoder-only causal LM with CPU/GPU-friendly defaults."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
    }
    if DEVICE == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if DEVICE != "cuda":
        model.to(DEVICE)
    model.eval()
    return model, tokenizer


def get_transformer_layers(model):
    """Return the transformer block list for common decoder-only architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Unsupported model architecture for this minimal demo.")


get_transformer_blocks = get_transformer_layers


def get_model_device(model) -> torch.device:
    """Pick the input device, including models loaded with device_map='auto'."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for value in device_map.values():
            if value in {"disk", "meta"}:
                continue
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            return torch.device(value)
    return next(model.parameters()).device


def tokenize_text(text: str, tokenizer, device: str | torch.device):
    """Tokenize text on the requested device."""
    return tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    ).to(device)


def get_hidden_states(text: str, model, tokenizer, device: str | torch.device):
    """Run the model and return embedding + per-layer hidden states."""
    inputs = tokenize_text(text, tokenizer, device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states


def get_last_token_hidden(
    text: str,
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Return the final-token hidden state for the selected transformer block.

    hidden_states[0] is the embedding output, so block layer_idx lives at
    hidden_states[layer_idx + 1].
    """
    num_layers = len(get_transformer_layers(model))
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(
            f"Invalid layer_idx {layer_idx}; expected a transformer block index "
            f"in the range [0, {num_layers - 1}] for this model."
        )
    hidden_states = get_hidden_states(text, model, tokenizer, device)
    return hidden_states[layer_idx + 1][0, -1, :].detach().float().cpu()


def collect_last_token_hiddens(
    texts: list[str],
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
) -> torch.Tensor:
    """Collect final-token hidden states for a batch of input texts."""
    return torch.stack(
        [get_last_token_hidden(text, layer_idx, model, tokenizer, device) for text in texts]
    )


def build_mean_difference_vector(
    positive_examples: list[str],
    negative_examples: list[str],
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
):
    """Build a normalized mean-difference steering vector."""
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

    def apply_steering(self, hidden: torch.Tensor) -> torch.Tensor:
        scale = self.get_scale(hidden)
        correction = (scale * self.vector).view(1, 1, -1).to(hidden.device, dtype=hidden.dtype)
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


def generate(
    prompt: str,
    model,
    tokenizer,
    device: str | torch.device | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = DO_SAMPLE,
) -> str:
    """Generate text deterministically by default."""
    run_device = torch.device(device) if device is not None else get_model_device(model)
    inputs = tokenize_text(prompt, tokenizer, run_device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = 0.7
    output_ids = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def train_probe(
    positive_examples: list[str],
    negative_examples: list[str],
    layer_idx: int,
    model,
    tokenizer,
    device: str | torch.device,
):
    """Train a tiny logistic-regression probe and return its normalized vector."""
    pos = collect_last_token_hiddens(positive_examples, layer_idx, model, tokenizer, device)
    neg = collect_last_token_hiddens(negative_examples, layer_idx, model, tokenizer, device)
    X = torch.cat([pos, neg], dim=0).numpy()
    y = np.array([1] * len(pos) + [0] * len(neg))

    probe = LogisticRegression(max_iter=5000, random_state=SEED)
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
        self.alpha = alpha
        self.beta = beta
        self.hook_handle = None

    def _get_scale(self, hidden: torch.Tensor) -> float:
        last_hidden = hidden[0, -1, :].detach().float().cpu().numpy().reshape(1, -1)
        p_positive = float(self.probe.predict_proba(last_hidden)[0, 1])
        return self.alpha * (1.0 - p_positive + self.beta)

    def _hook_fn(self, module, inputs, output):
        hidden, rest = ActivationSteerer._split_output(output)
        scale = self._get_scale(hidden)
        correction = (scale * self.vector).view(1, 1, -1).to(hidden.device, dtype=hidden.dtype)
        return ActivationSteerer._merge_output(hidden + correction, rest)

    def __enter__(self):
        block = get_transformer_layers(self.model)[self.layer_idx]
        self.hook_handle = block.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def collect_evaluation_rows(
    prompts: list[str],
    model,
    tokenizer,
    layer_idx: int,
    steering_vector: torch.Tensor,
    probe,
    probe_vector: torch.Tensor,
    device: str | torch.device,
    fixed_alpha: float = 1.5,
    adaptive_alpha: float = 2.0,
    beta: float = 0.0,
    max_new_tokens: int = 30,
):
    """Run a tiny comparison harness for baseline, fixed, and adaptive steering."""
    rows = []
    for prompt in prompts:
        baseline = generate(prompt, model, tokenizer, device, max_new_tokens=max_new_tokens)
        fixed = generate_with_steering(
            prompt,
            model,
            tokenizer,
            layer_idx,
            steering_vector,
            device,
            alpha=fixed_alpha,
            max_new_tokens=max_new_tokens,
        )
        adaptive = generate_with_adaptive_steering(
            prompt,
            model,
            tokenizer,
            layer_idx,
            probe_vector,
            probe,
            device,
            alpha=adaptive_alpha,
            beta=beta,
            max_new_tokens=max_new_tokens,
        )
        rows.append(
            {
                "prompt": prompt,
                "baseline": baseline,
                "fixed": fixed,
                "adaptive": adaptive,
            }
        )
    return rows


def generate_with_steering(
    prompt: str,
    model,
    tokenizer,
    layer_idx: int,
    steering_vector: torch.Tensor,
    device: str | torch.device,
    alpha: float = 1.5,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate with a fixed steering vector applied at one layer."""
    with ActivationSteerer(model, layer_idx, steering_vector, alpha=alpha):
        return generate(prompt, model, tokenizer, device, max_new_tokens=max_new_tokens)


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
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate with adaptive probe-scaled steering applied at one layer."""
    with AdaptiveActivationSteerer(model, layer_idx, probe_vector, probe, alpha=alpha, beta=beta):
        return generate(prompt, model, tokenizer, device, max_new_tokens=max_new_tokens)


def run_demo():
    """Run the full minimal demo in one script."""
    print(f"DEVICE: {DEVICE}")
    print(f"Loading {DEFAULT_MODEL} ...")
    model, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL)
    blocks = get_transformer_layers(model)
    assert 0 <= DEFAULT_LAYER_IDX < len(blocks), "DEFAULT_LAYER_IDX out of range"
    device = get_model_device(model)

    steering_vector, pos_hidden, neg_hidden = build_mean_difference_vector(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        DEFAULT_LAYER_IDX,
        model,
        tokenizer,
        device,
    )

    print("Vector shape:", tuple(steering_vector.shape))
    print("Mean positive/negative cosine separation:")
    print("  avg cos(pos_i, vec):", np.mean([cosine(x, steering_vector) for x in pos_hidden]))
    print("  avg cos(neg_i, vec):", np.mean([cosine(x, steering_vector) for x in neg_hidden]))

    for prompt in TEST_PROMPTS:
        print("=" * 100)
        print("PROMPT")
        print(prompt)
        print("\nBASELINE")
        print(generate(prompt, model, tokenizer, device, max_new_tokens=40))
        print("\nSTEERED (alpha=1.5)")
        print(
            generate_with_steering(
                prompt,
                model,
                tokenizer,
                DEFAULT_LAYER_IDX,
                steering_vector,
                device,
                alpha=1.5,
                max_new_tokens=40,
            )
        )
        print()

    sweep_prompt = "Question: What is the capital of Germany?\nAnswer:"
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        print("\n" + "#" * 100)
        print(f"alpha = {alpha}")
        print(
            generate(sweep_prompt, model, tokenizer, device, max_new_tokens=40)
            if alpha == 0.0
            else generate_with_steering(
                sweep_prompt,
                model,
                tokenizer,
                DEFAULT_LAYER_IDX,
                steering_vector,
                device,
                alpha=alpha,
                max_new_tokens=40,
            )
        )

    probe, probe_vector = train_probe(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        DEFAULT_LAYER_IDX,
        model,
        tokenizer,
        device,
    )
    print("Probe trained.")

    compare_prompt = "Question: What is the capital of Italy?\nAnswer:"
    print("BASELINE")
    print(generate(compare_prompt, model, tokenizer, device, max_new_tokens=40))
    print("\nFIXED STEERING (mean-diff vector)")
    print(
        generate_with_steering(
            compare_prompt,
            model,
            tokenizer,
            DEFAULT_LAYER_IDX,
            steering_vector,
            device,
            alpha=1.5,
            max_new_tokens=40,
        )
    )
    print("\nADAPTIVE STEERING (probe vector)")
    print(
        generate_with_adaptive_steering(
            compare_prompt,
            model,
            tokenizer,
            DEFAULT_LAYER_IDX,
            probe_vector,
            probe,
            device,
            alpha=2.0,
            beta=0.0,
            max_new_tokens=40,
        )
    )

    rows = collect_evaluation_rows(
        EVALUATION_PROMPTS,
        model=model,
        tokenizer=tokenizer,
        layer_idx=DEFAULT_LAYER_IDX,
        steering_vector=steering_vector,
        probe=probe,
        probe_vector=probe_vector,
        device=device,
    )
    print("\nEvaluation rows:")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run_demo()
