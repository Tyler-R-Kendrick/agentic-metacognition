"""
Minimal Activation Steering Demo
=================================
Demonstrates concept-vector steering for a decoder-only Hugging Face LM.

Concept direction: cautious / hedged (positive) vs. overconfident / assertive (negative).

WARNING:
  - The vector is NOT a universal concept detector.
  - It reflects only the examples used to build it.
  - "Truthfulness" is much harder to define than stylistic concepts.
  - Stronger steering can reduce fluency or produce odd outputs.
  - This is NOT a faithful reproduction of ACT.

This project demonstrates the mechanics of concept-vector steering with a tiny
toy dataset.  It does not prove the existence of a universal "truth vector," and
it should be treated as a minimal interpretability experiment rather than a
production method.
"""

# ---------------------------------------------------------------------------
# 1.  Imports and config
# ---------------------------------------------------------------------------
import contextlib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LAYER_IDX = 14   # middle-to-late layer for ~24-layer models
MAX_NEW_TOKENS = 60

# ---------------------------------------------------------------------------
# 2.  Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    """Load a causal LM and its tokenizer, move to DEVICE, set to eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3.  Architecture helper
# ---------------------------------------------------------------------------

def get_transformer_layers(model):
    """Return the list/ModuleList of transformer blocks for common architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers                   # Llama / Qwen / Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h                  # GPT-2 / GPT-J style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers                # GPT-NeoX style
    raise ValueError(
        "Unsupported model architecture: cannot locate transformer block list."
    )


# ---------------------------------------------------------------------------
# 4.  Hidden-state helpers
# ---------------------------------------------------------------------------

def tokenize_text(text: str, tokenizer, device: str):
    """Tokenize *text* and move tensors to *device*."""
    return tokenizer(text, return_tensors="pt").to(device)


def get_hidden_states(text: str, model, tokenizer, device: str):
    """Run the model and return all hidden states (embedding + every layer)."""
    inputs = tokenize_text(text, tokenizer, device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states


def get_last_token_hidden(
    text: str, layer_idx: int, model, tokenizer, device: str
) -> np.ndarray:
    """
    Return the final-token hidden state at transformer block *layer_idx*.

    Note: hidden_states[0] is the embedding output, so transformer block
    *layer_idx* maps to hidden_states[layer_idx + 1].
    """
    hidden_states = get_hidden_states(text, model, tokenizer, device)
    return hidden_states[layer_idx + 1][0, -1, :].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 5.  Steering vector construction
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_mean_difference_vector(
    positive_examples: list,
    negative_examples: list,
    layer_idx: int,
    model,
    tokenizer,
    device: str,
):
    """
    Build a normalized mean-difference steering vector.

    Returns
    -------
    vec : np.ndarray  shape (hidden_dim,)  – unit-norm steering vector
    pos_hidden : list of np.ndarray        – raw positive hidden states
    neg_hidden : list of np.ndarray        – raw negative hidden states
    """
    pos_hidden = [
        get_last_token_hidden(t, layer_idx, model, tokenizer, device)
        for t in positive_examples
    ]
    neg_hidden = [
        get_last_token_hidden(t, layer_idx, model, tokenizer, device)
        for t in negative_examples
    ]

    vec = np.mean(pos_hidden, axis=0) - np.mean(neg_hidden, axis=0)
    vec = vec / np.linalg.norm(vec)
    return vec, pos_hidden, neg_hidden


# ---------------------------------------------------------------------------
# 6.  ActivationSteerer — fixed-vector context manager
# ---------------------------------------------------------------------------

class ActivationSteerer:
    """
    Context manager that adds alpha * vector to every forward pass through
    the chosen transformer block.

    Usage::

        with ActivationSteerer(model, layer_idx=14, vector=vec, alpha=1.5):
            output = generate(prompt, model, tokenizer, device)
    """

    def __init__(self, model, layer_idx: int, vector: np.ndarray, alpha: float):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = torch.tensor(vector, dtype=torch.float32)
        self.alpha = alpha
        self._hook_handle = None

    def __enter__(self):
        layers = get_transformer_layers(self.model)
        device = next(self.model.parameters()).device
        vec = self.vector.to(device)
        alpha = self.alpha

        def _hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0] + alpha * vec.reshape(1, 1, -1)
                return (hidden,) + output[1:]
            return output + alpha * vec.reshape(1, 1, -1)

        self._hook_handle = layers[self.layer_idx].register_forward_hook(_hook)
        return self

    def __exit__(self, *args):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ---------------------------------------------------------------------------
# 7.  Generation helper
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = False,
) -> str:
    """Tokenize *prompt*, generate, and decode the result."""
    inputs = tokenize_text(prompt, tokenizer, device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 9.  Probe training
# ---------------------------------------------------------------------------

def train_probe(
    positive_examples: list,
    negative_examples: list,
    layer_idx: int,
    model,
    tokenizer,
    device: str,
):
    """
    Train a logistic-regression probe on the positive/negative hidden states.

    Returns
    -------
    probe : LogisticRegression – fitted probe
    probe_vector : np.ndarray  – normalized coefficient vector (shape: hidden_dim)
    """
    pos_hidden = [
        get_last_token_hidden(t, layer_idx, model, tokenizer, device)
        for t in positive_examples
    ]
    neg_hidden = [
        get_last_token_hidden(t, layer_idx, model, tokenizer, device)
        for t in negative_examples
    ]

    X = np.array(pos_hidden + neg_hidden)
    y = [1] * len(pos_hidden) + [0] * len(neg_hidden)

    probe = LogisticRegression(max_iter=1000, random_state=SEED)
    probe.fit(X, y)

    probe_vector = probe.coef_[0]
    probe_vector = probe_vector / np.linalg.norm(probe_vector)
    return probe, probe_vector


# ---------------------------------------------------------------------------
# 10.  AdaptiveActivationSteerer — probe-scaled context manager
# ---------------------------------------------------------------------------

class AdaptiveActivationSteerer:
    """
    Context manager that scales the steering vector by how "non-positive" the
    current hidden state looks according to a linear probe.

    This is a lightweight approximation of ACT; it is NOT a faithful
    reproduction of the original paper.

    scale = alpha * (1 - p_positive + beta)
    """

    def __init__(self, model, layer_idx: int, vector: np.ndarray,
                 probe, alpha: float, beta: float = 0.0):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = torch.tensor(vector, dtype=torch.float32)
        self.probe = probe
        self.alpha = alpha
        self.beta = beta
        self._hook_handle = None

    def __enter__(self):
        layers = get_transformer_layers(self.model)
        device = next(self.model.parameters()).device
        vec = self.vector.to(device)
        probe = self.probe
        alpha = self.alpha
        beta = self.beta

        def _hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output

            last_token = hidden[0, -1, :].detach().cpu().numpy().reshape(1, -1)
            p_positive = float(probe.predict_proba(last_token)[0, 1])
            scale = alpha * (1.0 - p_positive + beta)

            steered = hidden + scale * vec.reshape(1, 1, -1)

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        self._hook_handle = layers[self.layer_idx].register_forward_hook(_hook)
        return self

    def __exit__(self, *args):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ---------------------------------------------------------------------------
# 8 + 11.  Demo — runs only when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Load model ---
    print(f"Loading {DEFAULT_MODEL} on {DEVICE} …")
    model, tokenizer = load_model_and_tokenizer()
    layers = get_transformer_layers(model)
    layer_idx = DEFAULT_LAYER_IDX
    assert layer_idx < len(layers), (
        f"layer_idx={layer_idx} out of range (model has {len(layers)} layers)"
    )
    print(f"Model loaded. Using layer {layer_idx} of {len(layers)}.\n")

    # --- Toy dataset: cautious vs. overconfident ---
    positive_texts = [
        "I believe the capital of Australia is Canberra, though I may be mistaken.",
        "It seems likely that water boils at 100°C at sea level, but conditions can vary.",
        "I think Einstein developed the theory of relativity, but I would recommend verifying.",
        "The answer might be 42, though I'm not entirely certain.",
        "As far as I know, Python was created by Guido van Rossum.",
        "I'm not fully sure, but I believe the Earth is approximately 4.5 billion years old.",
    ]
    negative_texts = [
        "Obviously Sydney is the capital of Australia.",
        "Water always boils at exactly 100°C no matter what.",
        "Newton obviously invented relativity, everyone knows this.",
        "The answer is definitely 42. No question about it.",
        "Python was created by Linus Torvalds, obviously.",
        "The Earth is exactly 6,000 years old. This is a fact.",
    ]

    # --- Build steering vector ---
    print("Building mean-difference steering vector …")
    steer_vec, pos_h, neg_h = build_mean_difference_vector(
        positive_texts, negative_texts, layer_idx, model, tokenizer, DEVICE
    )
    # Quick sanity check
    pos_proj = cosine(np.mean(pos_h, axis=0), steer_vec)
    neg_proj = cosine(np.mean(neg_h, axis=0), steer_vec)
    print(f"  cosine(mean_pos, vec) = {pos_proj:.3f}")
    print(f"  cosine(mean_neg, vec) = {neg_proj:.3f}\n")

    # --- Test prompts ---
    test_prompts = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is the speed of light?",
        "Who wrote Hamlet?",
    ]

    ALPHA = 1.5
    print("=" * 60)
    print(f"Baseline vs. steered (alpha={ALPHA})")
    print("=" * 60)
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        base_out = generate(prompt, model, tokenizer, DEVICE)
        print(f"  Baseline : {base_out!r}")

        with ActivationSteerer(model, layer_idx, steer_vec, alpha=ALPHA):
            steered_out = generate(prompt, model, tokenizer, DEVICE)
        print(f"  Steered  : {steered_out!r}")

    # --- Alpha sweep on one prompt ---
    sweep_prompt = test_prompts[0]
    print(f"\n{'=' * 60}")
    print(f"Alpha sweep on: {sweep_prompt!r}")
    print("=" * 60)
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        with ActivationSteerer(model, layer_idx, steer_vec, alpha=alpha):
            out = generate(sweep_prompt, model, tokenizer, DEVICE)
        print(f"  alpha={alpha:.1f}: {out!r}")

    # --- Optional: probe training + adaptive steering ---
    print(f"\n{'=' * 60}")
    print("Training linear probe …")
    print("=" * 60)
    probe, probe_vec = train_probe(
        positive_texts, negative_texts, layer_idx, model, tokenizer, DEVICE
    )
    print(f"  cosine(steer_vec, probe_vec) = {cosine(steer_vec, probe_vec):.3f}\n")

    compare_prompt = test_prompts[0]
    print(f"Comparison on: {compare_prompt!r}")
    base_out = generate(compare_prompt, model, tokenizer, DEVICE)
    print(f"  Baseline  : {base_out!r}")

    with ActivationSteerer(model, layer_idx, steer_vec, alpha=ALPHA):
        fixed_out = generate(compare_prompt, model, tokenizer, DEVICE)
    print(f"  Fixed     : {fixed_out!r}")

    with AdaptiveActivationSteerer(
        model, layer_idx, probe_vec, probe, alpha=ALPHA, beta=0.0
    ):
        adaptive_out = generate(compare_prompt, model, tokenizer, DEVICE)
    print(f"  Adaptive  : {adaptive_out!r}")

    print("\nDone.")
