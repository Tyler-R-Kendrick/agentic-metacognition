---
name: activation-steering
description: Build and apply activation steering vectors for decoder-only Hugging Face models. Use this skill whenever the user wants to steer, bias, or adjust an LLM's hidden-state activations, build contrastive mean-difference vectors, inject steering hooks during generation, compare baseline vs steered outputs, or train adaptive probes. Also use when the user mentions activation engineering, representation engineering, steering vectors, or contrastive pairs — even if they don't say "activation steering" explicitly.
---

# Activation Steering

Use this skill to build, apply, and evaluate activation steering vectors on decoder-only Hugging Face transformer models via the `activation_steering` Python package in this repository.

## When to use this skill

- Building a steering vector from contrastive (positive/negative) text examples
- Injecting a steering vector into a transformer block during generation
- Comparing baseline generation with fixed-alpha or adaptive steered generation
- Training a logistic-regression probe for adaptive steering
- Collecting hidden states from specific decoder layers
- Working with decaying or adaptive steering strategies

## Key concepts

**Mean-difference vector**: A normalized vector computed as the difference between mean last-token hidden states from positive and negative examples. This is the core steering signal.

**Alpha (α)**: The scalar multiplier that controls steering strength. Higher values produce stronger steering effects. Typical range is 0.5–3.0.

**Layer index**: The transformer decoder block where the steering hook is attached. Middle layers (e.g., layer 5 for GPT-2's 12 layers) are common starting points.

**Adaptive steering**: Uses a logistic-regression probe trained on the same contrastive examples to decide steering intensity per token. The probe monitors the model's hidden state and adjusts alpha dynamically.

**Decaying steering**: Applies a geometric decay to alpha over generation steps, so steering influence fades as the model generates more tokens.

## Quick-start workflow

1. **Load model and tokenizer**
   ```python
   from activation_steering import load_model_and_tokenizer, get_default_device
   model, tokenizer = load_model_and_tokenizer("gpt2")
   device = get_default_device()
   ```

2. **Define contrastive examples**
   ```python
   positive = ["The scientist carefully analyzed the data and reported findings."]
   negative = ["The scientist made up results without any evidence."]
   ```

3. **Build the steering vector**
   ```python
   from activation_steering import build_mean_difference_vector
   vector, pos_hiddens, neg_hiddens = build_mean_difference_vector(
       positive, negative, layer_idx=5, model=model, tokenizer=tokenizer, device=device
   )
   ```

4. **Generate with steering**
   ```python
   from activation_steering import generate_with_steering
   result = generate_with_steering(
       "The researcher found that",
       model, tokenizer, layer_idx=5, vector=vector, device=device, alpha=1.5
   )
   ```

## Available steering strategies

| Strategy | Function | When to use |
|----------|----------|-------------|
| Fixed | `generate_with_steering(...)` | Constant steering strength throughout |
| Adaptive | `generate_with_adaptive_steering(...)` | Probe-guided per-token intensity |
| Decaying | `generate_with_decaying_steering(...)` | Steering fades over generation steps |

## Steerer classes

For more control, use the OOP interface:

- `ActivationSteerer(model, layer_idx, vector, alpha)` — fixed hook
- `AdaptiveActivationSteerer(model, layer_idx, vector, alpha, probe, probe_vector)` — probe-guided
- `DecayingActivationSteerer(model, layer_idx, vector, alpha, decay)` — geometric decay

Each steerer is a context manager that attaches/removes the forward hook automatically.

## Evaluation harness

Use `collect_evaluation_rows(...)` to run a quick comparison across baseline, fixed, and adaptive steering for a list of prompts. Returns a list of dicts with keys `prompt`, `baseline`, `fixed`, `adaptive`.

## Utility functions

- `cosine(a, b)` — cosine similarity between two 1-D tensors
- `train_probe(pos_hiddens, neg_hiddens)` — train a logistic-regression probe for adaptive steering
- `collect_last_token_hiddens(texts, layer_idx, model, tokenizer, device)` — collect hidden states
- `get_hidden_states(model, input_ids, layer_idx)` — get hidden states for a single input
- `get_last_token_hidden(model, input_ids, layer_idx)` — get the last token's hidden state
- `tokenize_text(text, tokenizer, device)` — tokenize text for model input

## API reference

For the full API surface, read `activation_steering/steering.py` (vector building, steerer classes, generation functions) and `activation_steering/models.py` (model loading, hidden-state collection, tokenization).
