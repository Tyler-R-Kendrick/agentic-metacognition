from __future__ import annotations

import torch

from .steering import generate_with_adaptive_steering, generate_with_steering
from .models import generate


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
