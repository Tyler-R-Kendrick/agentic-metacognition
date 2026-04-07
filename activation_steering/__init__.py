"""Reusable activation steering helpers for decoder-only Hugging Face models."""

from .catalog import (
    STANDARD_ACTIVATIONS_PATH,
    get_standard_activation_models,
    get_standard_activations,
    load_standard_activation_catalog,
)
from .evaluation import collect_evaluation_rows
from .models import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_NEW_TOKENS,
    collect_last_token_hiddens,
    generate,
    get_default_device,
    get_hidden_states,
    get_last_token_hidden,
    get_model_device,
    get_transformer_blocks,
    get_transformer_layers,
    load_model_and_tokenizer,
    tokenize_text,
)
from .steering import (
    ActivationSteerer,
    AdaptiveActivationSteerer,
    build_mean_difference_vector,
    cosine,
    generate_with_adaptive_steering,
    generate_with_steering,
    train_probe,
)

__all__ = [
    "ActivationSteerer",
    "AdaptiveActivationSteerer",
    "DEFAULT_DO_SAMPLE",
    "DEFAULT_MAX_NEW_TOKENS",
    "STANDARD_ACTIVATIONS_PATH",
    "build_mean_difference_vector",
    "collect_evaluation_rows",
    "collect_last_token_hiddens",
    "cosine",
    "generate",
    "generate_with_adaptive_steering",
    "generate_with_steering",
    "get_default_device",
    "get_hidden_states",
    "get_last_token_hidden",
    "get_model_device",
    "get_standard_activation_models",
    "get_standard_activations",
    "get_transformer_blocks",
    "get_transformer_layers",
    "load_standard_activation_catalog",
    "load_model_and_tokenizer",
    "tokenize_text",
    "train_probe",
]
