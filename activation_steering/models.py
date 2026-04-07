from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MAX_NEW_TOKENS = 80
DEFAULT_DO_SAMPLE = False


def get_default_device() -> str:
    """Return the default runtime device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(
    model_name: str,
    device: str | torch.device | None = None,
):
    """Load a decoder-only causal LM and its tokenizer."""
    run_device = torch.device(device) if device is not None else torch.device(get_default_device())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float16 if run_device.type == "cuda" else torch.float32,
    }
    if device is None and run_device.type == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(run_device)
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
    raise ValueError("Unsupported model architecture for this activation steering library.")


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
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return tuple(hidden_state.detach() for hidden_state in outputs.hidden_states)


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


def generate(
    prompt: str,
    model,
    tokenizer,
    device: str | torch.device | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    do_sample: bool = DEFAULT_DO_SAMPLE,
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
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
