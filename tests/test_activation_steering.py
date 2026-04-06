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
]


def make_test_tokenizer():
    vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
    all_texts = POSITIVE_TEXTS + NEGATIVE_TEXTS + TEST_PROMPTS + EVALUATION_PROMPTS
    for text in all_texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    return fast


def make_gpt2_model(tokenizer):
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=64,
        n_ctx=64,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def make_llama_model(tokenizer):
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def make_neox_model(tokenizer):
    config = GPTNeoXConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPTNeoXForCausalLM(config)
    model.eval()
    return model


@pytest.fixture()
def tokenizer():
    return make_test_tokenizer()


@pytest.fixture()
def model(tokenizer):
    return make_gpt2_model(tokenizer)


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


def test_activation_steerer_modifies_real_model_logits(model, tokenizer):
    inputs = steering.tokenize_text(TEST_PROMPTS[0], tokenizer, "cpu")
    with torch.no_grad():
        baseline = model(**inputs).logits
    vector, _, _ = steering.build_mean_difference_vector(
        POSITIVE_TEXTS, NEGATIVE_TEXTS, 1, model, tokenizer, "cpu"
    )
    with steering.ActivationSteerer(model, layer_idx=1, vector=vector, alpha=1.5):
        with torch.no_grad():
            steered = model(**inputs).logits
    assert not torch.allclose(baseline, steered)


def test_activation_steerer_hook_removed_after_exit(model, tokenizer):
    inputs = steering.tokenize_text(TEST_PROMPTS[0], tokenizer, "cpu")
    vector, _, _ = steering.build_mean_difference_vector(
        POSITIVE_TEXTS, NEGATIVE_TEXTS, 1, model, tokenizer, "cpu"
    )
    with torch.no_grad():
        baseline = model(**inputs).logits
    with steering.ActivationSteerer(model, layer_idx=1, vector=vector, alpha=1.5):
        _ = model(**inputs).logits
    with torch.no_grad():
        after = model(**inputs).logits
    assert torch.allclose(baseline, after)


def test_activation_steerer_zero_alpha_no_change(model, tokenizer):
    inputs = steering.tokenize_text(TEST_PROMPTS[0], tokenizer, "cpu")
    vector, _, _ = steering.build_mean_difference_vector(
        POSITIVE_TEXTS, NEGATIVE_TEXTS, 1, model, tokenizer, "cpu"
    )
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


def test_train_probe_returns_probe_and_unit_vector(model, tokenizer):
    probe, probe_vector = steering.train_probe(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    assert probe_vector.shape == (model.config.n_embd,)
    assert torch.isclose(probe_vector.norm(), torch.tensor(1.0), atol=1e-5)
    sample = steering.get_last_token_hidden(
        TEST_PROMPTS[0], 1, model=model, tokenizer=tokenizer, device="cpu"
    )
    proba = probe.predict_proba(sample.reshape(1, -1).numpy())
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


def test_adaptive_steerer_modifies_real_model_logits(model, tokenizer):
    probe, probe_vector = steering.train_probe(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    inputs = steering.tokenize_text(TEST_PROMPTS[0], tokenizer, "cpu")
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
    probe, probe_vector = steering.train_probe(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    inputs = steering.tokenize_text(TEST_PROMPTS[0], tokenizer, "cpu")
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


def test_collect_evaluation_rows_returns_expected_shape(model, tokenizer):
    steering_vector, _, _ = steering.build_mean_difference_vector(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    probe, probe_vector = steering.train_probe(
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        layer_idx=1,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    rows = steering.collect_evaluation_rows(
        EVALUATION_PROMPTS,
        model=model,
        tokenizer=tokenizer,
        layer_idx=1,
        steering_vector=steering_vector,
        probe=probe,
        probe_vector=probe_vector,
        device="cpu",
        max_new_tokens=3,
    )
    assert len(rows) == len(EVALUATION_PROMPTS)
    assert set(rows[0]) == {"prompt", "baseline", "fixed", "adaptive"}
