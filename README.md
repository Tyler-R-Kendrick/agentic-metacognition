# agentic-metacognition
Meta-layers to extract and reason about llm cognitive patterns with agents to enhance reasoning quality.

## Minimal activation steering demo

This repo includes `activation_steering.py`, a small Hugging Face–based example of:

- collecting hidden states from one decoder block
- building a mean-difference steering vector
- injecting that vector with a forward hook during generation
- comparing baseline, fixed steering, and an optional adaptive probe-based variant

### Install

```bash
python -m pip install -r requirements.txt
```

### Run

```bash
python activation_steering.py
```

By default it uses `Qwen/Qwen2.5-0.5B-Instruct` and prints baseline vs. steered generations plus a tiny evaluation table.
