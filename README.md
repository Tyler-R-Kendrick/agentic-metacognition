# agentic-metacognition
Meta-layers to extract and reason about llm cognitive patterns with agents to enhance reasoning quality.

## Minimal activation steering demo

This repo includes an `activation_steering` Python package with reusable Hugging Face–based helpers for:

- collecting hidden states from one decoder block
- building a mean-difference steering vector
- injecting that vector with a forward hook during generation
- comparing baseline, fixed steering, and an optional adaptive probe-based variant

### Install

```bash
python -m pip install -r requirements.txt
```

### Dev container

This repo includes a VS Code dev container config that preinstalls the Python and
Jupyter extensions and installs the repo requirements plus `ipykernel`.

### Notebook example

Open `notebooks/minimal_activation_steering.ipynb` to run the minimal end-to-end use case. The notebook defines its own sample prompts and imports the shared `activation_steering` library helpers directly.

### Use

Use the notebook for the minimal example, or import the package from Python applications that want to build vectors, attach steering hooks, and run steered generation.
