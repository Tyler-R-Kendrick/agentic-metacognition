---
name: Daily gh aw training
description: Research training artifacts for promising missing feature extractions, then open one delegated Copilot issue for the best candidate
on:
  schedule: daily on weekdays
permissions:
  contents: read
  issues: read
  pull-requests: read
  actions: read
engine: copilot
network:
  allowed:
    - defaults
    - github
tools:
  github:
    toolsets: [default, actions]
  bash:
    - "find activation_steering tests -type f"
    - "cat README.md"
    - "cat activation_steering/models.py"
    - "cat activation_steering/features.py"
    - "cat activation_steering/discovery.py"
    - "cat activation_steering/data/*.json"
    - "cat tests/data/*.json"
    - "git log --oneline"
safe-outputs:
  create-issue:
    title-prefix: "[daily-gh-aw-training] "
    assignees: copilot
    max: 1
timeout-minutes: 20
strict: true
---

# Daily gh aw training

Research the repository's feature-extraction training artifacts and identify **one** high-value feature that is not already extracted but looks actionable for the models this repository supports.

## Research scope

Inspect these sources before proposing work:

- `README.md`
- `activation_steering/models.py`
- `activation_steering/features.py`
- `activation_steering/discovery.py`
- `activation_steering/data/standard_activations.json`
- `activation_steering/data/standard_feature_specs.json`
- `tests/data/minimal_identified_feature_vectors.json`
- recent workflow runs or artifacts related to training, activation steering, or feature extraction, if any exist in GitHub Actions
- open issues and pull requests so you do not duplicate work that is already planned or in progress

## Candidate selection rules

Only propose a feature when all of the following are true:

1. It is **not** already represented in `activation_steering/data/standard_feature_specs.json`.
2. The repository's existing artifacts suggest the feature is observable or extractable with the current activation-steering pipeline.
3. The feature would be useful for steering, evaluation, or reflective reasoning quality.
4. The work looks feasible for the repository's currently supported models and APIs.
5. There is no open issue or active pull request already covering the same idea.

If you cannot find a novel, evidence-backed candidate, call `noop` with a short explanation instead of creating an issue.

## What to create when you find a candidate

Create exactly **one** issue for the best candidate and assign it to Copilot using the configured safe output.

The issue should be implementation-ready and include:

- a concise title naming the missing feature
- why the feature appears in the training artifacts or repository evidence
- why it is useful
- the files or modules most likely to change
- a step-by-step implementation plan
- acceptance criteria that require:
  - updating the feature specification/catalog so the feature is represented explicitly
  - extending the extraction/discovery flow for the repository's supported models where needed
  - adding or updating focused tests
  - running `python -m pytest tests/ -q`

## Issue body template

Use this structure:

```markdown
# Proposed feature extraction: <feature name>

## Why this feature
- Evidence from artifacts:
- Why it matters:

## Suggested implementation
1. Review the current feature-spec and discovery pipeline.
2. Add or extend the feature definition.
3. Update extraction/discovery logic for supported models as needed.
4. Add or update tests and fixture data.
5. Run `python -m pytest tests/ -q`.

## Likely files
- `activation_steering/features.py`
- `activation_steering/discovery.py`
- `activation_steering/data/standard_feature_specs.json`
- `tests/...`

## Acceptance criteria
- [ ] A new useful feature is defined and documented in the repository artifacts.
- [ ] Feature extraction works for the supported models in this repo.
- [ ] Tests cover the new feature extraction path.
- [ ] `python -m pytest tests/ -q` passes.
```
