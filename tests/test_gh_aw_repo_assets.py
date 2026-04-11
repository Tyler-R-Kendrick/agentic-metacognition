from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_devcontainer_installs_github_cli_and_gh_aw() -> None:
    config = json.loads((REPO_ROOT / '.devcontainer' / 'devcontainer.json').read_text(encoding='utf-8'))

    assert config['features']['ghcr.io/devcontainers/features/github-cli:1'] == {}
    assert 'install-gh-aw.sh' in config['postCreateCommand']
    assert 'github/gh-aw/13ac7dee59ec5127393ec22dc3f4d0f6987a3842/install-gh-aw.sh' in config['postCreateCommand']
    assert 'bash /tmp/install-gh-aw.sh v0.67.1' in config['postCreateCommand']


def test_copilot_setup_steps_installs_repo_dependencies_and_gh_aw() -> None:
    workflow_text = (REPO_ROOT / '.github' / 'workflows' / 'copilot-setup-steps.yml').read_text(encoding='utf-8')

    assert 'copilot-setup-steps:' in workflow_text
    assert 'actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd' in workflow_text
    assert 'actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065' in workflow_text
    assert 'python -m pip install -r requirements.txt' in workflow_text
    assert 'github/gh-aw/13ac7dee59ec5127393ec22dc3f4d0f6987a3842/install-gh-aw.sh' in workflow_text
    assert 'gh aw version' in workflow_text


def test_daily_training_workflow_delegates_feature_issue_to_copilot() -> None:
    workflow_text = (REPO_ROOT / '.github' / 'workflows' / 'daily-gh-aw-training.md').read_text(encoding='utf-8')
    lockfile_text = (REPO_ROOT / '.github' / 'workflows' / 'daily-gh-aw-training.lock.yml').read_text(encoding='utf-8')

    assert 'schedule: daily on weekdays' in workflow_text
    assert 'toolsets: [default, actions]' in workflow_text
    assert 'cat activation_steering/models.py' in workflow_text
    assert 'cat activation_steering/features.py' in workflow_text
    assert 'cat activation_steering/discovery.py' in workflow_text
    assert 'python *' not in workflow_text
    assert 'assignees: copilot' in workflow_text
    assert 'activation_steering/data/standard_feature_specs.json' in workflow_text
    assert 'tests/data/minimal_identified_feature_vectors.json' in workflow_text
    assert 'python -m pytest tests/ -q' in workflow_text
    assert 'call `noop`' in workflow_text
    assert 'GITHUB_TOKEN: ${{ github.token }}' in lockfile_text


def test_gh_aw_skill_links_reference_docs_for_all_documented_commands() -> None:
    expected_commands = {
        'add',
        'add-wizard',
        'audit',
        'checks',
        'compile',
        'completion',
        'disable',
        'domains',
        'enable',
        'fix',
        'hash-frontmatter',
        'health',
        'init',
        'list',
        'logs',
        'mcp',
        'mcp-server',
        'new',
        'pr',
        'project',
        'remove',
        'run',
        'secrets',
        'status',
        'trial',
        'update',
        'upgrade',
        'validate',
        'version',
    }
    skill_text = (REPO_ROOT / 'skills' / 'gh-aw' / 'SKILL.md').read_text(encoding='utf-8')
    reference_dir = REPO_ROOT / 'skills' / 'gh-aw' / 'references'
    actual_commands = {path.stem for path in reference_dir.glob('*.md')}

    assert actual_commands == expected_commands
    for command in sorted(expected_commands):
        assert f'./references/{command}.md' in skill_text



def test_agentic_workflow_lockfile_and_gitattributes_are_checked_in() -> None:
    assert (REPO_ROOT / '.github' / 'workflows' / 'daily-gh-aw-training.lock.yml').exists()
    gitattributes = (REPO_ROOT / '.gitattributes').read_text(encoding='utf-8')

    assert '.github/workflows/*.lock.yml linguist-generated=true merge=ours' in gitattributes
