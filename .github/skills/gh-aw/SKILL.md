---
name: gh-aw
description: Use when creating, compiling, validating, running, or debugging GitHub Agentic Workflows with the gh-aw CLI.
---

# gh-aw

Use this skill whenever work touches GitHub Agentic Workflows in `.github/workflows/`, their generated `.lock.yml` files, or the `gh aw` command line workflow around them.

## When to use this skill

- creating a new agentic workflow
- editing workflow frontmatter or prompt instructions
- compiling or validating `.md` workflows into `.lock.yml` files
- enabling, disabling, or manually running workflows
- investigating workflow logs, health, status, or audits
- managing secrets or MCP integrations used by gh-aw workflows
- setting up local or Copilot environments so `gh aw` is installed and ready

## Working rules

1. Treat `.github/workflows/*.md` as the source of truth and `.github/workflows/*.lock.yml` as generated output.
2. Re-run `gh aw compile --validate` after any frontmatter change or when you add a new workflow.
3. Keep the agent job read-only; use `safe-outputs:` for issue, PR, or comment writes.
4. Prefer fuzzy schedules like `daily on weekdays` when the workflow is meant to run regularly.
5. Ensure `.gitattributes` marks `.lock.yml` files as generated with `linguist-generated=true merge=ours`.
6. If `gh aw` is missing, install it with an immutable script URL such as `curl -fsSL https://raw.githubusercontent.com/github/gh-aw/13ac7dee59ec5127393ec22dc3f4d0f6987a3842/install-gh-aw.sh -o /tmp/install-gh-aw.sh && bash /tmp/install-gh-aw.sh v0.67.1 && rm -f /tmp/install-gh-aw.sh`, then verify with `gh aw version`.

## Recommended authoring loop

1. Use `gh aw new <workflow-id>` or write `.github/workflows/<workflow-id>.md` directly.
2. Edit the workflow instructions in Markdown and keep frontmatter minimal and secure.
3. Run `gh aw validate <workflow-id>` while iterating.
4. Run `gh aw compile <workflow-id> --validate` before committing.
5. Use `gh aw run <workflow-id>` or `gh aw trial <workflow-id>` when you need an execution check.
6. Use `gh aw status`, `gh aw logs`, and `gh aw audit` to troubleshoot runtime behavior.

## Command references

- [`gh aw add`](./references/add.md)
- [`gh aw add-wizard`](./references/add-wizard.md)
- [`gh aw audit`](./references/audit.md)
- [`gh aw checks`](./references/checks.md)
- [`gh aw compile`](./references/compile.md)
- [`gh aw completion`](./references/completion.md)
- [`gh aw disable`](./references/disable.md)
- [`gh aw domains`](./references/domains.md)
- [`gh aw enable`](./references/enable.md)
- [`gh aw fix`](./references/fix.md)
- [`gh aw hash-frontmatter`](./references/hash-frontmatter.md)
- [`gh aw health`](./references/health.md)
- [`gh aw init`](./references/init.md)
- [`gh aw list`](./references/list.md)
- [`gh aw logs`](./references/logs.md)
- [`gh aw mcp`](./references/mcp.md)
- [`gh aw mcp-server`](./references/mcp-server.md)
- [`gh aw new`](./references/new.md)
- [`gh aw pr`](./references/pr.md)
- [`gh aw project`](./references/project.md)
- [`gh aw remove`](./references/remove.md)
- [`gh aw run`](./references/run.md)
- [`gh aw secrets`](./references/secrets.md)
- [`gh aw status`](./references/status.md)
- [`gh aw trial`](./references/trial.md)
- [`gh aw update`](./references/update.md)
- [`gh aw upgrade`](./references/upgrade.md)
- [`gh aw validate`](./references/validate.md)
- [`gh aw version`](./references/version.md)

## Useful setup snippets

### Dev container

```json
{
  "features": {
    "ghcr.io/devcontainers/features/github-cli:1": {}
  },
  "postCreateCommand": "curl -fsSL https://raw.githubusercontent.com/github/gh-aw/13ac7dee59ec5127393ec22dc3f4d0f6987a3842/install-gh-aw.sh -o /tmp/install-gh-aw.sh && bash /tmp/install-gh-aw.sh v0.67.1 && rm -f /tmp/install-gh-aw.sh"
}
```

### Copilot setup workflow step

```yaml
- name: Install gh-aw extension
  run: curl -fsSL https://raw.githubusercontent.com/github/gh-aw/13ac7dee59ec5127393ec22dc3f4d0f6987a3842/install-gh-aw.sh -o /tmp/install-gh-aw.sh && bash /tmp/install-gh-aw.sh v0.67.1 && rm -f /tmp/install-gh-aw.sh
```
