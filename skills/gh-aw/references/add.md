# gh aw add

Copy one or more published agentic workflows into `.github/workflows/` without an interactive wizard.

## When to use it

Use when you already know the source workflow you want to import.

## How to use it

1. Start with `gh aw add --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw add githubnext/agentics/daily-repo-status`
- `gh aw add githubnext/agentics/workflows/ci-doctor.md@main`
- `gh aw add https://github.com/githubnext/agentics/blob/main/workflows/ci-doctor.md`

## Notes

- Prefer `add-wizard` when you also need help setting secrets or choosing an engine.

## Verify

- Re-run `gh aw add --help` if you are unsure about supported flags in the installed release.
