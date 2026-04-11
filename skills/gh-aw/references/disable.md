# gh aw disable

Disable one or more agentic workflows in the repository.

## When to use it

Use when a workflow should stay in the repo but should not run.

## How to use it

1. Start with `gh aw disable --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw disable daily-gh-aw-training`
- `gh aw disable --help`

## Notes

- Re-enable with `gh aw enable` when the workflow is ready again.

## Verify

- Re-run `gh aw disable --help` if you are unsure about supported flags in the installed release.
