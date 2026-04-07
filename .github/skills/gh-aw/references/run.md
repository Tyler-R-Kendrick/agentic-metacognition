# gh aw run

Trigger one or more agentic workflows on GitHub Actions.

## When to use it

Use when you want to manually execute a workflow after authoring or updating it.

## How to use it

1. Start with `gh aw run --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw run daily-gh-aw-training`
- `gh aw run daily-gh-aw-training --ref main`

## Notes

- This generally uses `workflow_dispatch` under the hood.

## Verify

- Re-run `gh aw run --help` if you are unsure about supported flags in the installed release.
