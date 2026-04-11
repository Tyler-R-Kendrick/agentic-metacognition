# gh aw remove

Remove workflow files that match a workflow name or pattern.

## When to use it

Use when deleting a workflow from the repository.

## How to use it

1. Start with `gh aw remove --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw remove daily-gh-aw-training`
- `gh aw remove "daily-*"`

## Notes

- Review the diff before committing so you do not remove unrelated workflows.

## Verify

- Re-run `gh aw remove --help` if you are unsure about supported flags in the installed release.
