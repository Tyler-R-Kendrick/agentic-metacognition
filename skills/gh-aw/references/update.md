# gh aw update

Refresh imported workflows from their upstream source repositories.

## When to use it

Use after adding shared workflows that should track upstream improvements.

## How to use it

1. Start with `gh aw update --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw update`
- `gh aw update daily-repo-status`

## Notes

- Review upstream changes carefully before shipping them.

## Verify

- Re-run `gh aw update --help` if you are unsure about supported flags in the installed release.
