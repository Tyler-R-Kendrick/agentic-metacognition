# gh aw trial

Run a workflow in trial mode as if it were executing in a repository.

## When to use it

Use when you want a safer dry-run style validation path before enabling a workflow for real runs.

## How to use it

1. Start with `gh aw trial --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw trial daily-gh-aw-training`
- `gh aw trial daily-gh-aw-training --logical-repo owner/repo`

## Notes

- Trial mode is especially helpful while iterating on workflows with write safe-outputs.

## Verify

- Re-run `gh aw trial --help` if you are unsure about supported flags in the installed release.
