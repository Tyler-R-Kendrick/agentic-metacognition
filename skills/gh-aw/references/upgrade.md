# gh aw upgrade

Upgrade repository agent files and workflow syntax to the latest gh-aw patterns.

## When to use it

Use when moving a repo forward to a newer gh-aw release.

## How to use it

1. Start with `gh aw upgrade --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw upgrade`
- `gh aw upgrade --help`

## Notes

- Run `gh aw compile --validate` after upgrading.

## Verify

- Re-run `gh aw upgrade --help` if you are unsure about supported flags in the installed release.
