# gh aw checks

Classify pull-request CI state so you can quickly understand check health.

## When to use it

Use when reviewing a PR and you want a summarized interpretation of checks.

## How to use it

1. Start with `gh aw checks --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw checks 42`
- `gh aw checks 42 --repo owner/repo`

## Notes

- Run `gh aw checks --help` to confirm the accepted PR identifier format.

## Verify

- Re-run `gh aw checks --help` if you are unsure about supported flags in the installed release.
