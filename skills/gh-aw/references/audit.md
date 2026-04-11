# gh aw audit

Analyze a workflow run and produce a detailed report about what happened.

## When to use it

Use when an agentic workflow failed, behaved strangely, or needs a post-run investigation.

## How to use it

1. Start with `gh aw audit --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw audit 123456789`
- `gh aw audit 123456789 --help`

## Notes

- Pair this with `gh aw logs` when you need raw execution details.

## Verify

- Re-run `gh aw audit --help` if you are unsure about supported flags in the installed release.
