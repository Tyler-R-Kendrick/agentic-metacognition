# gh aw add-wizard

Interactively import workflows and walk through setup tasks such as engine choice and secrets.

## When to use it

Use when you want guided setup instead of a direct import.

## How to use it

1. Start with `gh aw add-wizard --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw add-wizard githubnext/agentics/daily-repo-status`
- `gh aw add-wizard`

## Notes

- This is the safest import path for first-time setup in a repository.

## Verify

- Re-run `gh aw add-wizard --help` if you are unsure about supported flags in the installed release.
