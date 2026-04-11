# gh aw pr

Use PR utilities, including transferring a pull request between repositories.

## When to use it

Use when moving work from a trial repo to a production repo.

## How to use it

1. Start with `gh aw pr --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw pr transfer https://github.com/source/repo/pull/123 --repo owner/target`
- `gh aw pr --help`

## Notes

- Today the main subcommand is `transfer`.

## Verify

- Re-run `gh aw pr --help` if you are unsure about supported flags in the installed release.
