# gh aw fix

Apply automatic codemod-style fixes to workflow files.

## When to use it

Use when upgrading syntax or cleaning up deprecated patterns before compiling.

## How to use it

1. Start with `gh aw fix --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw fix`
- `gh aw fix --write`

## Notes

- Review the diff after running this command because it can rewrite workflow sources.

## Verify

- Re-run `gh aw fix --help` if you are unsure about supported flags in the installed release.
