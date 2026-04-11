# gh aw validate

Validate workflows without writing `.lock.yml` files.

## When to use it

Use for fast syntax and configuration checks while editing.

## How to use it

1. Start with `gh aw validate --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw validate`
- `gh aw validate daily-gh-aw-training`

## Notes

- Use `compile --validate` when you also want generated lock files refreshed.

## Verify

- Re-run `gh aw validate --help` if you are unsure about supported flags in the installed release.
