# gh aw hash-frontmatter

Compute the frontmatter hash for a workflow.

## When to use it

Use when you need to understand whether config changed in a way that should affect the lock file.

## How to use it

1. Start with `gh aw hash-frontmatter --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw hash-frontmatter daily-gh-aw-training`
- `gh aw hash-frontmatter .github/workflows/daily-gh-aw-training.md`

## Notes

- This is mainly helpful for debugging generated workflow changes.

## Verify

- Re-run `gh aw hash-frontmatter --help` if you are unsure about supported flags in the installed release.
