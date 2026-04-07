# gh aw compile

Compile `.github/workflows/*.md` workflow sources into executable `.lock.yml` files.

## When to use it

Use after creating a workflow or any time you change frontmatter.

## How to use it

1. Start with `gh aw compile --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw compile`
- `gh aw compile daily-gh-aw-training --validate`
- `gh aw compile --strict --validate`

## Notes

- Edit the Markdown source, not the `.lock.yml` file.
- Frontmatter changes require recompilation; body-only prompt edits do not.

## Verify

- Re-run `gh aw compile --help` if you are unsure about supported flags in the installed release.
