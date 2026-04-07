# gh aw project

Manage GitHub Projects V2 resources for repositories.

## When to use it

Use when you need to create and optionally link a project board.

## How to use it

1. Start with `gh aw project --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw project new "Research Backlog" --owner @me`
- `gh aw project new "Team Board" --owner myorg --link myorg/myrepo`

## Notes

- The primary subcommand is `new`.

## Verify

- Re-run `gh aw project --help` if you are unsure about supported flags in the installed release.
