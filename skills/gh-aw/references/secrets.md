# gh aw secrets

Manage GitHub Actions secrets required by agentic workflows.

## When to use it

Use when a workflow needs AI tokens, GitHub tokens, or agent-assignment credentials.

## How to use it

1. Start with `gh aw secrets --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw secrets set GH_AW_AGENT_TOKEN --value "<token>"`
- `gh aw secrets bootstrap`

## Notes

- The main subcommands are `set` and `bootstrap`.

## Verify

- Re-run `gh aw secrets --help` if you are unsure about supported flags in the installed release.
