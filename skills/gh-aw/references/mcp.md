# gh aw mcp

Inspect and manage MCP server configuration for workflows.

## When to use it

Use when adding MCP-backed tools or understanding which MCP tools a workflow can call.

## How to use it

1. Start with `gh aw mcp --help` to see the flags for your installed version.
2. Run the smallest command that answers your immediate need.
3. Review the diff or GitHub-side effect before moving on when the command changes repository state.

## Example commands

- `gh aw mcp list`
- `gh aw mcp inspect daily-gh-aw-training`
- `gh aw mcp add my-workflow tavily`

## Notes

- Useful subcommands are `list`, `list-tools`, `inspect`, and `add`.

## Verify

- Re-run `gh aw mcp --help` if you are unsure about supported flags in the installed release.
