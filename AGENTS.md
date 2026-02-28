# AGENTS.md

## Prompt Trigger: "agent team"

When a user prompt contains the phrase `agent team` (case-insensitive), treat it as an instruction to execute the local Codex team runner instead of asking the user to run a command manually.

### Required behavior

1. Run:
   - `python3 scripts/codex_team.py --task "<derived task>" --spawn-terminal-windows`
2. Derive `<derived task>` from the same user prompt:
   - Remove only the first `agent team` phrase.
   - Trim leading punctuation/spaces (`:`, `-`, `,`, whitespace).
   - If nothing remains, use the full original prompt as the task.
3. Unless the user explicitly asks otherwise, run the full flow (do not add `--skip-lead`).
4. After execution, report:
   - Exit status
   - Run artifacts path under `.codex-team/runs/...`
   - Concise summary of specialist + lead results

### Examples

- `agent team add a new risk guard for earnings week`
  - Run with task: `add a new risk guard for earnings week`
- `can you use an AGENT TEAM: refactor risk manager and add tests`
  - Run with task: `refactor risk manager and add tests`
