#!/usr/bin/env bash
# PreToolUse hook: block python3/python, force `uv run`.
# Reads the tool call JSON on stdin; emits a deny decision with a reason that
# is sent back to the model so it self-corrects to `uv run`.
set -euo pipefail

COMMAND=$(jq -r '.tool_input.command // ""')

if printf '%s' "$COMMAND" | grep -qE '(^|[[:space:]])python3?([[:space:]]|$)'; then
  jq -n '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:"python3/python is forbidden in this project. Run Python with: uv run <script> (e.g. uv run scripts/inspect-ops.py)."}}'
  exit 0
fi

exit 0
