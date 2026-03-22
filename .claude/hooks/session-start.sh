#!/bin/bash
# Loads project context at session start
cd /home/stam/nsf 2>/dev/null || exit 0

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
LAST_COMMIT=$(git log -1 --pretty=format:"%h %s" 2>/dev/null || echo "none")
OPEN_ISSUES=$(gh issue list --limit 10 --json number,title,state 2>/dev/null || echo "[]")

jq -n \
  --arg branch "$BRANCH" \
  --arg commit "$LAST_COMMIT" \
  --arg issues "$OPEN_ISSUES" \
  '{"systemMessage": "NSF session context: branch=\($branch), last_commit=\($commit)\nOpen issues:\n\($issues)"}'
