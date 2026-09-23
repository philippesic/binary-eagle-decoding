"""Refresh goal context and request rotation after the second compact.

Codex runs SessionStart(source=compact) after compaction. Hooks cannot create a
successor agent; this emits a model-visible instruction at the safe boundary.
"""

import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def common_dir(cwd: str) -> Path:
    result = subprocess.run(
        ["git", "-C", cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def context(message: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": message,
                }
            }
        )
    )


def main() -> None:
    event = json.load(sys.stdin)
    cwd = event.get("cwd")
    session_id = event.get("session_id")
    if not isinstance(cwd, str) or not isinstance(session_id, str) or not session_id:
        return
    if event.get("source") != "compact":
        context(
            "For active project goals, use docs/STATUS.md and the linked goal file "
            "as durable state. Follow docs/AGENT_OPERATIONS.md for handoffs."
        )
        return
    try:
        state_dir = common_dir(cwd) / "binary-eagle-agent-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(session_id.encode()).hexdigest()[:24]
        path = state_dir / f"compact-{key}.json"
        with path.open("a+", encoding="utf-8") as state_file:
            fcntl.flock(state_file, fcntl.LOCK_EX)
            state_file.seek(0)
            raw = state_file.read()
            prior = json.loads(raw) if raw.strip() else {}
            count = int(prior.get("count", 0)) + 1
            state_file.seek(0)
            state_file.truncate()
            json.dump({"count": count}, state_file)
            state_file.flush()
            fcntl.flock(state_file, fcntl.LOCK_UN)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        context("Re-read the active goal file after compaction; checkpoint before any handoff.")
        return

    if count >= 2:
        context(
            f"This session has compacted {count} times. At the next safe boundary, "
            "checkpoint docs/STATUS.md and the active docs/goals/ file: objective, "
            "completed work, commits, tests, live agents, remote jobs, exact next "
            "actions, and unresolved user decisions. Finish or explicitly transfer "
            "live work; then launch a fresh successor Codex task with that file as "
            "the handoff and stop continuing this long-running session. "
            "Do not interrupt an in-flight GPU job solely for this rotation."
        )
    else:
        context(
            "First compaction in this session. Re-read docs/STATUS.md and the "
            "active goal file before continuing; update them after the next milestone."
        )


if __name__ == "__main__":
    main()
