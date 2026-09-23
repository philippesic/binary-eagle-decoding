"""Manage machine-local GPU hosts and the shared pause request."""

import argparse
import json
import os
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path.home() / ".config" / "binary-eagle-decoding"
HOSTS = ROOT / "hosts.toml"
CONTROL = ROOT / "gpu-control.json"
MACHINES = ("rtx5080", "rtx2080ti")


def ensure_hosts() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if HOSTS.exists():
        return
    HOSTS.write_text(
        "# Machine-local; shared by all worktrees. No secrets here.\n"
        '[rtx5080]\nhost = ""\nuser = ""\nport = 22\n'
        'workdir = "~/binary-eagle-decoding"\n\n'
        '[rtx2080ti]\nhost = ""\nuser = ""\nport = 22\n'
        'workdir = "~/binary-eagle-decoding"\n'
    )
    os.chmod(HOSTS, 0o600)


def write_hosts(hosts: dict) -> None:
    lines = ["# Machine-local; shared by all worktrees. No secrets here."]
    for machine in MACHINES:
        entry = hosts[machine]
        lines += [
            f"[{machine}]",
            f'host = "{entry["host"]}"',
            f'user = "{entry["user"]}"',
            f"port = {entry['port']}",
            f'workdir = "{entry["workdir"]}"',
            "",
        ]
    temp = HOSTS.with_suffix(".tmp")
    temp.write_text("\n".join(lines))
    os.chmod(temp, 0o600)
    temp.replace(HOSTS)


def validate_field(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._:/~+-]+", value):
        raise SystemExit(f"Invalid {label}: use a plain IP/hostname or SSH username")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    set_host = sub.add_parser("set-host")
    set_host.add_argument("machine", choices=MACHINES)
    set_host.add_argument("host")
    set_host.add_argument("user")
    set_host.add_argument("--port", type=int, default=22)
    for action in ("pause", "resume"):
        sub.add_parser(action).add_argument("machine", choices=(*MACHINES, "all"))
    args = parser.parse_args()
    ensure_hosts()
    if args.action == "init":
        print(HOSTS)
        return
    if args.action == "set-host":
        validate_field(args.host, "host")
        validate_field(args.user, "user")
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        hosts = tomllib.loads(HOSTS.read_text())
        hosts[args.machine].update(host=args.host, user=args.user, port=args.port)
        write_hosts(hosts)
        print(f"Updated {args.machine} in {HOSTS}")
        return
    control = json.loads(CONTROL.read_text()) if CONTROL.exists() else {}
    if args.action in ("pause", "resume"):
        targets = MACHINES if args.machine == "all" else (args.machine,)
        for machine in targets:
            control[machine] = {
                "pause_requested": args.action == "pause",
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        temp = CONTROL.with_suffix(".tmp")
        temp.write_text(json.dumps(control, indent=2) + "\n")
        os.chmod(temp, 0o600)
        temp.replace(CONTROL)
        if args.action == "pause":
            print("New runs blocked. Stop active jobs via tmux MCP and verify GPU memory is free.")
        else:
            print("New runs allowed. Confirm host availability before starting.")
        return
    print(
        json.dumps(
            {
                "hosts_file": str(HOSTS),
                "hosts": tomllib.loads(HOSTS.read_text()),
                "gpu_control": control,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
