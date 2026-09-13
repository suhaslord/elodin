#!/usr/bin/env python3
"""Replay a frozen state trace twice and require identical Xronos command traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from elodin_bridge import XronosBridge


def load_states(path: Path) -> list[dict]:
    states = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        msg = json.loads(line)
        for key in ("tick", "position", "velocity"):
            if key not in msg:
                raise ValueError(f"{path}:{lineno}: missing {key}")
        states.append(msg)
    if not states:
        raise ValueError("state trace is empty")
    return states


def run_once(states: list[dict], controller: list[str]) -> list[dict]:
    out = []
    with XronosBridge(controller) as bridge:
        for state in states:
            command = bridge.command(state["tick"], state["position"], state["velocity"])
            out.append({"tick": int(state["tick"]), "command": command})
    return out


def canonical(trace: list[dict]) -> bytes:
    text = "\n".join(json.dumps(x, sort_keys=True, separators=(",", ":")) for x in trace)
    return (text + "\n").encode()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("states", type=Path)
    p.add_argument(
        "--controller",
        nargs="+",
        default=[sys.executable, "xronos_controller.py"],
    )
    args = p.parse_args()

    states = load_states(args.states)
    run_a = run_once(states, args.controller)
    run_b = run_once(states, args.controller)

    a = canonical(run_a)
    b = canonical(run_b)
    hash_a = hashlib.sha256(a).hexdigest()
    hash_b = hashlib.sha256(b).hexdigest()

    print(f"run A sha256: {hash_a}")
    print(f"run B sha256: {hash_b}")
    if a != b:
        print("REPLAY: FAIL")
        return 1

    print("REPLAY: PASS")
    print(f"commands compared: {len(run_a)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
