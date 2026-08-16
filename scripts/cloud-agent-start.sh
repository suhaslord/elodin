#!/usr/bin/env bash
# Per-boot Cloud Agent init. Creates runtime dirs; does not start servers.
set -euo pipefail

export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$HOME/.local/bin:$PATH"
mkdir -p "${ELODIN_DB_PATH:-$HOME/.local/share/elodin/db}"
mkdir -p "$HOME/.local/bin"
