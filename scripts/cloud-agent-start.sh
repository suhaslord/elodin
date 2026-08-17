#!/usr/bin/env bash
# Per-boot Cloud Agent init. Creates runtime dirs; does not start servers.
set -euo pipefail

export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$HOME/.local/bin:$PATH"
# Parent only: an empty DB dir is treated as an existing database and fails to open.
mkdir -p "$(dirname "${ELODIN_DB_PATH:-$HOME/.local/share/elodin/db}")"
mkdir -p "$HOME/.local/bin"
