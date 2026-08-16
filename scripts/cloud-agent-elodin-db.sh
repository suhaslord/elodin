#!/usr/bin/env bash
# Foreground Elodin-DB for the Cloud Agent terminals panel.
set -euo pipefail

export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$HOME/.local/bin:$PATH"
DB_DIR="${ELODIN_DB_PATH:-$HOME/.local/share/elodin/db}"
mkdir -p "$DB_DIR"
exec elodin-db run 127.0.0.1:2240 "$DB_DIR" --log-level warn
