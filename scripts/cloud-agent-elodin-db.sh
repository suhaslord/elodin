#!/usr/bin/env bash
# Foreground Elodin-DB for the Cloud Agent terminals panel.
set -euo pipefail

export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$HOME/.local/bin:$PATH"
DB_DIR="${ELODIN_DB_PATH:-$HOME/.local/share/elodin/db}"
mkdir -p "$(dirname "$DB_DIR")"
if [ -d "$DB_DIR" ] && [ ! -f "$DB_DIR/db_state" ]; then
  rmdir "$DB_DIR" 2>/dev/null || true
fi
exec elodin-db run 127.0.0.1:2240 "$DB_DIR" --log-level warn
