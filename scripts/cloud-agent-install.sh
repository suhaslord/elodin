#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for Elodin (Ubuntu alternative to nix develop).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$HOME/.local/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-18/lib}"

need_cmd() { command -v "$1" >/dev/null 2>&1; }

if need_cmd sudo; then
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    just git-lfs pkg-config cmake gfortran patchelf protobuf-compiler \
    libclang-dev libssl-dev libasound2-dev libudev-dev libopenblas-dev \
    libx11-dev libxcursor-dev libxrandr-dev libxi-dev libxkbcommon-dev \
    libwayland-dev software-properties-common curl ca-certificates \
    python3-dev
  if ! need_cmd python3.13; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
      python3.13 python3.13-dev python3.13-venv
  fi
fi

if ! need_cmd uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

if need_cmd rustup; then
  rustup show active-toolchain >/dev/null
fi

git lfs install --skip-repo >/dev/null 2>&1 || true

# Drop 0-byte libelodin.so left by maturin 1.13+'s broken staging dance
find target -maxdepth 4 -name 'libelodin.so' -size 0 -delete 2>/dev/null || true

mkdir -p "${CARGO_HOME:-$HOME/.cargo}/bin"

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.13 --python-preference only-system
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uvx maturin@1.12.6 develop --uv --release --manifest-path=libs/nox-py/Cargo.toml

cargo build --release -p elodin-db
install -m 755 target/release/elodin-db "${CARGO_HOME:-$HOME/.cargo}/bin/"

echo "cloud-agent-install: python SDK and elodin-db ready"
.venv/bin/python -c "import elodin, elodin.db; print('elodin', elodin.__file__)"
elodin-db --version
