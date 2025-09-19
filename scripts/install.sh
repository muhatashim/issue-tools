#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
INCLUDE_TESTS=0

if [[ "${1:-}" == "--with-tests" ]]; then
  INCLUDE_TESTS=1
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "Usage: scripts/install.sh [--with-tests]" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found. Set PYTHON_BIN to a Python 3.11+ interpreter." >&2
  exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

PIP_BIN="${VENV_PATH}/bin/pip"

"$PIP_BIN" install --upgrade pip

TARGET="."
if [[ $INCLUDE_TESTS -eq 1 ]]; then
  TARGET=".[test]"
fi

(
  cd "$ROOT_DIR"
  "$PIP_BIN" install -e "$TARGET"
)

echo "issue-tools installed in ${VENV_PATH}"
echo "Run ${VENV_PATH}/bin/issue-tools or use ./issue-tools"
