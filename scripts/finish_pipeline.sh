#!/usr/bin/env bash
# Compatibility entry point; successful update publication is managed by the CLI.
set -euo pipefail
cd "$(dirname "$0")/.."

# Existing callers can still pass up to three in-flight process IDs first.
while [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; do
  while ps -p "$1" > /dev/null 2>&1; do sleep 60; done
  shift
done
exec uv run --python 3.12 --extra all music-tastes update "$@"
