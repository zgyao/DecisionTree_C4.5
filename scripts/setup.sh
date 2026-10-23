#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Requires a C++20 compiler (GCC 11+ or recent Clang), make and Python 3.9+.
# On Debian/Ubuntu, install prerequisites with:
#   sudo apt-get update && sudo apt-get install -y build-essential python3
# This script itself needs no root access, network access or package downloads.
for tool in "${CXX:-g++}" make python3; do
    if ! command -v "$tool" > /dev/null; then
        echo "Missing prerequisite: $tool; see README.md" >&2
        exit 1
    fi
done
case "${1:-}" in
    ""|--sanitize) ;;
    *) echo "Usage: scripts/setup.sh [--sanitize]" >&2; exit 1 ;;
esac
python3 scripts/prepare_data.py --check
make -j "${BUILD_JOBS:-2}" all
make test
if [[ "${1:-}" == --sanitize ]]; then
    make -j "${BUILD_JOBS:-2}" sanitize
fi
