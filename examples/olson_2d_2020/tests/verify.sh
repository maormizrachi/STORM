#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
storm=$(cd "$here/../../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"${CXX:-c++}" -std=c++17 -O2 -I"$storm" -I"$storm/deps/CMMC/src" "$here/verify.cpp" -o "$work/verify"
"$work/verify" > "$work/physics.txt"
python3 "$here/verify.py" "$work/physics.txt"
