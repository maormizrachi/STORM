#!/usr/bin/env bash
# install_deps.sh — Clone RDMont's external dependencies for standalone builds.
#
# When RDMont is used inside RICH, these packages are already present as
# submodules.  Standalone users should run this script once before building.
#
# Usage:
#   ./install_deps.sh [DEST_DIR]
#
# DEST_DIR defaults to ./deps

set -euo pipefail

DEST="${1:-deps}"
mkdir -p "$DEST"

clone_if_missing() {
    local name="$1" url="$2" branch="${3:-}"
    local target="$DEST/$name"
    if [ -d "$target" ]; then
        echo "[skip] $name already exists at $target"
    else
        echo "[clone] $name -> $target"
        if [ -n "$branch" ]; then
            git clone --depth 1 -b "$branch" "$url" "$target"
        else
            git clone --depth 1 "$url" "$target"
        fi
    fi
}

echo "=== Installing RDMont dependencies into $DEST ==="
echo ""

# MPI utilities (serialization, exchange, collectives)
clone_if_missing "mpi_utils" "https://github.com/maormizrachi/mpi_utils.git"

# MadCart (Cartesian mesh, for examples)
clone_if_missing "MadCart" "https://github.com/maormizrachi/MadCart.git"

echo ""
echo "=== Done ==="
echo ""
echo "Pass the following to CMake:"
echo "  -DRDMONT_DEPS_DIR=$(cd "$DEST" && pwd)"
