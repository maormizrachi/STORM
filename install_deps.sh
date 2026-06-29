#!/usr/bin/env bash
# install_deps.sh — Clone STORM's external dependencies for standalone builds.
#
# When STORM is used inside RICH, these packages are already present as
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

echo "=== Installing STORM dependencies into $DEST ==="
echo ""

# MPI utilities (serialization, exchange, collectives)
clone_if_missing "mpi_utils" "git@github.com:maormizrachi/mpi_utils.git"

# Spatial data structures (OctTree, KDTree, RangeTree, BoundingBox, Sphere)
clone_if_missing "spatial_ds" "git@github.com:maormizrachi/spatial_ds.git"

# Mesh decomposition (Hilbert ordering, load balancing, points manager)
clone_if_missing "MeshDecomposer3D" "git@github.com:maormizrachi/MeshDecomposer3D.git"

# MadVoro (3D Voronoi tessellation)
clone_if_missing "MadVoro" "git@github.com:maormizrachi/MadVoro.git"

# MadCart (3D Cartesian mesh)
clone_if_missing "MadCart" "git@github.com:maormizrachi/MadCart.git"

# EasyRMA (Remote Memory Agent — RMA-based one-sided MPI communication, needed for MPI builds)
clone_if_missing "rma" "git@github.com:maormizrachi/EasyRMA.git"

echo ""
echo "=== Done ==="
echo ""
echo "CMake will find deps/ automatically. To use a custom path:"
echo "  -DSTORM_DEPS_DIR=/path/to/deps"
