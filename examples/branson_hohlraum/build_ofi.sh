#!/usr/bin/env bash
set -euo pipefail
example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$example_dir/../../../.." && pwd)
name=$(basename "$example_dir")
build_dir=${BUILD_DIR:-"$repo/build/branson_hohlraum"}
cmake -S "$repo/source/monte" -B "$build_dir" \
  -DSTORM_WITH_MPI=ON -DSTORM_BUILD_EXAMPLES=ON -DBUILD_TESTING=OFF \
  -DSTORM_WITH_COMPTON=ON -DSTORM_NO_IBV=ON \
  -DSTORM_DEPS_DIR="$repo/source/utils" \
  -DSTORM_MADCART_DIR="$repo/source/3D/tessellation/cartesian" \
  -DSTORM_MADVORO_DIR="$repo/source/3D/tessellation/voronoi" \
  -DSTORM_MESH_DECOMPOSER_DIR="$repo/source/3D/tessellation/MeshDecomposer3D"
cmake --build "$build_dir" --target "$name" "${name}_serial" -j "${BUILD_JOBS:-4}"
