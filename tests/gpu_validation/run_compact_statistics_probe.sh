#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_gpu_probe}"

cmake --build "$BUILD_DIR" --target compact_statistics_probe -j
"$BUILD_DIR/bin/compact_statistics_probe"
