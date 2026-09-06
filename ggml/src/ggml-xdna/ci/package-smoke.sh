#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || $(uname -s) != Linux || $EUID == 0 ]]; then
    echo "usage (unprivileged Linux): bash $0 OUTPUT_DIR XRT_PREFIX [JOBS=2]" >&2
    exit 2
fi
readonly here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly repo=$(cd -- "$here/../../../.." && pwd -P)
readonly output=$(realpath -m -- "$1")
readonly sdk=$(realpath -e -- "$2")
readonly jobs=${3:-2}
[[ $jobs =~ ^[1-9][0-9]*$ && -d "$sdk/share/cmake/XRT" && -d "$sdk/lib" ]]
[[ ! -L "$1" && ( ! -e "$output" || -d "$output" ) ]]
if [[ -d "$output" && -n $(find "$output" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "Output directory must be empty: $output" >&2
    exit 2
fi
mkdir -p -- "$output"
stage=initial
trap 'rc=$?; printf "stage=%s exit_code=%s hardware_execution=false\n" "$stage" "$rc" > "$output/terminal.txt"' EXIT
run() {
    stage=$1
    shift
    printf '%s\n' "$stage"
    "$@" > "$output/$stage.log" 2>&1
}
common=(-G Ninja "-DLLAMA_SOURCE=$repo" -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_BINDIR=bin
        -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF)
for mode in static dl off; do
    options=(-DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DGGML_XDNA=ON
             "-DGGML_XDNA_XRT_ROOT=$sdk" "-DXRT_DIR=$sdk/share/cmake/XRT")
    if [[ $mode == dl ]]; then
        options+=(-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON)
    elif [[ $mode == off ]]; then
        options=(-DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DGGML_XDNA=OFF
                 -DCMAKE_DISABLE_FIND_PACKAGE_XRT=TRUE)
    fi
    run "configure-$mode" cmake -S "$here/producer" -B "$output/build-$mode" "${common[@]}" \
        "${options[@]}" "-DCMAKE_INSTALL_PREFIX=$output/install-$mode"
    run "build-$mode" cmake --build "$output/build-$mode" --parallel "$jobs"
    run "install-$mode" env -u DESTDIR -u CMAKE_INSTALL_MODE cmake --install "$output/build-$mode"
    stage=relocate-$mode
    mv -- "$output/install-$mode" "$output/relocated-$mode"
done
ln -s -- "$sdk" "$output/consumer-sdk"
for mode in static dl off; do
    options=(-DCMAKE_DISABLE_FIND_PACKAGE_XRT=TRUE)
    if [[ $mode == static ]]; then options=("-DGGML_XDNA_XRT_ROOT=$output/consumer-sdk"); fi
    run "consumer-configure-$mode" env -u XILINX_XRT -u CMAKE_PREFIX_PATH \
        cmake -S "$here/consumer" -B "$output/consumer-$mode" -G Ninja \
        "-DPACKAGE_TEST_MODE=${mode^^}" "-DPACKAGE_PREFIX=$output/relocated-$mode" \
        -DCMAKE_SKIP_RPATH=ON "${options[@]}"
    run "consumer-build-$mode" cmake --build "$output/consumer-$mode" --parallel "$jobs"
    run "consumer-elf-$mode" readelf -d "$output/consumer-$mode/package-consumer"
    if [[ $mode != static ]]; then
        if grep -E 'NEEDED.*libxrt' "$output/consumer-elf-$mode.log"; then
            echo "Unexpected direct XRT runtime dependency: $mode" >&2
            exit 1
        fi
    fi
done
run consumer-run-static env -u LD_PRELOAD LD_LIBRARY_PATH="$sdk/lib" "$output/consumer-static/package-consumer"
run consumer-run-dl env -u LD_PRELOAD LD_LIBRARY_PATH="$output/relocated-dl/lib:$sdk/lib" \
    "$output/consumer-dl/package-consumer" "$output/relocated-dl/bin/libggml-xdna.so"
run consumer-run-off env -u LD_PRELOAD -u LD_LIBRARY_PATH "$output/consumer-off/package-consumer"
for kind in producer consumer; do
    stage=missing-sdk-$kind
    options=("-DLLAMA_SOURCE=$repo" -DGGML_XDNA=ON -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF)
    if [[ $kind == consumer ]]; then
        options=(-DPACKAGE_TEST_MODE=STATIC "-DPACKAGE_PREFIX=$output/relocated-static")
    fi
    if cmake -S "$here/$kind" -B "$output/$stage" -G Ninja "${options[@]}" \
            -DCMAKE_DISABLE_FIND_PACKAGE_XRT=TRUE > "$output/$stage.log" 2>&1; then
        echo "Expected a required-XRT configure failure: $kind" >&2
        exit 1
    fi
    grep -F 'find_package for module XRT called with REQUIRED' "$output/$stage.log"
    grep -F 'CMAKE_DISABLE_FIND_PACKAGE_XRT is enabled.' "$output/$stage.log"
done
stage=complete
echo "PASS: static, dynamic-backend and XDNA-disabled installed packages; no device initialization"
