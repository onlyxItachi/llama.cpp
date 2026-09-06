#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 || $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "usage (Linux x86_64): bash $0 BUILD_BIN XRT_ROOT BF16_288_BUNDLE NEW_OUTPUT_DIR" >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$here/../../../.." && pwd -P)
bin=$(realpath -e -- "$1")
sdk=$(realpath -e -- "$2")
bundle=$(realpath -e -- "$3")
output=$(realpath -m -- "$4")
readonly here repo bin sdk bundle output
[[ ! -e "$output" && ! -L "$4" ]]
[[ -f "$bin/libggml-xdna.so" && -f "$bin/libggml-cpu.so" && -f "$sdk/lib/libxrt_coreutil.so.2" ]]
[[ -f "$sdk/include/xrt/xrt_kernel.h" && -r "$bundle" ]]
mkdir -- "$output"
stage=setup
finish() {
    local status=$?
    printf 'stage=%s exit_code=%s timing_valid=false in_flight_recovery=false\n' "$stage" "$status" > "$output/terminal.txt"
}
trap finish EXIT
printf '%s\n' 'B: exceptions only after actual device completion.' \
    'C: synthetic states only after actual device completion.' \
    'pre-submit: host-injected exception on a real unstarted NEW run; device setup is required, not a host-only test.' \
    'No real in-flight device failure or cancellation recovery is claimed.' > "$output/scope.txt"
git -C "$repo" rev-parse HEAD > "$output/source-head.txt"
readonly compiler=${CXX:-c++}
stage=build-shim
"$compiler" -std=c++17 -O2 -pthread -fPIC -shared "-I$sdk/include" \
    "$here/failure-shim.cpp" -ldl -o "$output/failure-shim.so" > "$output/build-shim.log" 2>&1
stage=build-runner
"$compiler" -std=c++17 -O2 -pthread "-I$repo/ggml/include" \
    "$here/failure-test.cpp" "-L$bin" -lggml -lggml-base -ldl \
    -o "$output/failure-test" > "$output/build-runner.log" 2>&1
stage=seal-inputs
sha256sum "$here/failure-shim.cpp" "$here/failure-test.cpp" "$here/failure-smoke.sh" \
    "$repo/ggml/src/ggml-xdna/xdna-runtime.cpp" \
    "$output/failure-shim.so" "$output/failure-test" "$bin/libggml-xdna.so" "$bin/libggml-cpu.so" \
    "$bin/libggml.so" "$bin/libggml-base.so" "$sdk/include/xrt/xrt_kernel.h" \
    "$sdk/include/xrt/detail/ert.h" "$sdk/version.json" "$bundle" > "$output/inputs.sha256"
for library in "$sdk"/lib/libxrt*.so*; do
    if [[ -f "$library" ]]; then sha256sum "$library" >> "$output/inputs.sha256"; fi
done
while IFS= read -r name; do unset "$name"; done < <(compgen -e GGML_XDNA_)
unset GGML_BACKEND_PATH LD_PRELOAD XRT_INI_PATH
export LD_LIBRARY_PATH="$sdk/lib:$bin"
export XILINX_XRT="$sdk"
export XDNA_FAILURE_XRT_LIBRARY="$sdk/lib/libxrt_coreutil.so.2"
export GGML_XDNA_AIE2P_BF16_GEMV_BUNDLE="$bundle"
export OMP_NUM_THREADS=1
ldd "$output/failure-test" > "$output/runner-libraries.txt"
ldd "$bin/libggml-xdna.so" > "$output/backend-libraries.txt"
stage=verify-xrt-resolution
resolved_coreutil=$(sed -n 's/^[[:space:]]*libxrt_coreutil\.so\.2 => \(.*\) (0x[[:xdigit:]]*)$/\1/p' "$output/backend-libraries.txt")
if [[ -z "$resolved_coreutil" || $(realpath -e -- "$resolved_coreutil") != $(realpath -e -- "$sdk/lib/libxrt_coreutil.so.2") ]]; then
    echo "Backend did not resolve libxrt_coreutil.so.2 from the selected XRT SDK" >&2
    exit 1
fi
readonly lock_path=${XDNA_CI_PHYSICAL_LOCK:-${XDG_CACHE_HOME:-$HOME/.cache}/llama-xdna/physical.lock}
mkdir -p -- "$(dirname -- "$lock_path")"
exec 8>"$lock_path"
stage=ownership
flock -n 8
[[ -c /dev/accel/accel0 ]]
if fuser /dev/accel/accel0 > "$output/device-owners.txt" 2>&1; then
    echo "NPU already has an owner; no tests executed" >&2
    exit 3
else
    owner_status=$?
    [[ "$owner_status" == 1 ]]
fi
ulimit -c 0
for mode in wait-throw start-throw wait-noresponse wait-submitted abort-noresponse \
            start-state-new start-state-submitted start-state-throw prestart-throw abort-throw; do
    expected=134
    marker=
    diagnostic='ggml_xdna: cannot establish XRT command quiescence'
    category=C
    case "$mode" in
        wait-throw)
            expected=0; category=B
            marker='failure-shim: category=B phase=wait real_state=4 injected_exception=1' ;;
        start-throw)
            expected=0; category=B
            marker='failure-shim: category=B phase=start real_state=4 injected_exception=1' ;;
        wait-noresponse) marker='failure-shim: category=C phase=wait real_state=4 synthetic_state=9' ;;
        wait-submitted) marker='failure-shim: category=C phase=wait real_state=4 synthetic_state=7' ;;
        abort-noresponse) marker='failure-shim: category=C phase=abort real_state=4 synthetic_state=9' ;;
        start-state-new) marker='failure-shim: category=C phase=state real_state=4 synthetic_state=1' ;;
        start-state-submitted) marker='failure-shim: category=C phase=state real_state=4 synthetic_state=7' ;;
        start-state-throw)
            marker='failure-shim: category=C phase=state real_state=4 injected_exception=1'
            diagnostic='ggml_xdna: cannot query command quiescence after XRT start failure' ;;
        prestart-throw)
            category=pre-submit
            marker='failure-shim: category=pre-submit phase=start actual_state=1 real_submissions=0 device_setup_required=1' ;;
        abort-throw)
            category=B
            marker='failure-shim: category=B phase=abort real_state=4 injected_exception=1'
            diagnostic='ggml_xdna: failed to quiesce an XRT command after an execution error' ;;
    esac
    trial="$output/$mode"
    mkdir -- "$trial"
    # Return the child's numeric status so timeout does not re-raise SIGABRT
    # in its own process, where the runner's no-core setting does not apply.
    command=(timeout --signal=TERM --kill-after=15s 180s bash -c '"$@"; status=$?; exit "$status"' _
             env "LD_PRELOAD=$output/failure-shim.so"
             "$output/failure-test" "$bin/libggml-xdna.so" "$bin/libggml-cpu.so" "$mode")
    printf '%q ' "${command[@]}" > "$trial/command.txt"
    printf '\n' >> "$trial/command.txt"
    stage="execute-$mode"
    set +e
    "${command[@]}" > "$trial/combined.log" 2>&1
    result=$?
    set -e
    printf 'category=%s expected_exit=%s actual_exit=%s in_flight_recovery=false\n' \
        "$category" "$expected" "$result" > "$trial/status.txt"
    stage="verify-$mode"
    [[ "$result" == "$expected" ]]
    grep -F -- "$marker" "$trial/combined.log" > "$trial/injection-receipt.txt"
    if [[ "$expected" == 0 ]]; then
        grep -F 'PASS: post-completion failure accounting and persistent retry; NOT in-flight recovery' "$trial/combined.log"
    else
        grep -F -- "$diagnostic" "$trial/combined.log" > "$trial/failstop-receipt.txt"
    fi
    printf 'PASS mode=%s category=%s expected_exit=%s\n' "$mode" "$category" "$expected"
done
stage=verify-input-identity
sha256sum -c "$output/inputs.sha256" > "$output/inputs-after.log"
stage=complete
echo 'PASS: ten attributed failure-policy cases; no real in-flight failure claim'
