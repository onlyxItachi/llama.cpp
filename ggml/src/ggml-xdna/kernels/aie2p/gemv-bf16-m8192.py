# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# IRON object-FIFO plumbing adapted from Xilinx/mlir-aie's matrix-vector programming example at commit a8fea363dfd63015cffd380e0679950419a0a7cd.

"""Generate a 32-core native-BF16 GEMV for M=8192, N=1 and a compile-time K."""

import argparse


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _encoded_iteration_stride(byte_stride: int, name: str) -> int:
    _require(byte_stride > 0, f"{name} byte stride must be positive")
    _require(byte_stride % 4 == 0, f"{name} byte stride must be 4-byte aligned")
    encoded_stride = byte_stride // 4 - 1
    _require(
        encoded_stride <= (1 << 20) - 1,
        f"{name} encoded word stride exceeds the 20-bit shim field",
    )
    return encoded_stride


def _repeat_count(wave_count: int) -> int:
    _require(wave_count > 0, "wave count must be positive")
    repeat_count = wave_count - 1
    _require(
        repeat_count <= (1 << 6) - 1,
        "wave repeat exceeds the 6-bit shim iteration field",
    )
    _require(
        repeat_count <= (1 << 8) - 1, "wave repeat exceeds the 8-bit task queue field"
    )
    return repeat_count


def validate_geometry(columns: int) -> dict[str, int]:
    rows = 8192
    group_count = 8
    workers_per_group = 4
    worker_count = group_count * workers_per_group
    rows_per_worker = 8
    rows_per_group = rows_per_worker * workers_per_group
    rows_per_wave = rows_per_worker * worker_count
    bf16_bytes = 2
    f32_bytes = 4
    stack_bytes = 0x1000
    # The pinned toolchain's placed MLIR materializes this per worker.
    toolchain_observed_iron_state_bytes = 12
    core_memory_bytes = 64 * 1024
    memtile_memory_bytes = 512 * 1024
    core_dma_word_limit = (1 << 14) - 1

    _require(
        isinstance(columns, int) and not isinstance(columns, bool),
        "columns must be an integer",
    )
    _require(
        columns > 0 and columns % 32 == 0, "columns must be a positive multiple of 32"
    )
    _require(worker_count == 32, "the AIE2P topology requires 32 workers")
    _require(rows_per_wave == 256, "the AIE2P topology requires 256 rows per wave")
    _require(rows % rows_per_wave == 0, "rows must contain an integer number of waves")

    wave_count = rows // rows_per_wave
    repeat_count = _repeat_count(wave_count)
    _require(
        wave_count == 32 and repeat_count == 31,
        "M=8192 must map to 32 waves and 31 repeats",
    )

    worker_weight_bytes = rows_per_worker * columns * bf16_bytes
    group_weight_bytes = rows_per_group * columns * bf16_bytes
    weight_wave_bytes = rows_per_wave * columns * bf16_bytes
    output_group_bytes = rows_per_group * f32_bytes
    output_wave_bytes = rows_per_wave * f32_bytes
    activation_bytes = columns * bf16_bytes
    for name, byte_count in (
        ("worker weight transfer", worker_weight_bytes),
        ("group weight transfer", group_weight_bytes),
        ("weight wave stride", weight_wave_bytes),
        ("output group transfer", output_group_bytes),
        ("output wave stride", output_wave_bytes),
        ("activation transfer", activation_bytes),
    ):
        _require(byte_count % 4 == 0, f"{name} must be 4-byte aligned")

    weight_iteration_stride = _encoded_iteration_stride(
        weight_wave_bytes, "weight wave"
    )
    output_iteration_stride = _encoded_iteration_stride(
        output_wave_bytes, "output wave"
    )
    _require(
        worker_weight_bytes // 4 <= core_dma_word_limit,
        "worker weight transfer exceeds the core DMA word limit",
    )
    _require(
        stack_bytes
        + worker_weight_bytes
        + activation_bytes
        + rows_per_worker * f32_bytes
        + toolchain_observed_iron_state_bytes
        <= core_memory_bytes,
        "worker buffers exceed AIE2P core memory",
    )
    _require(
        group_weight_bytes + output_group_bytes <= memtile_memory_bytes,
        "group buffers exceed AIE2P memtile memory",
    )

    weight_last_exclusive = (
        (wave_count - 1) * weight_wave_bytes
        + (group_count - 1) * group_weight_bytes
        + group_weight_bytes
    )
    output_last_exclusive = (
        (wave_count - 1) * output_wave_bytes
        + (group_count - 1) * output_group_bytes
        + output_group_bytes
    )
    _require(
        weight_last_exclusive == rows * activation_bytes,
        "weight taps do not end at the matrix boundary",
    )
    _require(
        output_last_exclusive == rows * f32_bytes,
        "output taps do not end at the vector boundary",
    )

    covered_rows = [
        wave * rows_per_wave + group * rows_per_group + row
        for wave in range(wave_count)
        for group in range(group_count)
        for row in range(rows_per_group)
    ]
    _require(
        covered_rows == list(range(rows)),
        "strided taps must cover each output row exactly once",
    )

    return {
        "rows": rows,
        "group_count": group_count,
        "workers_per_group": workers_per_group,
        "worker_count": worker_count,
        "rows_per_worker": rows_per_worker,
        "rows_per_group": rows_per_group,
        "rows_per_wave": rows_per_wave,
        "wave_count": wave_count,
        "repeat_count": repeat_count,
        "stack_bytes": stack_bytes,
        "weight_iteration_stride": weight_iteration_stride,
        "output_iteration_stride": output_iteration_stride,
        "weight_last_exclusive": weight_last_exclusive,
        "output_last_exclusive": output_last_exclusive,
    }


def build_program(columns: int, kernel_symbol: str, kernel_object: str):
    geometry = validate_geometry(columns)
    rows = geometry["rows"]
    group_count = geometry["group_count"]
    workers_per_group = geometry["workers_per_group"]
    worker_count = geometry["worker_count"]
    rows_per_worker = geometry["rows_per_worker"]
    rows_per_group = geometry["rows_per_group"]
    rows_per_wave = geometry["rows_per_wave"]
    wave_count = geometry["wave_count"]
    stack_bytes = geometry["stack_bytes"]

    import numpy as np
    from ml_dtypes import bfloat16

    from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2, Tile

    bf16_type = np.dtype[bfloat16]
    f32_type = np.dtype[np.float32]
    weights_type = np.ndarray[(rows, columns), bf16_type]
    activation_type = np.ndarray[(1, columns), bf16_type]
    output_type = np.ndarray[(1, rows), f32_type]
    group_weight_type = np.ndarray[(rows_per_group, columns), bf16_type]
    worker_weight_type = np.ndarray[(rows_per_worker, columns), bf16_type]
    activation_tile_type = np.ndarray[(columns,), bf16_type]
    group_output_type = np.ndarray[(rows_per_group,), f32_type]
    worker_output_type = np.ndarray[(rows_per_worker,), f32_type]

    gemv = Kernel(
        kernel_symbol,
        kernel_object,
        [worker_weight_type, activation_tile_type, worker_output_type],
    )

    def core_fn(weight_fifo, activation_fifo, output_fifo, gemv_kernel):
        activation = activation_fifo.acquire(1)
        for _ in range_(wave_count):
            output = output_fifo.acquire(1)
            weights = weight_fifo.acquire(1)
            gemv_kernel(weights, activation, output)
            weight_fifo.release(1)
            output_fifo.release(1)
        activation_fifo.release(1)

    activation_fifo = ObjectFifo(
        activation_tile_type, depth=1, name="activation_bf16_broadcast"
    )
    group_weight_fifos = [
        ObjectFifo(group_weight_type, depth=1, name=f"weights_group_{group}")
        for group in range(group_count)
    ]
    worker_weight_fifos = []
    for group in range(group_count):
        worker_weight_fifos.extend(
            group_weight_fifos[group]
            .cons()
            .split(
                [
                    worker * rows_per_worker * columns
                    for worker in range(workers_per_group)
                ],
                tile=Tile(group, 1),
                depths=[1] * workers_per_group,
                obj_types=[worker_weight_type] * workers_per_group,
                names=[
                    f"weights_group_{group}_worker_{worker}"
                    for worker in range(workers_per_group)
                ],
            )
        )

    group_output_fifos = [
        ObjectFifo(group_output_type, depth=1, name=f"output_group_{group}")
        for group in range(group_count)
    ]
    worker_output_fifos = []
    for group in range(group_count):
        worker_output_fifos.extend(
            group_output_fifos[group]
            .prod(depth=1)
            .join(
                [worker * rows_per_worker for worker in range(workers_per_group)],
                tile=Tile(group, 1),
                depths=[1] * workers_per_group,
                obj_types=[worker_output_type] * workers_per_group,
                names=[
                    f"output_group_{group}_worker_{worker}"
                    for worker in range(workers_per_group)
                ],
            )
        )

    worker_tiles = [
        Tile(column, row)
        for column in range(group_count)
        for row in range(2, 2 + workers_per_group)
    ]
    workers = [
        Worker(
            core_fn,
            [
                worker_weight_fifos[worker].cons(depth=1),
                activation_fifo.cons(depth=1),
                worker_output_fifos[worker].prod(depth=1),
                gemv,
            ],
            tile=worker_tiles[worker],
            stack_size=stack_bytes,
        )
        for worker in range(worker_count)
    ]

    # The outer dimension lowers to 31 task repeats, one per 256-row wave.
    group_weight_taps = [
        TensorAccessPattern(
            (rows, columns),
            offset=group * rows_per_group * columns,
            sizes=[wave_count, 1, rows_per_group, columns],
            strides=[rows_per_wave * columns, 0, columns, 1],
        )
        for group in range(group_count)
    ]
    group_output_taps = [
        TensorAccessPattern(
            (1, rows),
            offset=group * rows_per_group,
            sizes=[wave_count, 1, 1, rows_per_group],
            strides=[rows_per_wave, 0, 0, 1],
        )
        for group in range(group_count)
    ]
    activation_tap = TensorTiler2D.simple_tiler((1, columns), prune_step=False)[0]

    runtime = Runtime()
    with runtime.sequence(weights_type, activation_type, output_type) as (
        weights,
        activation,
        output,
    ):
        runtime.start(*workers)

        activation_tasks = runtime.task_group()
        runtime.fill(
            activation_fifo.prod(depth=1),
            activation,
            activation_tap,
            task_group=activation_tasks,
            wait=True,
        )
        runtime.finish_task_group(activation_tasks)

        matrix_tasks = runtime.task_group()
        for group in range(group_count):
            runtime.fill(
                group_weight_fifos[group].prod(depth=1),
                weights,
                group_weight_taps[group],
                task_group=matrix_tasks,
            )
        for group in range(group_count):
            runtime.drain(
                group_output_fifos[group].cons(depth=1),
                output,
                group_output_taps[group],
                task_group=matrix_tasks,
                wait=True,
            )
        runtime.finish_task_group(matrix_tasks)

    print(Program(NPU2(), runtime).resolve_program())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", required=True, type=int)
    parser.add_argument("--kernel-symbol")
    parser.add_argument("--kernel-object")
    args = parser.parse_args()
    if args.kernel_symbol is None:
        args.kernel_symbol = f"gemv_bf16_f32_m8_k{args.columns}"
    if args.kernel_object is None:
        args.kernel_object = f"gemv-bf16-m8192-k{args.columns}.o"
    return args


if __name__ == "__main__":
    arguments = parse_args()
    build_program(arguments.columns, arguments.kernel_symbol, arguments.kernel_object)
