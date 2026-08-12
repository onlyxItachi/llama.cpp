# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# IRON object-FIFO plumbing adapted from Xilinx/mlir-aie's matrix-vector programming example at commit a8fea363dfd63015cffd380e0679950419a0a7cd.

"""Generate a 32-core native-BF16 GEMV for M=8192, N=1 and a compile-time K."""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorTiler2D
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile


def build_program(columns: int, kernel_symbol: str, kernel_object: str):
    rows = 8192
    group_count = 8
    workers_per_group = 4
    worker_count = group_count * workers_per_group
    rows_per_worker = 8
    rows_per_group = rows_per_worker * workers_per_group
    rows_per_wave = rows_per_worker * worker_count
    wave_count = rows // rows_per_wave
    task_group_waves = 4
    bf16_bytes = 2
    f32_bytes = 4
    stack_bytes = 0x1000
    # The pinned toolchain's placed MLIR materializes this per worker.
    toolchain_observed_iron_state_bytes = 12
    core_memory_bytes = 64 * 1024
    memtile_memory_bytes = 512 * 1024
    core_dma_word_limit = (1 << 14) - 1

    assert columns > 0 and columns % 32 == 0
    assert worker_count == 32
    assert rows_per_wave == 256
    assert wave_count == 32
    assert rows_per_worker * columns * bf16_bytes // 4 <= core_dma_word_limit
    assert (
        stack_bytes
        + rows_per_worker * columns * bf16_bytes
        + columns * bf16_bytes
        + rows_per_worker * f32_bytes
        + toolchain_observed_iron_state_bytes
        <= core_memory_bytes
    )
    assert (
        rows_per_group * columns * bf16_bytes + rows_per_group * f32_bytes
        <= memtile_memory_bytes
    )

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
            group_weight_fifos[group].cons().split(
                [worker * rows_per_worker * columns for worker in range(workers_per_group)],
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
            group_output_fifos[group].prod(depth=1).join(
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

    group_weight_taps = TensorTiler2D.simple_tiler(
        (rows, columns), (rows_per_group, columns), prune_step=False
    )
    group_output_taps = TensorTiler2D.simple_tiler(
        (1, rows), (1, rows_per_group), prune_step=False
    )
    activation_tap = TensorTiler2D.simple_tiler((1, columns), prune_step=False)[0]
    assert len(group_weight_taps) == wave_count * group_count
    assert len(group_output_taps) == wave_count * group_count

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

        for first_wave in range(0, wave_count, task_group_waves):
            wave_tasks = runtime.task_group()
            last_wave = min(wave_count, first_wave + task_group_waves)
            for wave in range(first_wave, last_wave):
                tap_base = wave * group_count
                for group in range(group_count):
                    runtime.fill(
                        group_weight_fifos[group].prod(depth=1),
                        weights,
                        group_weight_taps[tap_base + group],
                        task_group=wave_tasks,
                    )
                for group in range(group_count):
                    runtime.drain(
                        group_output_fifos[group].cons(depth=1),
                        output,
                        group_output_taps[tap_base + group],
                        task_group=wave_tasks,
                        wait=True,
                    )
            runtime.finish_task_group(wave_tasks)

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
