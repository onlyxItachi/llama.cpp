# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# IRON object-FIFO plumbing adapted from Xilinx/mlir-aie's matrix-vector
# programming example at commit a8fea363dfd63015cffd380e0679950419a0a7cd.

"""Generate the fixed native-GGML-Q4_0 288x288 AIE2P GEMV design."""

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile


def build_program():
    rows = 288
    columns = 288
    worker_count = 9
    rows_per_worker = rows // worker_count
    columns_per_block = 32
    bytes_per_block = 18

    column_blocks = columns // columns_per_block
    q4_row_bytes = column_blocks * bytes_per_block

    q4_type = np.dtype[np.uint8]
    bf16_type = np.dtype[bfloat16]
    f32_type = np.dtype[np.float32]

    weights_type = np.ndarray[(rows, q4_row_bytes), q4_type]
    activation_type = np.ndarray[(1, columns), bf16_type]
    output_type = np.ndarray[(1, rows), f32_type]
    weight_tile_type = np.ndarray[(rows_per_worker, q4_row_bytes), q4_type]
    activation_tile_type = np.ndarray[(columns,), bf16_type]
    output_tile_type = np.ndarray[(rows_per_worker,), f32_type]
    workers_per_group = [4, 4, 1]
    group_row_counts = [count * rows_per_worker for count in workers_per_group]
    group_weight_types = [
        np.ndarray[(row_count, q4_row_bytes), q4_type]
        for row_count in group_row_counts
    ]
    group_output_types = [
        np.ndarray[(row_count,), f32_type] for row_count in group_row_counts
    ]

    gemv = Kernel(
        "gemv_q4_0_bf16_f32",
        "gemv-q4_0-288.o",
        [weight_tile_type, activation_tile_type, output_tile_type],
    )

    def core_fn(weight_fifo, activation_fifo, output_fifo, gemv_kernel):
        output = output_fifo.acquire(1)
        weights = weight_fifo.acquire(1)
        activation = activation_fifo.acquire(1)
        gemv_kernel(weights, activation, output)
        weight_fifo.release(1)
        activation_fifo.release(1)
        output_fifo.release(1)

    # Depth one remains required by the pinned 1.3.4 toolchain. One shim stream
    # broadcasts the activation directly to all nine compute tiles.
    activation_fifo = ObjectFifo(
        activation_tile_type, depth=1, name="activation_bf16_broadcast"
    )
    # Three host transfers cover 4 + 4 + 1 worker slices. The two four-way
    # memory-tile splits stay within the five-output-channel tile limit, while
    # retaining native contiguous GGML rows and depth-one worker FIFOs.
    group_weight_fifos = [
        ObjectFifo(group_weight_types[group], depth=1, name=f"weights_group_{group}")
        for group in range(3)
    ]
    worker_weight_fifos = []
    group_start_worker = 0
    for group, group_workers in enumerate(workers_per_group):
        worker_weight_fifos.extend(
            group_weight_fifos[group].cons().split(
                [i * rows_per_worker * q4_row_bytes for i in range(group_workers)],
                tile=Tile(0 if group in (0, 2) else 4, 1),
                depths=[1] * group_workers,
                obj_types=[weight_tile_type] * group_workers,
                names=[
                    f"weights_rows_{i * rows_per_worker}_{(i + 1) * rows_per_worker - 1}"
                    for i in range(group_start_worker, group_start_worker + group_workers)
                ],
            )
        )
        group_start_worker += group_workers

    # Two four-way joins and one one-way join collect the same row groups. This
    # avoids the eight-input join rejected by the pinned memory-tile allocator.
    group_output_fifos = [
        ObjectFifo(group_output_types[group], depth=1, name=f"output_group_{group}")
        for group in range(3)
    ]
    worker_output_fifos = []
    group_start_worker = 0
    for group, group_workers in enumerate(workers_per_group):
        worker_output_fifos.extend(
            group_output_fifos[group].prod(depth=1).join(
                [i * rows_per_worker for i in range(group_workers)],
                tile=Tile((2, 6, 7)[group], 1),
                depths=[1] * group_workers,
                obj_types=[output_tile_type] * group_workers,
                names=[
                    f"output_rows_{i * rows_per_worker}_{(i + 1) * rows_per_worker - 1}"
                    for i in range(group_start_worker, group_start_worker + group_workers)
                ],
            )
        )
        group_start_worker += group_workers

    worker_tiles = [Tile(column, 2) for column in range(8)] + [Tile(0, 3)]
    workers = [
        Worker(
            core_fn,
            [
                worker_weight_fifos[i].cons(depth=1),
                activation_fifo.cons(depth=1),
                worker_output_fifos[i].prod(depth=1),
                gemv,
            ],
            tile=worker_tiles[i],
            stack_size=0x1000,
        )
        for i in range(worker_count)
    ]

    group_weight_taps = []
    group_output_taps = []
    row_offset = 0
    for row_count in group_row_counts:
        group_weight_taps.append(
            TensorAccessPattern(
                (rows, q4_row_bytes),
                offset=row_offset * q4_row_bytes,
                sizes=[1, 1, row_count, q4_row_bytes],
                strides=[0, 0, q4_row_bytes, 1],
            )
        )
        group_output_taps.append(
            TensorAccessPattern(
                (1, rows),
                offset=row_offset,
                sizes=[1, 1, 1, row_count],
                strides=[0, 0, 0, 1],
            )
        )
        row_offset += row_count
    activation_tap = TensorTiler2D.simple_tiler((1, columns), prune_step=False)[0]

    runtime = Runtime()
    with runtime.sequence(weights_type, activation_type, output_type) as (
        weights,
        activation,
        output,
    ):
        runtime.start(*workers)
        runtime.fill(activation_fifo.prod(depth=1), activation, activation_tap)
        for group in range(3):
            runtime.fill(
                group_weight_fifos[group].prod(depth=1),
                weights,
                group_weight_taps[group],
            )
        for group in range(3):
            runtime.drain(
                group_output_fifos[group].cons(depth=1),
                output,
                group_output_taps[group],
                wait=True,
            )

    print(Program(NPU2(), runtime).resolve_program())


if __name__ == "__main__":
    build_program()
