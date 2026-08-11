# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# IRON object-FIFO plumbing adapted from Xilinx/mlir-aie's matrix-vector
# programming example at commit a8fea363dfd63015cffd380e0679950419a0a7cd.

"""Generate the fixed 288x288 BF16 GEMV AIE2P bring-up design."""

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorTiler2D
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2


def build_program():
    rows = 288
    columns = 288
    tile_rows = 32
    tile_columns = 32

    row_tiles = rows // tile_rows
    column_tiles = columns // tile_columns

    dtype_in = np.dtype[bfloat16]
    dtype_out = np.dtype[np.float32]

    weights_type = np.ndarray[(rows, columns), dtype_in]
    activation_type = np.ndarray[(1, columns), dtype_in]
    output_type = np.ndarray[(1, rows), dtype_out]
    weight_tile_type = np.ndarray[(tile_rows, tile_columns), dtype_in]
    activation_tile_type = np.ndarray[(tile_columns,), dtype_in]
    output_tile_type = np.ndarray[(tile_rows,), dtype_out]

    zero = Kernel("zero_scalar_f32", "gemv-bf16-288.o", [output_tile_type])
    gemv = Kernel(
        "gemv_scalar_bf16_f32",
        "gemv-bf16-288.o",
        [weight_tile_type, activation_tile_type, output_tile_type],
    )

    def core_fn(weight_fifo, activation_fifo, output_fifo, zero_kernel, gemv_kernel):
        output = output_fifo.acquire(1)
        zero_kernel(output)
        for _ in range_(column_tiles):
            weights = weight_fifo.acquire(1)
            activation = activation_fifo.acquire(1)
            gemv_kernel(weights, activation, output)
            weight_fifo.release(1)
            activation_fifo.release(1)
        output_fifo.release(1)

    memory_weight_fifo = ObjectFifo(weight_tile_type, name="weights")
    compute_weight_fifo = memory_weight_fifo.cons().forward()
    activation_fifo = ObjectFifo(activation_tile_type, name="activation")
    output_fifo = ObjectFifo(output_tile_type, name="output")
    worker = Worker(
        core_fn,
        [compute_weight_fifo.cons(), activation_fifo.cons(), output_fifo.prod(), zero, gemv],
    )

    weight_taps = TensorTiler2D.group_tiler(
        (rows, columns),
        (tile_rows, tile_columns),
        (row_tiles, column_tiles),
        prune_step=False,
    )
    activation_tap = TensorTiler2D.simple_tiler(
        (1, columns), pattern_repeat=row_tiles, prune_step=False
    )[0]
    output_tap = TensorTiler2D.simple_tiler((1, rows), (1, rows), prune_step=False)[0]

    runtime = Runtime()
    with runtime.sequence(weights_type, activation_type, output_type) as (weights, activation, output):
        runtime.start(worker)
        runtime.fill(activation_fifo.prod(), activation, activation_tap)
        runtime.fill(memory_weight_fifo.prod(), weights, weight_taps[0])
        runtime.drain(output_fifo.cons(), output, output_tap, wait=True)

    print(Program(NPU2(), runtime).resolve_program())


if __name__ == "__main__":
    build_program()
