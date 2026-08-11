# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# IRON object-FIFO plumbing adapted from Xilinx/mlir-aie's matrix-vector
# programming example at commit a8fea363dfd63015cffd380e0679950419a0a7cd.

"""Generate the fixed native-GGML-Q4_0 288x288 AIE2P GEMV design."""

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2


def build_program():
    rows = 288
    columns = 288
    rows_per_tile = 32
    columns_per_block = 32
    bytes_per_block = 18

    row_tiles = rows // rows_per_tile
    column_blocks = columns // columns_per_block
    q4_row_bytes = column_blocks * bytes_per_block

    q4_type = np.dtype[np.uint8]
    bf16_type = np.dtype[bfloat16]
    f32_type = np.dtype[np.float32]

    weights_type = np.ndarray[(rows, q4_row_bytes), q4_type]
    activation_type = np.ndarray[(1, columns), bf16_type]
    output_type = np.ndarray[(1, rows), f32_type]
    weight_tile_type = np.ndarray[(rows_per_tile, q4_row_bytes), q4_type]
    activation_tile_type = np.ndarray[(columns,), bf16_type]
    output_tile_type = np.ndarray[(rows_per_tile,), f32_type]

    zero = Kernel("zero_scalar_f32", "gemv-q4_0-288.o", [output_tile_type])
    gemv = Kernel(
        "gemv_q4_0_bf16_f32",
        "gemv-q4_0-288.o",
        [weight_tile_type, activation_tile_type, output_tile_type],
    )

    def core_fn(weight_fifo, activation_fifo, output_fifo, zero_kernel, gemv_kernel):
        output = output_fifo.acquire(1)
        weights = weight_fifo.acquire(1)
        activation = activation_fifo.acquire(1)
        zero_kernel(output)
        gemv_kernel(weights, activation, output)
        weight_fifo.release(1)
        activation_fifo.release(1)
        output_fifo.release(1)

    # Depth one is required by the pinned 1.3.4 toolchain. A depth-two
    # forwarding FIFO corrupted the first scale of row 96 on even launches.
    memory_weight_fifo = ObjectFifo(weight_tile_type, depth=1, name="weights_q4_0")
    compute_weight_fifo = memory_weight_fifo.cons().forward(depth=1)
    activation_fifo = ObjectFifo(activation_tile_type, depth=1, name="activation_bf16")
    output_fifo = ObjectFifo(output_tile_type, depth=1, name="output_f32")
    worker = Worker(
        core_fn,
        [compute_weight_fifo.cons(), activation_fifo.cons(), output_fifo.prod(), zero, gemv],
        # Scalar FP32 helper calls need more than IRON's default 1 KiB stack.
        stack_size=0x1000,
    )

    # One native GGML row is 162 bytes, while contiguous AIE2P DMA lengths must
    # be divisible by four. Pairing rows gives exact 324-byte transfers: no
    # padding, over-read, reorder, or dequantized matrix is introduced.
    weight_tap = TensorAccessPattern(
        (rows, q4_row_bytes),
        offset=0,
        sizes=[row_tiles, rows_per_tile // 2, 2 * q4_row_bytes],
        strides=[rows_per_tile * q4_row_bytes, 2 * q4_row_bytes, 1],
    )
    activation_tap = TensorTiler2D.simple_tiler(
        (1, columns), pattern_repeat=row_tiles, prune_step=False
    )[0]
    output_tap = TensorTiler2D.simple_tiler((1, rows), (1, rows), prune_step=False)[0]

    runtime = Runtime()
    with runtime.sequence(weights_type, activation_type, output_type) as (
        weights,
        activation,
        output,
    ):
        runtime.start(worker)
        runtime.fill(activation_fifo.prod(), activation, activation_tap)
        runtime.fill(memory_weight_fifo.prod(), weights, weight_tap)
        runtime.drain(output_fifo.cons(), output, output_tap, wait=True)

    print(Program(NPU2(), runtime).resolve_program())


if __name__ == "__main__":
    build_program()
