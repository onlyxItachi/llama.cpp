#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Couple one fixed ggml-xdna kernel ABI, xclbin, and instruction stream."""

import argparse
import hashlib
import os
from pathlib import Path
import struct
import tempfile


MAGIC = b"GGXDNA1\0"
HEADER_BYTES = 64
ABI_VERSION = 1
KINDS = {
    "bf16": (1, 288 * 288 * 2),
    "q4_0": (2, 288 * (288 // 32) * 18),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    xclbin = args.xclbin.read_bytes()
    instructions = args.instructions.read_bytes()
    if not xclbin:
        raise ValueError("xclbin is empty")
    if not instructions or len(instructions) % 4:
        raise ValueError("instruction stream must be nonempty 32-bit words")

    kind, weight_bytes = KINDS[args.kind]
    header = struct.pack(
        "<8s8I3Q",
        MAGIC,
        HEADER_BYTES,
        ABI_VERSION,
        kind,
        288,
        288,
        weight_bytes,
        288 * 2,
        288 * 4,
        len(xclbin),
        len(instructions),
        0,
    )
    assert len(header) == HEADER_BYTES
    bundle = header + xclbin + instructions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=args.output.parent, prefix=f".{args.output.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(bundle)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, args.output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"{args.output}: sha256={hashlib.sha256(bundle).hexdigest()}")


if __name__ == "__main__":
    main()
