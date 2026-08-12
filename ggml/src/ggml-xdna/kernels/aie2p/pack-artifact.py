#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Couple one ggml-xdna kernel descriptor, xclbin, and instruction stream."""

import argparse
import hashlib
import os
from pathlib import Path
import struct
import tempfile


MAGIC = b"GGXDNA1\0"
HEADER_BYTES = 64
ARCHITECTURE_IDS = {
    "aie2": 1,
    "aie2p": 2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abi-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--architecture", choices=ARCHITECTURE_IDS)
    parser.add_argument("--kind-id", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--weight-bytes", type=int, required=True)
    parser.add_argument("--activation-bytes", type=int, required=True)
    parser.add_argument("--output-bytes", type=int, required=True)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.abi_version == 1:
        if args.architecture not in (None, "aie2p"):
            parser.error("ABI v1 is valid only for AIE2P artifacts")
        architecture_id = 0
    else:
        if args.architecture is None:
            parser.error("--architecture is required for ABI v2")
        architecture_id = ARCHITECTURE_IDS[args.architecture]

    xclbin = args.xclbin.read_bytes()
    instructions = args.instructions.read_bytes()
    if not xclbin:
        raise ValueError("xclbin is empty")
    if not instructions or len(instructions) % 4:
        raise ValueError("instruction stream must be nonempty 32-bit words")

    positive_metadata = {
        "kind id": args.kind_id,
        "M": args.m,
        "K": args.k,
        "weight bytes": args.weight_bytes,
        "activation bytes": args.activation_bytes,
        "output bytes": args.output_bytes,
    }
    for name, value in positive_metadata.items():
        if value <= 0 or value > 0xFFFFFFFF:
            raise ValueError(f"{name} must fit a nonzero ABI-v{args.abi_version} uint32")

    header = struct.pack(
        "<8s8I2Q2I",
        MAGIC,
        HEADER_BYTES,
        args.abi_version,
        args.kind_id,
        args.m,
        args.k,
        args.weight_bytes,
        args.activation_bytes,
        args.output_bytes,
        len(xclbin),
        len(instructions),
        architecture_id,
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
