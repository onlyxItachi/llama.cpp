#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Verify the AIE2P controller transaction for the BF16 M=8192 kernels."""

import argparse
import collections
from pathlib import Path
import struct


WRITE32 = 0x00
BLOCKWRITE = 0x01
MASKWRITE32 = 0x03
TCT = 0x80
DDR_PATCH = 0x81


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_equal(actual, expected, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _u32(data: bytes, offset: int, name: str) -> int:
    _require(
        offset >= 0 and offset + 4 <= len(data), f"truncated {name} at byte {offset}"
    )
    return struct.unpack_from("<I", data, offset)[0]


def _i32(data: bytes, offset: int, name: str) -> int:
    _require(
        offset >= 0 and offset + 4 <= len(data), f"truncated {name} at byte {offset}"
    )
    return struct.unpack_from("<i", data, offset)[0]


def parse_transaction(data: bytes) -> tuple[dict[str, int], list[dict]]:
    _require(
        len(data) >= 16, "controller transaction is shorter than its 16-byte header"
    )
    header = {
        "major": data[0],
        "minor": data[1],
        "device_generation": data[2],
        "rows": data[3],
        "columns": data[4],
        "memtile_rows": data[5],
        "operation_count": _u32(data, 8, "operation count"),
        "transaction_bytes": _u32(data, 12, "transaction size"),
    }
    _require_equal(data[6:8], b"\0\0", "transaction header reserved bytes")
    _require_equal(header["transaction_bytes"], len(data), "transaction byte count")

    operations = []
    offset = 16
    while offset < len(data):
        opcode = data[offset]
        operation = {"opcode": opcode, "offset": offset}
        _require_equal(
            _u32(data, offset, "operation opcode"),
            opcode,
            f"operation opcode word at byte {offset}",
        )
        if opcode == WRITE32:
            _require(offset + 24 <= len(data), f"truncated WRITE32 at byte {offset}")
            size = _u32(data, offset + 20, "WRITE32 size")
            _require_equal(size, 24, "WRITE32 size")
            address_low = _u32(data, offset + 8, "WRITE32 address low")
            address_high = _u32(data, offset + 12, "WRITE32 address high")
            operation.update(
                reserved=_u32(data, offset + 4, "WRITE32 reserved word"),
                address=address_low | (address_high << 32),
                address_high=address_high,
                value=_u32(data, offset + 16, "WRITE32 value"),
            )
        elif opcode == BLOCKWRITE:
            _require(
                offset + 16 <= len(data),
                f"truncated BLOCKWRITE header at byte {offset}",
            )
            size = _u32(data, offset + 12, "BLOCKWRITE size")
            _require_equal(size, 48, "BLOCKWRITE size")
            _require(
                offset + size <= len(data),
                f"truncated BLOCKWRITE payload at byte {offset}",
            )
            operation.update(
                tile=_u32(data, offset + 4, "BLOCKWRITE tile"),
                address=_u32(data, offset + 8, "BLOCKWRITE address"),
                payload=list(struct.unpack_from("<8I", data, offset + 16)),
            )
        elif opcode == MASKWRITE32:
            _require(
                offset + 28 <= len(data), f"truncated MASKWRITE32 at byte {offset}"
            )
            size = _u32(data, offset + 24, "MASKWRITE32 size")
            _require_equal(size, 28, "MASKWRITE32 size")
            address_low = _u32(data, offset + 8, "MASKWRITE32 address low")
            address_high = _u32(data, offset + 12, "MASKWRITE32 address high")
            operation.update(
                reserved=_u32(data, offset + 4, "MASKWRITE32 reserved word"),
                address=address_low | (address_high << 32),
                address_high=address_high,
                value=_u32(data, offset + 16, "MASKWRITE32 value"),
                mask=_u32(data, offset + 20, "MASKWRITE32 mask"),
            )
        elif opcode == TCT:
            _require(offset + 16 <= len(data), f"truncated TCT at byte {offset}")
            size = _u32(data, offset + 4, "TCT size")
            _require_equal(size, 16, "TCT size")
            descriptor = _u32(data, offset + 8, "TCT descriptor")
            config = _u32(data, offset + 12, "TCT config")
            operation.update(
                descriptor=descriptor,
                config=config,
                column=(descriptor >> 16) & 0xFF,
                row=(descriptor >> 8) & 0xFF,
                direction=descriptor & 0xFF,
                channel=(config >> 24) & 0xFF,
                column_count=(config >> 16) & 0xFF,
                row_count=(config >> 8) & 0xFF,
            )
        elif opcode == DDR_PATCH:
            _require(offset + 48 <= len(data), f"truncated DDR_PATCH at byte {offset}")
            size = _u32(data, offset + 4, "DDR_PATCH size")
            _require_equal(size, 48, "DDR_PATCH size")
            operation.update(
                reserved=[
                    _u32(data, offset + word * 4, f"DDR_PATCH reserved word {word}")
                    for word in (2, 3, 4, 7, 9, 11)
                ],
                action=_u32(data, offset + 20, "DDR_PATCH action"),
                address=_u32(data, offset + 24, "DDR_PATCH address"),
                argument=_i32(data, offset + 32, "DDR_PATCH argument"),
                argument_plus=_u32(data, offset + 40, "DDR_PATCH argument offset"),
            )
        else:
            raise ValueError(
                f"unexpected controller opcode 0x{opcode:02x} at byte {offset}"
            )

        _require(
            offset + size <= len(data),
            f"operation at byte {offset} exceeds the transaction",
        )
        operation["size"] = size
        operations.append(operation)
        offset += size

    _require_equal(offset, len(data), "parsed transaction byte count")
    _require_equal(len(operations), header["operation_count"], "parsed operation count")
    return header, operations


def _expected_geometry(columns: int) -> dict[str, int]:
    _require(
        columns in (2048, 3072), "controller verifier supports only K=2048 and K=3072"
    )
    rows = 8192
    group_count = 8
    rows_per_group = 32
    rows_per_wave = 256
    wave_count = rows // rows_per_wave
    repeat_count = wave_count - 1
    bf16_bytes = 2
    f32_bytes = 4
    weight_group_bytes = rows_per_group * columns * bf16_bytes
    weight_wave_bytes = rows_per_wave * columns * bf16_bytes
    output_group_bytes = rows_per_group * f32_bytes
    output_wave_bytes = rows_per_wave * f32_bytes
    for name, value in (
        ("activation", columns * bf16_bytes),
        ("weight group", weight_group_bytes),
        ("weight wave", weight_wave_bytes),
        ("output group", output_group_bytes),
        ("output wave", output_wave_bytes),
    ):
        _require(value % 4 == 0, f"{name} byte count must be 4-byte aligned")
    weight_iteration_stride = weight_wave_bytes // 4 - 1
    output_iteration_stride = output_wave_bytes // 4 - 1
    _require(
        0 <= weight_iteration_stride <= (1 << 20) - 1,
        "weight iteration stride exceeds 20 bits",
    )
    _require(
        0 <= output_iteration_stride <= (1 << 20) - 1,
        "output iteration stride exceeds 20 bits",
    )
    _require(
        0 <= repeat_count <= (1 << 6) - 1,
        "repeat count exceeds the shim iteration field",
    )
    _require(repeat_count <= (1 << 8) - 1, "repeat count exceeds the task queue field")

    weight_last_exclusive = (
        repeat_count * weight_wave_bytes
        + (group_count - 1) * weight_group_bytes
        + weight_group_bytes
    )
    output_last_exclusive = (
        repeat_count * output_wave_bytes
        + (group_count - 1) * output_group_bytes
        + output_group_bytes
    )
    _require_equal(
        weight_last_exclusive, rows * columns * bf16_bytes, "weight endpoint"
    )
    _require_equal(output_last_exclusive, rows * f32_bytes, "output endpoint")
    covered_rows = [
        wave * rows_per_wave + group * rows_per_group + row
        for wave in range(wave_count)
        for group in range(group_count)
        for row in range(rows_per_group)
    ]
    _require_equal(covered_rows, list(range(rows)), "exact-once row coverage")
    return {
        "repeat_count": repeat_count,
        "weight_group_bytes": weight_group_bytes,
        "weight_iteration_stride": weight_iteration_stride,
        "output_group_bytes": output_group_bytes,
        "output_iteration_stride": output_iteration_stride,
        "weight_last_exclusive": weight_last_exclusive,
        "output_last_exclusive": output_last_exclusive,
    }


def verify_controller(data: bytes, columns: int) -> dict[str, int]:
    geometry = _expected_geometry(columns)
    header, operations = parse_transaction(data)
    expected_header = {
        "major": 0,
        "minor": 1,
        "device_generation": 4,
        "rows": 6,
        "columns": 8,
        "memtile_rows": 1,
        "operation_count": 69,
        "transaction_bytes": 2452,
    }
    _require_equal(header, expected_header, "controller header")

    expected_opcodes = [BLOCKWRITE, DDR_PATCH, MASKWRITE32, WRITE32, TCT]
    expected_opcodes += [
        opcode for _ in range(8) for opcode in (BLOCKWRITE, DDR_PATCH, WRITE32)
    ]
    expected_opcodes += [
        opcode
        for _ in range(8)
        for opcode in (BLOCKWRITE, DDR_PATCH, MASKWRITE32, WRITE32)
    ]
    expected_opcodes += [TCT] * 8
    _require_equal(
        [operation["opcode"] for operation in operations],
        expected_opcodes,
        "controller operation order",
    )
    expected_counts = {
        WRITE32: 17,
        BLOCKWRITE: 17,
        MASKWRITE32: 9,
        TCT: 9,
        DDR_PATCH: 17,
    }
    _require_equal(
        dict(collections.Counter(operation["opcode"] for operation in operations)),
        expected_counts,
        "opcode counts",
    )

    bds = [operation for operation in operations if operation["opcode"] == BLOCKWRITE]
    patches = [
        operation for operation in operations if operation["opcode"] == DDR_PATCH
    ]
    pushes = [operation for operation in operations if operation["opcode"] == WRITE32]
    masks = [
        operation for operation in operations if operation["opcode"] == MASKWRITE32
    ]
    syncs = [operation for operation in operations if operation["opcode"] == TCT]

    activation_words = columns * 2 // 4
    weight_words = geometry["weight_group_bytes"] // 4
    output_words = geometry["output_group_bytes"] // 4
    weight_offsets = [group * geometry["weight_group_bytes"] for group in range(8)]
    output_offsets = [group * geometry["output_group_bytes"] for group in range(8)]
    column_bases = [group * 0x02000000 for group in range(8)]
    activation_bd_address = 0x0601D000
    weight_bd_addresses = [base + 0x0001D000 for base in column_bases]
    output_bd_addresses = [base + 0x0001D020 for base in column_bases]
    expected_bd_addresses = (
        [activation_bd_address] + weight_bd_addresses + output_bd_addresses
    )
    expected_iterations = [0]
    expected_iterations += [
        (geometry["repeat_count"] << 20) | geometry["weight_iteration_stride"]
    ] * 8
    expected_iterations += [
        (geometry["repeat_count"] << 20) | geometry["output_iteration_stride"]
    ] * 8
    _require_equal(
        [bd["payload"][0] for bd in bds],
        [activation_words] + [weight_words] * 8 + [output_words] * 8,
        "BD lengths",
    )
    _require_equal(
        [bd["payload"][1] for bd in bds],
        [0] + weight_offsets + output_offsets,
        "BD offsets",
    )
    _require_equal(
        [bd["payload"][6] for bd in bds], expected_iterations, "BD iterations"
    )
    _require_equal([bd["tile"] for bd in bds], [0] * 17, "BLOCKWRITE tile selectors")
    _require_equal(
        [bd["address"] for bd in bds], expected_bd_addresses, "BLOCKWRITE addresses"
    )
    _require(
        all(bd["payload"][2:6] == [0, 0, 0xC0000000, 0x02000000] for bd in bds),
        "unexpected BD dimension or lock fields",
    )
    _require(
        all(bd["payload"][7] == 0x02000000 for bd in bds), "a shim BD is not valid"
    )

    _require_equal(
        [patch["argument"] for patch in patches],
        [1] + [0] * 8 + [2] * 8,
        "DDR patch arguments",
    )
    _require_equal(
        [patch["argument_plus"] for patch in patches],
        [0] + weight_offsets + output_offsets,
        "DDR patch offsets",
    )
    _require_equal(
        [patch["reserved"] for patch in patches],
        [[0] * 6 for _ in range(17)],
        "DDR patch reserved words",
    )
    _require_equal(
        [patch["action"] for patch in patches], [0] * 17, "DDR patch actions"
    )
    _require_equal(
        [patch["address"] for patch in patches],
        [address + 4 for address in expected_bd_addresses],
        "DDR patch addresses",
    )

    expected_push_values = [0x80000000] + [0x001F0000] * 8 + [0x801F0001] * 8
    _require_equal(
        [push["value"] for push in pushes], expected_push_values, "task queue values"
    )
    _require_equal(
        [(push["value"] >> 16) & 0xFF for push in pushes],
        [0] + [31] * 16,
        "task repeat counts",
    )
    _require_equal(
        [push["value"] & 0xF for push in pushes],
        [0] + [0] * 8 + [1] * 8,
        "task BD identifiers",
    )
    _require_equal(
        [push["reserved"] for push in pushes], [0] * 17, "WRITE32 reserved words"
    )
    _require_equal(
        [push["address_high"] for push in pushes],
        [0] * 17,
        "WRITE32 address high words",
    )
    expected_weight_queue_addresses = [base + 0x0001D214 for base in column_bases]
    expected_weight_queue_addresses[3] = 0x0601D21C
    expected_output_queue_addresses = [base + 0x0001D204 for base in column_bases]
    _require_equal(
        [push["address"] for push in pushes],
        [0x0601D214]
        + expected_weight_queue_addresses
        + expected_output_queue_addresses,
        "task queue addresses",
    )
    _require_equal(
        [mask["reserved"] for mask in masks], [0] * 9, "MASKWRITE32 reserved words"
    )
    _require_equal(
        [mask["address_high"] for mask in masks],
        [0] * 9,
        "MASKWRITE32 address high words",
    )
    _require_equal(
        [(mask["value"], mask["mask"]) for mask in masks],
        [(0x0F00, 0x1F00)] * 9,
        "token-controller mask writes",
    )
    expected_output_mask_addresses = [base + 0x0001D200 for base in column_bases]
    _require_equal(
        [mask["address"] for mask in masks],
        [0x0601D210] + expected_output_mask_addresses,
        "token-controller mask addresses",
    )

    expected_syncs = [(3, 0, 1, 0)] + [(column, 0, 0, 0) for column in range(8)]
    _require_equal(
        [
            (sync["column"], sync["row"], sync["direction"], sync["channel"])
            for sync in syncs
        ],
        expected_syncs,
        "token-controller syncs",
    )
    _require_equal(
        [(sync["descriptor"], sync["config"]) for sync in syncs],
        [(0x00030001, 0x00010100)]
        + [(column << 16, 0x00010100) for column in range(8)],
        "token-controller encoded fields",
    )

    weight_task_endpoints = [
        bd["payload"][1]
        + geometry["repeat_count"] * ((bd["payload"][6] & 0xFFFFF) + 1) * 4
        + bd["payload"][0] * 4
        for bd in bds[1:9]
    ]
    output_task_endpoints = [
        bd["payload"][1]
        + geometry["repeat_count"] * ((bd["payload"][6] & 0xFFFFF) + 1) * 4
        + bd["payload"][0] * 4
        for bd in bds[9:]
    ]
    _require_equal(
        weight_task_endpoints,
        [
            geometry["repeat_count"] * 256 * columns * 2
            + (group + 1) * geometry["weight_group_bytes"]
            for group in range(8)
        ],
        "weight task endpoints",
    )
    _require_equal(
        output_task_endpoints,
        [
            geometry["repeat_count"] * 256 * 4
            + (group + 1) * geometry["output_group_bytes"]
            for group in range(8)
        ],
        "output task endpoints",
    )
    _require_equal(
        weight_task_endpoints[-1],
        geometry["weight_last_exclusive"],
        "final weight task endpoint",
    )
    _require_equal(
        output_task_endpoints[-1],
        geometry["output_last_exclusive"],
        "final output task endpoint",
    )

    return {
        "columns": columns,
        "transaction_bytes": header["transaction_bytes"],
        "operation_count": header["operation_count"],
        "repeat_count": geometry["repeat_count"],
        "weight_iteration_stride": geometry["weight_iteration_stride"],
        "output_iteration_stride": geometry["output_iteration_stride"],
        "weight_last_exclusive": geometry["weight_last_exclusive"],
        "output_last_exclusive": geometry["output_last_exclusive"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("instructions", type=Path)
    args = parser.parse_args()
    try:
        summary = verify_controller(args.instructions.read_bytes(), args.columns)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        "status=PASS "
        f"columns={summary['columns']} "
        f"transaction_bytes={summary['transaction_bytes']} "
        f"operation_count={summary['operation_count']} "
        f"repeat_count={summary['repeat_count']} "
        f"weight_iteration_stride={summary['weight_iteration_stride']} "
        f"output_iteration_stride={summary['output_iteration_stride']}"
    )


if __name__ == "__main__":
    main()
