#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Host-only tests for the BF16 M=8192 generator and controller verifier."""

import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_module("gemv_bf16_m8192", "gemv-bf16-m8192.py")
VERIFIER = _load_module(
    "verify_bf16_m8192_controller", "verify-bf16-m8192-controller.py"
)


def _blockwrite(address: int, payload: list[int]) -> bytes:
    operation = bytearray(48)
    operation[0] = VERIFIER.BLOCKWRITE
    struct.pack_into("<III8I", operation, 4, 0, address, 48, *payload)
    return bytes(operation)


def _patch(address: int, argument: int, argument_plus: int) -> bytes:
    operation = bytearray(48)
    operation[0] = VERIFIER.DDR_PATCH
    struct.pack_into("<I", operation, 4, 48)
    struct.pack_into("<II", operation, 20, 0, address)
    struct.pack_into("<i", operation, 32, argument)
    struct.pack_into("<i", operation, 40, argument_plus)
    return bytes(operation)


def _maskwrite(address: int) -> bytes:
    operation = bytearray(28)
    operation[0] = VERIFIER.MASKWRITE32
    struct.pack_into("<IIIII", operation, 8, address, 0, 0x0F00, 0x1F00, 28)
    return bytes(operation)


def _write32(address: int, value: int) -> bytes:
    operation = bytearray(24)
    operation[0] = VERIFIER.WRITE32
    struct.pack_into("<IIII", operation, 8, address, 0, value, 24)
    return bytes(operation)


def _tct(column: int, direction: int) -> bytes:
    operation = bytearray(16)
    operation[0] = VERIFIER.TCT
    descriptor = (column << 16) | direction
    config = (1 << 16) | (1 << 8)
    struct.pack_into("<III", operation, 4, 16, descriptor, config)
    return bytes(operation)


def _controller_transaction(columns: int) -> bytes:
    activation_words = columns * 2 // 4
    weight_group_bytes = 32 * columns * 2
    output_group_bytes = 32 * 4
    weight_iteration = (31 << 20) | (256 * columns * 2 // 4 - 1)
    output_iteration = (31 << 20) | (256 * 4 // 4 - 1)
    fixed_bd_fields = [0, 0, 0xC0000000, 0x02000000]
    valid = 0x02000000
    operations = []
    column_bases = [group * 0x02000000 for group in range(8)]

    activation_address = 0x0601D000
    operations.extend(
        (
            _blockwrite(
                activation_address, [activation_words, 0, *fixed_bd_fields, 0, valid]
            ),
            _patch(activation_address + 4, 1, 0),
            _maskwrite(activation_address + 528),
            _write32(activation_address + 532, 0x80000000),
            _tct(3, 1),
        )
    )
    for group in range(8):
        address = column_bases[group] + 0x0001D000
        offset = group * weight_group_bytes
        operations.extend(
            (
                _blockwrite(
                    address,
                    [
                        weight_group_bytes // 4,
                        offset,
                        *fixed_bd_fields,
                        weight_iteration,
                        valid,
                    ],
                ),
                _patch(address + 4, 0, offset),
                _write32(
                    column_bases[group] + (0x0001D21C if group == 3 else 0x0001D214),
                    0x001F0000,
                ),
            )
        )
    for group in range(8):
        address = column_bases[group] + 0x0001D020
        offset = group * output_group_bytes
        operations.extend(
            (
                _blockwrite(
                    address,
                    [
                        output_group_bytes // 4,
                        offset,
                        *fixed_bd_fields,
                        output_iteration,
                        valid,
                    ],
                ),
                _patch(address + 4, 2, offset),
                _maskwrite(column_bases[group] + 0x0001D200),
                _write32(column_bases[group] + 0x0001D204, 0x801F0001),
            )
        )
    operations.extend(_tct(column, 0) for column in range(8))

    body = b"".join(operations)
    header = bytearray(16)
    header[0:6] = bytes((0, 1, 4, 6, 8, 1))
    struct.pack_into("<II", header, 8, len(operations), len(header) + len(body))
    return bytes(header) + body


def _mutate_u32(data: bytes, offset: int, value: int = 1) -> bytes:
    mutated = bytearray(data)
    struct.pack_into("<I", mutated, offset, value)
    return bytes(mutated)


def _relocate_registers(data: bytes, delta: int) -> bytes:
    _, operations = VERIFIER.parse_transaction(data)
    mutated = bytearray(data)
    address_offsets = {
        VERIFIER.BLOCKWRITE: 8,
        VERIFIER.DDR_PATCH: 24,
        VERIFIER.WRITE32: 8,
        VERIFIER.MASKWRITE32: 8,
    }
    for operation in operations:
        field_offset = address_offsets.get(operation["opcode"])
        if field_offset is not None:
            absolute_offset = operation["offset"] + field_offset
            value = struct.unpack_from("<I", mutated, absolute_offset)[0]
            struct.pack_into("<I", mutated, absolute_offset, value + delta)
    return bytes(mutated)


def _invalid_transactions(data: bytes) -> dict[str, bytes]:
    _, operations = VERIFIER.parse_transaction(data)
    bds = [
        operation
        for operation in operations
        if operation["opcode"] == VERIFIER.BLOCKWRITE
    ]
    patches = [
        operation
        for operation in operations
        if operation["opcode"] == VERIFIER.DDR_PATCH
    ]
    pushes = [
        operation for operation in operations if operation["opcode"] == VERIFIER.WRITE32
    ]
    masks = [
        operation
        for operation in operations
        if operation["opcode"] == VERIFIER.MASKWRITE32
    ]
    syncs = [
        operation for operation in operations if operation["opcode"] == VERIFIER.TCT
    ]

    header_reserved = bytearray(data)
    header_reserved[6] = 1
    opcode_upper = bytearray(data)
    opcode_upper[bds[0]["offset"] + 1] = 1
    endpoint = bytearray(data)
    endpoint_offset = bds[-1]["offset"] + 20
    struct.pack_into(
        "<I",
        endpoint,
        endpoint_offset,
        struct.unpack_from("<I", endpoint, endpoint_offset)[0] + 4,
    )

    invalid = {
        "header-reserved": bytes(header_reserved),
        "opcode-upper": bytes(opcode_upper),
        "blockwrite-tile": _mutate_u32(data, bds[0]["offset"] + 4),
        "patch-action": _mutate_u32(data, patches[0]["offset"] + 20),
        "write-reserved": _mutate_u32(data, pushes[0]["offset"] + 4),
        "write-address-high": _mutate_u32(data, pushes[0]["offset"] + 12),
        "mask-reserved": _mutate_u32(data, masks[0]["offset"] + 4),
        "mask-address-high": _mutate_u32(data, masks[0]["offset"] + 12),
        "tct-unused-config": _mutate_u32(data, syncs[0]["offset"] + 12, 0x00010101),
        "relocated-register-map": _relocate_registers(data, 0x1000),
        "final-output-endpoint": bytes(endpoint),
    }
    for word in (2, 3, 4, 7, 9, 11):
        invalid[f"patch-reserved-{word}"] = _mutate_u32(
            data, patches[0]["offset"] + word * 4
        )
    return invalid


class GeometryTests(unittest.TestCase):
    def test_registered_shapes(self):
        expected = {
            2048: (262143, 33554432),
            3072: (393215, 50331648),
        }
        for columns, (weight_stride, endpoint) in expected.items():
            with self.subTest(columns=columns):
                geometry = GENERATOR.validate_geometry(columns)
                self.assertEqual(geometry["wave_count"], 32)
                self.assertEqual(geometry["repeat_count"], 31)
                self.assertEqual(geometry["weight_iteration_stride"], weight_stride)
                self.assertEqual(geometry["output_iteration_stride"], 255)
                self.assertEqual(geometry["weight_last_exclusive"], endpoint)
                self.assertEqual(geometry["output_last_exclusive"], 32768)

    def test_invalid_geometry_fails_closed(self):
        for columns in (True, 0, 31, 4096):
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    GENERATOR.validate_geometry(columns)
        with self.assertRaisesRegex(ValueError, "4-byte aligned"):
            GENERATOR._encoded_iteration_stride(1025, "test")
        with self.assertRaisesRegex(ValueError, "20-bit"):
            GENERATOR._encoded_iteration_stride(4 * ((1 << 20) + 1), "test")
        with self.assertRaisesRegex(ValueError, "positive"):
            GENERATOR._repeat_count(0)
        with self.assertRaisesRegex(ValueError, "6-bit"):
            GENERATOR._repeat_count(65)


class ControllerVerifierTests(unittest.TestCase):
    def test_registered_transactions(self):
        for columns, stride, endpoint in (
            (2048, 262143, 33554432),
            (3072, 393215, 50331648),
        ):
            with self.subTest(columns=columns):
                summary = VERIFIER.verify_controller(
                    _controller_transaction(columns), columns
                )
                self.assertEqual(summary["operation_count"], 69)
                self.assertEqual(summary["transaction_bytes"], 2452)
                self.assertEqual(summary["weight_iteration_stride"], stride)
                self.assertEqual(summary["weight_last_exclusive"], endpoint)

    def test_truncated_and_mutated_transactions_fail(self):
        data = _controller_transaction(2048)
        with self.assertRaisesRegex(ValueError, "transaction byte count"):
            VERIFIER.verify_controller(data[:-4], 2048)

        mutated = bytearray(data)
        _, operations = VERIFIER.parse_transaction(data)
        pushes = [
            operation
            for operation in operations
            if operation["opcode"] == VERIFIER.WRITE32
        ]
        struct.pack_into("<I", mutated, pushes[1]["offset"] + 16, 0x001E0000)
        with self.assertRaisesRegex(ValueError, "task queue values"):
            VERIFIER.verify_controller(bytes(mutated), 2048)

    def test_all_encoded_fields_and_endpoints_fail_closed(self):
        data = _controller_transaction(2048)
        for name, mutated in _invalid_transactions(data).items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    VERIFIER.verify_controller(mutated, 2048)

    def test_cli_is_fail_closed_under_python_optimize(self):
        valid = _controller_transaction(3072)
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.bin"
            valid_path.write_bytes(valid)
            valid_run = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    str(HERE / "verify-bf16-m8192-controller.py"),
                    "--columns",
                    "3072",
                    str(valid_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            invalid_runs = []
            for name, mutated in _invalid_transactions(valid).items():
                invalid_path = Path(directory) / f"{name}.bin"
                invalid_path.write_bytes(mutated)
                invalid_runs.append(
                    (
                        name,
                        subprocess.run(
                            [
                                sys.executable,
                                "-O",
                                str(HERE / "verify-bf16-m8192-controller.py"),
                                "--columns",
                                "3072",
                                str(invalid_path),
                            ],
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        ),
                    )
                )
        self.assertEqual(valid_run.returncode, 0, valid_run.stderr)
        self.assertIn("status=PASS", valid_run.stdout)
        for name, invalid_run in invalid_runs:
            with self.subTest(name=name):
                self.assertNotEqual(invalid_run.returncode, 0)
                self.assertIn("error:", invalid_run.stderr)


if __name__ == "__main__":
    unittest.main()
