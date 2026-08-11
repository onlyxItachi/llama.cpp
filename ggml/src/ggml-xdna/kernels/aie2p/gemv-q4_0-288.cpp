// SPDX-License-Identifier: MIT

// Per-worker vector kernel for native GGML block_q4_0 bytes. The IRON design
// instantiates this 32-row primitive across the selected worker topology.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

__aie_inline float fp16_to_fp32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t unbiased_exponent = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1u;
                --unbiased_exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                   (static_cast<uint32_t>(unbiased_exponent + 127) << 23u) |
                   (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((exponent + (127u - 15u)) << 23u) | (mantissa << 13u);
    }

    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

__aie_inline aie::vector<int8, 32> unpack_q4_0(const uint8_t * packed, bool safe_tail) {
    // The pointer is two-byte aligned inside an 18-byte block.  This load
    // consumes 16 packed bytes while allowing that alignment. Peano may issue
    // wider surrounding loads, so the very last block uses an exact byte load
    // to avoid reading past the object-FIFO allocation.
    aie::vector<uint4, 32> q4;
    if (safe_tail) {
        alignas(32) uint8_t scratch[32] = {};
        const volatile uint8_t * exact = packed;
        for (unsigned int i = 0; i < 16; ++i) {
            scratch[i] = exact[i];
        }
        q4 = aie::load_v<32>(reinterpret_cast<const uint4 *>(scratch));
    } else {
        q4 = aie::load_unaligned_v<32>(reinterpret_cast<const uint4 *>(packed), 4);
    }
    const auto q8_interleaved = q4.template unpack_sign<int8>(false);

    // uint4 unpack produces [low0, high0, low1, high1, ...]. GGML's logical
    // block order is low[0..15], high[0..15], so restore that order and keep
    // the existing natural activation layout unchanged.
    const auto q8_natural = aie::concat(
        aie::filter_even(q8_interleaved, 1),
        aie::filter_odd(q8_interleaved, 1));
    return aie::sub(q8_natural, int8_t(8));
}

__aie_inline aie::vector<bfloat16, 32> dequantize_block(
        const uint8_t * block, bool safe_tail) {
    const auto q_bf16 = aie::to_float<bfloat16>(unpack_q4_0(block + 2, safe_tail));
    const uint16_t scale_bits = static_cast<uint16_t>(block[0]) |
                                (static_cast<uint16_t>(block[1]) << 8u);
    // AIE2P/arch21 has no native IEEE-fp16 arithmetic. Decode the GGML scale
    // exactly, then round it to BF16 so scaling and accumulation remain on the
    // vector datapath instead of calling a scalar soft-float multiply helper.
    const bfloat16 scale = static_cast<bfloat16>(fp16_to_fp32(scale_bits));
    return aie::mul(
        q_bf16, aie::broadcast<bfloat16, 32>(scale)).template to_vector<bfloat16>();
}

__aie_inline void dot_rows_3(
        const uint8_t * weights, const bfloat16 * activation, float * output) {
    auto acc0 = aie::zeros<accfloat, 32>();
    auto acc1 = aie::zeros<accfloat, 32>();
    auto acc2 = aie::zeros<accfloat, 32>();
    for (int block_index = 0; block_index < 9; ++block_index) {
        const auto x = aie::load_v<32>(activation + block_index * 32);
        acc0 = aie::mac(acc0, dequantize_block(weights + block_index * 18, false), x);
        acc1 = aie::mac(acc1, dequantize_block(weights + 162 + block_index * 18, false), x);
        acc2 = aie::mac(acc2, dequantize_block(weights + 324 + block_index * 18, false), x);
    }
    output[0] = aie::reduce_add<float>(acc0);
    output[1] = aie::reduce_add<float>(acc1);
    output[2] = aie::reduce_add<float>(acc2);
}

__aie_inline void dot_final_rows_2(
        const uint8_t * weights, const bfloat16 * activation, float * output) {
    auto acc0 = aie::zeros<accfloat, 32>();
    auto acc1 = aie::zeros<accfloat, 32>();
    for (int block_index = 0; block_index < 9; ++block_index) {
        const auto x = aie::load_v<32>(activation + block_index * 32);
        acc0 = aie::mac(acc0, dequantize_block(weights + block_index * 18, false), x);
        acc1 = aie::mac(
            acc1,
            dequantize_block(weights + 162 + block_index * 18, block_index == 8),
            x);
    }
    output[0] = aie::reduce_add<float>(acc0);
    output[1] = aie::reduce_add<float>(acc1);
}

} // namespace

extern "C" {

// Each input object contains 32 complete GGML rows. A row holds nine
// block_q4_0 values; each block is an LE fp16 scale plus 16 packed bytes.
void gemv_q4_0_bf16_f32(const uint8_t * weights, const bfloat16 * activation, float * output) {
    // Three independent row accumulators reuse each activation vector load.
    // Four rows spill vector state with the pinned Peano toolchain; the final
    // two rows also provide the exact-load boundary case for the FIFO object.
    for (int row = 0; row < 30; row += 3) {
        dot_rows_3(weights + row * 162, activation, output + row);
    }
    dot_final_rows_2(weights + 30 * 162, activation, output + 30);
}

}
