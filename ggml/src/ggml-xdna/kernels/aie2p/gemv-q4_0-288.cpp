// SPDX-License-Identifier: MIT

// Single-worker vector kernel for native GGML block_q4_0 bytes. Multi-tile row
// distribution is deliberately kept in the IRON design rather than this
// shape-specific compute primitive.

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

__aie_inline float dot_row(const uint8_t * weights, const bfloat16 * activation, bool final_row) {
    float sum = 0.0f;
    for (int block_index = 0; block_index < 9; ++block_index) {
        const uint8_t * block = weights + block_index * 18;
        const auto q_bf16 = aie::to_float<bfloat16>(
            unpack_q4_0(block + 2, final_row && block_index == 8));
        const auto x = aie::load_v<32>(activation + block_index * 32);
        const float dot = aie::reduce_add<float>(aie::mul(q_bf16, x));

        const uint16_t scale_bits = static_cast<uint16_t>(block[0]) |
                                    (static_cast<uint16_t>(block[1]) << 8u);
        sum += fp16_to_fp32(scale_bits) * dot;
    }
    return sum;
}

} // namespace

extern "C" {

// Each input object contains 32 complete GGML rows. A row holds nine
// block_q4_0 values; each block is an LE fp16 scale plus 16 packed bytes.
void gemv_q4_0_bf16_f32(const uint8_t * weights, const bfloat16 * activation, float * output) {
    for (int row = 0; row < 32; ++row) {
        output[row] = dot_row(weights + row * 162, activation, row == 31);
    }
}

}
