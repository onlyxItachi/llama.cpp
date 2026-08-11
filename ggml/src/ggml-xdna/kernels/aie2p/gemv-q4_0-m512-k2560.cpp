// SPDX-License-Identifier: MIT

// K=2560 compute-tile kernel shared by the registered M=512 and M=10240
// specializations. One AIE2P core processes one contiguous 16-row
// native-GGML-Q4_0 object per invocation.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int kColumns = 2560;
constexpr int kBlockValues = 32;
constexpr int kBlockBytes = 18;
constexpr int kBlocksPerRow = kColumns / kBlockValues;
constexpr int kRowBytes = kBlocksPerRow * kBlockBytes;

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
    const bfloat16 scale = static_cast<bfloat16>(fp16_to_fp32(scale_bits));
    return aie::mul(
        q_bf16, aie::broadcast<bfloat16, 32>(scale)).template to_vector<bfloat16>();
}

__aie_inline void dot_rows_3(
        const uint8_t * weights, const bfloat16 * activation, float * output) {
    auto acc0 = aie::zeros<accfloat, 32>();
    auto acc1 = aie::zeros<accfloat, 32>();
    auto acc2 = aie::zeros<accfloat, 32>();
    const uint8_t * weight0 = weights;
    const uint8_t * weight1 = weights + kRowBytes;
    const uint8_t * weight2 = weights + 2 * kRowBytes;
    const bfloat16 * activation_block = activation;
    for (int block_index = 0; block_index < kBlocksPerRow; ++block_index) {
        const auto x = aie::load_v<32>(activation_block);
        acc0 = aie::mac(acc0, dequantize_block(weight0, false), x);
        acc1 = aie::mac(acc1, dequantize_block(weight1, false), x);
        acc2 = aie::mac(acc2, dequantize_block(weight2, false), x);
        weight0 += kBlockBytes;
        weight1 += kBlockBytes;
        weight2 += kBlockBytes;
        activation_block += kBlockValues;
    }
    output[0] = aie::reduce_add<float>(acc0);
    output[1] = aie::reduce_add<float>(acc1);
    output[2] = aie::reduce_add<float>(acc2);
}

__aie_inline float dot_final_row(const uint8_t * weights, const bfloat16 * activation) {
    auto acc = aie::zeros<accfloat, 32>();
    for (int block_index = 0; block_index < kBlocksPerRow; ++block_index) {
        const auto x = aie::load_v<32>(activation + block_index * kBlockValues);
        acc = aie::mac(
            acc,
            dequantize_block(
                weights + block_index * kBlockBytes,
                block_index == kBlocksPerRow - 1),
            x);
    }
    return aie::reduce_add<float>(acc);
}

} // namespace

extern "C" {

void gemv_q4_0_bf16_f32_m16_k2560(
        const uint8_t * weights, const bfloat16 * activation, float * output) {
    for (int row = 0; row < 15; row += 3) {
        dot_rows_3(weights + row * kRowBytes, activation, output + row);
    }
    output[15] = dot_final_row(weights + 15 * kRowBytes, activation);
}

}
