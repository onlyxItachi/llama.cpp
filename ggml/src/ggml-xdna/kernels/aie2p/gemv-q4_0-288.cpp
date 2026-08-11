// SPDX-License-Identifier: MIT

// Scalar bring-up kernel for native GGML block_q4_0 bytes. This validates the
// compressed hardware path; a production kernel must vectorize the unpack and
// distribute output rows over more AIE tiles.

#include <stdint.h>

namespace {

float fp16_to_fp32(uint16_t value) {
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

} // namespace

extern "C" {

void zero_scalar_f32(float * output) {
    for (int row = 0; row < 32; ++row) {
        output[row] = 0.0f;
    }
}

// Each input object contains 32 complete GGML rows. A row holds nine
// block_q4_0 values; each block is an LE fp16 scale plus 16 packed bytes.
void gemv_q4_0_bf16_f32(const uint8_t * weights, const bfloat16 * activation, float * output) {
    for (int row = 0; row < 32; ++row) {
        float sum = 0.0f;
        for (int block_index = 0; block_index < 9; ++block_index) {
            const uint8_t * block = weights + row * 162 + block_index * 18;
            const uint16_t scale_bits = static_cast<uint16_t>(block[0]) |
                                        (static_cast<uint16_t>(block[1]) << 8u);
            const float scale = fp16_to_fp32(scale_bits);
            const int activation_base = block_index * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed = block[2 + j];
                const int low = static_cast<int>(packed & 0x0fu) - 8;
                const int high = static_cast<int>(packed >> 4u) - 8;
                sum += scale * static_cast<float>(low) *
                       static_cast<float>(activation[activation_base + j]);
                sum += scale * static_cast<float>(high) *
                       static_cast<float>(activation[activation_base + j + 16]);
            }
        }
        output[row] += sum;
    }
}

}
