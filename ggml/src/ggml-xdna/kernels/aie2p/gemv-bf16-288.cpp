// SPDX-License-Identifier: MIT

// A deliberately scalar, single-tile kernel used to validate the complete
// GGML -> XRT -> AIE2P path.  It is a bring-up kernel, not a performance
// implementation.
extern "C" {

void zero_scalar_f32(float * output) {
    for (int row = 0; row < 32; ++row) {
        output[row] = 0.0f;
    }
}

void gemv_scalar_bf16_f32(const bfloat16 * weights, const bfloat16 * activation, float * output) {
    for (int row = 0; row < 32; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < 32; ++column) {
            sum += static_cast<float>(weights[row * 32 + column]) * static_cast<float>(activation[column]);
        }
        output[row] += sum;
    }
}

}
