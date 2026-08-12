// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>

#ifndef GGML_XDNA_BF16_COLUMNS
#error "GGML_XDNA_BF16_COLUMNS must name the compile-time K dimension"
#endif

#ifndef GGML_XDNA_BF16_KERNEL_NAME
#error "GGML_XDNA_BF16_KERNEL_NAME must name the exported kernel symbol"
#endif

namespace {

constexpr int kColumns = GGML_XDNA_BF16_COLUMNS;
constexpr int kVectorSize = 32;
constexpr int kRowsPerWorker = 8;

static_assert(kColumns > 0 && kColumns % kVectorSize == 0);

__aie_inline void dot_rows_3(const bfloat16 * weights, const bfloat16 * activation, float * output) {
    auto acc0 = aie::zeros<accfloat, kVectorSize>();
    auto acc1 = aie::zeros<accfloat, kVectorSize>();
    auto acc2 = aie::zeros<accfloat, kVectorSize>();
    const bfloat16 * weight0 = weights;
    const bfloat16 * weight1 = weights + kColumns;
    const bfloat16 * weight2 = weights + 2 * kColumns;

    for (int column = 0; column < kColumns; column += kVectorSize) {
        const auto x = aie::load_v<kVectorSize>(activation + column);
        acc0 = aie::mac(acc0, aie::load_v<kVectorSize>(weight0 + column), x);
        acc1 = aie::mac(acc1, aie::load_v<kVectorSize>(weight1 + column), x);
        acc2 = aie::mac(acc2, aie::load_v<kVectorSize>(weight2 + column), x);
    }

    output[0] = aie::reduce_add<float>(acc0);
    output[1] = aie::reduce_add<float>(acc1);
    output[2] = aie::reduce_add<float>(acc2);
}

__aie_inline void dot_rows_2(const bfloat16 * weights, const bfloat16 * activation, float * output) {
    auto acc0 = aie::zeros<accfloat, kVectorSize>();
    auto acc1 = aie::zeros<accfloat, kVectorSize>();
    const bfloat16 * weight0 = weights;
    const bfloat16 * weight1 = weights + kColumns;

    for (int column = 0; column < kColumns; column += kVectorSize) {
        const auto x = aie::load_v<kVectorSize>(activation + column);
        acc0 = aie::mac(acc0, aie::load_v<kVectorSize>(weight0 + column), x);
        acc1 = aie::mac(acc1, aie::load_v<kVectorSize>(weight1 + column), x);
    }

    output[0] = aie::reduce_add<float>(acc0);
    output[1] = aie::reduce_add<float>(acc1);
}

} // namespace

extern "C" {

void GGML_XDNA_BF16_KERNEL_NAME(
        const bfloat16 * weights,
        const bfloat16 * activation,
        float * output) {
    for (int row = 0; row < kRowsPerWorker - 2; row += 3) {
        dot_rows_3(weights + row * kColumns, activation, output + row);
    }
    dot_rows_2(
        weights + (kRowsPerWorker - 2) * kColumns,
        activation,
        output + (kRowsPerWorker - 2));
}

}
