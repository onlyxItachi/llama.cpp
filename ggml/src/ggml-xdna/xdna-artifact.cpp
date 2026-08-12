#include "xdna-artifact.h"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ggml_xdna {
namespace detail {

namespace {

constexpr size_t artifact_header_bytes = 64;
constexpr char artifact_magic[8] = { 'G', 'G', 'X', 'D', 'N', 'A', '1', '\0' };

uint32_t read_le32(const unsigned char * value) {
    return static_cast<uint32_t>(value[0]) |
           (static_cast<uint32_t>(value[1]) << 8u) |
           (static_cast<uint32_t>(value[2]) << 16u) |
           (static_cast<uint32_t>(value[3]) << 24u);
}

uint64_t read_le64(const unsigned char * value) {
    return static_cast<uint64_t>(read_le32(value)) |
           (static_cast<uint64_t>(read_le32(value + 4)) << 32u);
}

} // namespace

artifact_contents read_artifact_bundle(const std::string & path, const xdna_kernel_variant & variant) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("failed to open XDNA artifact bundle " + path);
    }

    const auto end = stream.tellg();
    if (end < static_cast<std::streamoff>(artifact_header_bytes)) {
        throw std::runtime_error("truncated XDNA artifact bundle " + path);
    }

    std::vector<unsigned char> bytes(static_cast<size_t>(end));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(end));
    if (!stream) {
        throw std::runtime_error("failed to read XDNA artifact bundle " + path);
    }

    if (std::memcmp(bytes.data(), artifact_magic, sizeof(artifact_magic)) != 0 ||
            read_le32(bytes.data() + 8) != artifact_header_bytes ||
            read_le32(bytes.data() + 12) != variant.artifact_abi_version ||
            read_le32(bytes.data() + 16) != variant.artifact_kind ||
            read_le32(bytes.data() + 20) != variant.m ||
            read_le32(bytes.data() + 24) != variant.k ||
            read_le32(bytes.data() + 28) != variant.weight_bytes ||
            read_le32(bytes.data() + 32) != variant.device_activation_bytes ||
            read_le32(bytes.data() + 36) != variant.device_output_bytes ||
            read_le64(bytes.data() + 56) != 0 || variant.n != 1) {
        throw std::runtime_error("XDNA artifact ABI metadata mismatch in " + path);
    }

    const uint64_t xclbin_bytes_u64 = read_le64(bytes.data() + 40);
    const uint64_t instruction_bytes_u64 = read_le64(bytes.data() + 48);
    const size_t payload_bytes = bytes.size() - artifact_header_bytes;
    if (xclbin_bytes_u64 == 0 || instruction_bytes_u64 == 0 ||
            xclbin_bytes_u64 > payload_bytes ||
            instruction_bytes_u64 != payload_bytes - xclbin_bytes_u64 ||
            instruction_bytes_u64 % sizeof(uint32_t) != 0) {
        throw std::runtime_error("invalid XDNA artifact payload sizes in " + path);
    }

    const size_t xclbin_bytes = static_cast<size_t>(xclbin_bytes_u64);
    const size_t instruction_bytes = static_cast<size_t>(instruction_bytes_u64);
    artifact_contents result;
    result.xclbin_data.assign(
        reinterpret_cast<const char *>(bytes.data() + artifact_header_bytes),
        reinterpret_cast<const char *>(bytes.data() + artifact_header_bytes + xclbin_bytes));
    result.instructions.resize(instruction_bytes / sizeof(uint32_t));
    std::memcpy(
        result.instructions.data(),
        bytes.data() + artifact_header_bytes + xclbin_bytes,
        instruction_bytes);
    return result;
}

} // namespace detail
} // namespace ggml_xdna
