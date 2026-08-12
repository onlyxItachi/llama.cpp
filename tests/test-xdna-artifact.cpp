#include "xdna-artifact.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t header_bytes = 64;
constexpr unsigned char magic[8] = { 'G', 'G', 'X', 'D', 'N', 'A', '1', '\0' };
static_assert(static_cast<uint32_t>(ggml_xdna::detail::artifact_architecture_id::aie2) == 1);
static_assert(static_cast<uint32_t>(ggml_xdna::detail::artifact_architecture_id::aie2p) == 2);

void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-xdna-artifact: %s\n", message);
        abort();
    }
}

class temporary_file {
public:
    temporary_file() {
        const char * temporary_directory = getenv("TMPDIR");
        if (temporary_directory == nullptr || temporary_directory[0] == '\0') {
            temporary_directory = "/tmp";
        }
        std::string pattern = std::string(temporary_directory) + "/test-xdna-artifact-XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        const int descriptor = mkstemp(writable.data());
        require(descriptor >= 0, "failed to create the temporary bundle");
        require(close(descriptor) == 0, "failed to close the temporary bundle");
        path = writable.data();
    }

    ~temporary_file() {
        unlink(path.c_str());
    }

    temporary_file(const temporary_file &) = delete;
    temporary_file & operator=(const temporary_file &) = delete;

    void write(const std::vector<unsigned char> & bytes) const {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(stream), "failed to open the temporary bundle for writing");
        stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(static_cast<bool>(stream), "failed to write the temporary bundle");
    }

    std::string path;
};

void write_le32(std::vector<unsigned char> & bytes, size_t offset, uint32_t value) {
    require(offset <= bytes.size() && bytes.size() - offset >= sizeof(value), "test write exceeded the bundle");
    for (size_t i = 0; i < sizeof(value); ++i) {
        bytes[offset + i] = static_cast<unsigned char>(value >> (8 * i));
    }
}

void write_le64(std::vector<unsigned char> & bytes, size_t offset, uint64_t value) {
    require(offset <= bytes.size() && bytes.size() - offset >= sizeof(value), "test write exceeded the bundle");
    for (size_t i = 0; i < sizeof(value); ++i) {
        bytes[offset + i] = static_cast<unsigned char>(value >> (8 * i));
    }
}

ggml_xdna::xdna_kernel_variant make_variant(
        ggml_xdna::device_architecture architecture = ggml_xdna::device_architecture::aie2p,
        uint32_t artifact_abi_version = 1) {
    ggml_xdna::xdna_kernel_variant variant = {};
    variant.architecture = architecture;
    variant.m = 13;
    variant.n = 1;
    variant.k = 17;
    variant.artifact_kind = 7;
    variant.artifact_abi_version = artifact_abi_version;
    variant.weight_bytes = 221;
    variant.device_activation_bytes = 34;
    variant.device_output_bytes = 52;
    return variant;
}

std::vector<unsigned char> make_bundle(
        const ggml_xdna::xdna_kernel_variant & variant,
        const std::vector<unsigned char> & xclbin,
        const std::vector<unsigned char> & instructions,
        uint32_t architecture_id = 0) {
    std::vector<unsigned char> bytes(header_bytes + xclbin.size() + instructions.size(), 0);
    std::memcpy(bytes.data(), magic, sizeof(magic));
    write_le32(bytes, 8, header_bytes);
    write_le32(bytes, 12, variant.artifact_abi_version);
    write_le32(bytes, 16, variant.artifact_kind);
    write_le32(bytes, 20, static_cast<uint32_t>(variant.m));
    write_le32(bytes, 24, static_cast<uint32_t>(variant.k));
    write_le32(bytes, 28, static_cast<uint32_t>(variant.weight_bytes));
    write_le32(bytes, 32, static_cast<uint32_t>(variant.device_activation_bytes));
    write_le32(bytes, 36, static_cast<uint32_t>(variant.device_output_bytes));
    write_le64(bytes, 40, xclbin.size());
    write_le64(bytes, 48, instructions.size());
    write_le32(bytes, 56, architecture_id);
    if (!xclbin.empty()) {
        std::memcpy(bytes.data() + header_bytes, xclbin.data(), xclbin.size());
    }
    if (!instructions.empty()) {
        std::memcpy(bytes.data() + header_bytes + xclbin.size(), instructions.data(), instructions.size());
    }
    return bytes;
}

void expect_accepted(
        const temporary_file & file,
        const std::vector<unsigned char> & bytes,
        const ggml_xdna::xdna_kernel_variant & variant,
        const char * name) {
    file.write(bytes);
    try {
        (void) ggml_xdna::detail::read_artifact_bundle(file.path, variant);
    } catch (const std::runtime_error & error) {
        fprintf(stderr, "test-xdna-artifact: %s was rejected: %s\n", name, error.what());
        abort();
    }
}

void expect_rejected(
        const temporary_file & file,
        const std::vector<unsigned char> & bytes,
        const ggml_xdna::xdna_kernel_variant & variant,
        const char * expected,
        const char * name) {
    file.write(bytes);
    try {
        (void) ggml_xdna::detail::read_artifact_bundle(file.path, variant);
    } catch (const std::runtime_error & error) {
        if (std::string(error.what()).find(expected) != std::string::npos) {
            return;
        }
        fprintf(stderr, "test-xdna-artifact: %s produced an unexpected error: %s\n", name, error.what());
        abort();
    }
    fprintf(stderr, "test-xdna-artifact: %s was accepted\n", name);
    abort();
}

void test_valid_bundle(const temporary_file & file) {
    const ggml_xdna::xdna_kernel_variant variant = make_variant();
    const std::vector<unsigned char> xclbin { 0x78, 0x63, 0x6c, 0x62, 0x69 };
    const std::vector<uint32_t> instruction_words { 0x01020304, 0xaabbccdd };
    std::vector<unsigned char> instructions(instruction_words.size() * sizeof(uint32_t));
    std::memcpy(instructions.data(), instruction_words.data(), instructions.size());
    file.write(make_bundle(variant, xclbin, instructions));

    const ggml_xdna::detail::artifact_contents contents =
        ggml_xdna::detail::read_artifact_bundle(file.path, variant);
    require(contents.xclbin_data.size() == xclbin.size(), "the valid xclbin size changed");
    require(std::memcmp(contents.xclbin_data.data(), xclbin.data(), xclbin.size()) == 0, "the valid xclbin data changed");
    require(contents.instructions == instruction_words, "the valid instruction stream changed");
}

void test_header_validation(const temporary_file & file) {
    const ggml_xdna::xdna_kernel_variant variant = make_variant();
    const std::vector<unsigned char> xclbin { 1, 2, 3, 4, 5 };
    const std::vector<unsigned char> instructions { 1, 0, 0, 0, 2, 0, 0, 0 };
    const std::vector<unsigned char> valid = make_bundle(variant, xclbin, instructions);

    std::vector<unsigned char> changed(valid.begin(), valid.begin() + header_bytes - 1);
    expect_rejected(file, changed, variant, "truncated XDNA artifact bundle", "truncated header");

    changed = valid;
    changed[0] ^= 0xff;
    expect_rejected(file, changed, variant, "ABI metadata mismatch", "bad magic");

    changed = valid;
    changed[6] = '2';
    expect_rejected(file, changed, variant, "ABI metadata mismatch", "bad container version");

    struct metadata_case {
        size_t offset;
        uint32_t value;
        const char * name;
    };
    const metadata_case cases[] = {
        { 8,  static_cast<uint32_t>(header_bytes + 1), "bad payload offset" },
        { 12, variant.artifact_abi_version + 1, "bad ABI version" },
        { 16, variant.artifact_kind + 1,        "bad kernel kind" },
        { 20, static_cast<uint32_t>(variant.m + 1), "bad M" },
        { 24, static_cast<uint32_t>(variant.k + 1), "bad K" },
        { 28, static_cast<uint32_t>(variant.weight_bytes + 1), "bad weight size" },
        { 32, static_cast<uint32_t>(variant.device_activation_bytes + 1), "bad activation size" },
        { 36, static_cast<uint32_t>(variant.device_output_bytes + 1), "bad output size" },
    };
    for (const metadata_case & test : cases) {
        changed = valid;
        write_le32(changed, test.offset, test.value);
        expect_rejected(file, changed, variant, "ABI metadata mismatch", test.name);
    }

    changed = valid;
    write_le32(changed, 56, 1);
    expect_rejected(file, changed, variant, "ABI metadata mismatch", "nonzero ABI v1 architecture field");

    changed = valid;
    write_le32(changed, 60, 1);
    expect_rejected(file, changed, variant, "ABI metadata mismatch", "nonzero reserved header field");

    ggml_xdna::xdna_kernel_variant batched = variant;
    batched.n = 2;
    expect_rejected(file, valid, batched, "ABI metadata mismatch", "unsupported N");
}

void test_architecture_validation(const temporary_file & file) {
    const std::vector<unsigned char> xclbin { 1, 2, 3, 4, 5 };
    const std::vector<unsigned char> instructions { 1, 0, 0, 0, 2, 0, 0, 0 };

    const ggml_xdna::xdna_kernel_variant aie2_v1 =
        make_variant(ggml_xdna::device_architecture::aie2, 1);
    expect_rejected(
        file, make_bundle(aie2_v1, xclbin, instructions), aie2_v1,
        "ABI metadata mismatch", "AIE2 ABI v1");

    const ggml_xdna::xdna_kernel_variant aie2_v2 =
        make_variant(ggml_xdna::device_architecture::aie2, 2);
    const std::vector<unsigned char> valid_aie2 =
        make_bundle(
            aie2_v2, xclbin, instructions,
            static_cast<uint32_t>(ggml_xdna::detail::artifact_architecture_id::aie2));
    expect_accepted(file, valid_aie2, aie2_v2, "AIE2 ABI v2");

    const ggml_xdna::xdna_kernel_variant aie2p_v2 =
        make_variant(ggml_xdna::device_architecture::aie2p, 2);
    const std::vector<unsigned char> valid_aie2p =
        make_bundle(
            aie2p_v2, xclbin, instructions,
            static_cast<uint32_t>(ggml_xdna::detail::artifact_architecture_id::aie2p));
    expect_accepted(file, valid_aie2p, aie2p_v2, "AIE2P ABI v2");

    expect_rejected(
        file, valid_aie2p, aie2_v2,
        "ABI metadata mismatch", "AIE2P artifact for AIE2 variant");
    expect_rejected(
        file, valid_aie2, aie2p_v2,
        "ABI metadata mismatch", "AIE2 artifact for AIE2P variant");

    std::vector<unsigned char> changed = valid_aie2;
    write_le32(changed, 56, 0);
    expect_rejected(file, changed, aie2_v2, "ABI metadata mismatch", "zero ABI v2 architecture id");

    changed = valid_aie2;
    write_le32(changed, 56, 3);
    expect_rejected(file, changed, aie2_v2, "ABI metadata mismatch", "unknown ABI v2 architecture id");

    changed = valid_aie2;
    write_le32(changed, 60, 1);
    expect_rejected(file, changed, aie2_v2, "ABI metadata mismatch", "nonzero ABI v2 reserved field");

    const ggml_xdna::xdna_kernel_variant unknown_v2 =
        make_variant(ggml_xdna::device_architecture::unknown, 2);
    expect_rejected(
        file, valid_aie2, unknown_v2,
        "ABI metadata mismatch", "unknown variant architecture");

    const ggml_xdna::xdna_kernel_variant unsupported_abi =
        make_variant(ggml_xdna::device_architecture::aie2p, 3);
    expect_rejected(
        file,
        make_bundle(
            unsupported_abi, xclbin, instructions,
            static_cast<uint32_t>(ggml_xdna::detail::artifact_architecture_id::aie2p)),
        unsupported_abi,
        "ABI metadata mismatch", "unsupported artifact ABI version");
}

void test_payload_validation(const temporary_file & file) {
    const ggml_xdna::xdna_kernel_variant variant = make_variant();
    const std::vector<unsigned char> xclbin { 1, 2, 3, 4, 5 };
    const std::vector<unsigned char> instructions { 1, 0, 0, 0, 2, 0, 0, 0 };
    const std::vector<unsigned char> valid = make_bundle(variant, xclbin, instructions);

    std::vector<unsigned char> changed = valid;
    changed.pop_back();
    expect_rejected(file, changed, variant, "invalid XDNA artifact payload sizes", "truncated payload");

    changed = valid;
    changed.push_back(0);
    expect_rejected(file, changed, variant, "invalid XDNA artifact payload sizes", "trailing payload data");

    changed = make_bundle(variant, {}, instructions);
    expect_rejected(file, changed, variant, "invalid XDNA artifact payload sizes", "empty xclbin");

    changed = make_bundle(variant, xclbin, {});
    expect_rejected(file, changed, variant, "invalid XDNA artifact payload sizes", "empty instruction stream");

    changed = make_bundle(variant, xclbin, { 1, 2, 3 });
    expect_rejected(file, changed, variant, "invalid XDNA artifact payload sizes", "unaligned instruction stream");

    changed = valid;
    write_le64(changed, 40, std::numeric_limits<uint64_t>::max());
    write_le64(changed, 48, 1);
    expect_rejected(file, changed, variant, "invalid XDNA artifact payload sizes", "overflowing xclbin size");

    changed = valid;
    write_le64(changed, 48, std::numeric_limits<uint64_t>::max());
    expect_rejected(file, changed, variant, "invalid XDNA artifact payload sizes", "overflowing instruction size");

    changed = valid;
    write_le64(changed, 40, valid.size() - header_bytes + 1);
    write_le64(changed, 48, 4);
    expect_rejected(file, changed, variant, "invalid XDNA artifact payload sizes", "payload boundary underflow");
}

} // namespace

int main() {
    temporary_file file;
    test_valid_bundle(file);
    test_header_validation(file);
    test_architecture_validation(file);
    test_payload_validation(file);
    return 0;
}
