# Build-time dependencies are intentionally external to llama.cpp.
IRON_ENV ?=

PYTHON ?= $(IRON_ENV)/bin/python
SITE_PACKAGES = $(shell "$(PYTHON)" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null)
MLIR_AIE ?= $(SITE_PACKAGES)/mlir_aie
PEANO ?= $(SITE_PACKAGES)/llvm-aie
AIECC ?= $(MLIR_AIE)/bin/aiecc
PEANO_CLANGXX ?= $(PEANO)/bin/clang++
AIE_API_INCLUDE ?= $(MLIR_AIE)/include
XCLBINUTIL ?= xclbinutil

BF16_PROGRAM := gemv-bf16-288.mlir
BF16_OBJECT := gemv-bf16-288.o
BF16_XCLBIN := gemv-bf16-288.xclbin
BF16_INSTRUCTIONS := gemv-bf16-288.insts.bin
BF16_BUNDLE := gemv-bf16-288.ggmlxdna
BF16_M8192_K2048_PROGRAM := gemv-bf16-m8192-k2048.mlir
BF16_M8192_K2048_OBJECT := gemv-bf16-m8192-k2048.o
BF16_M8192_K2048_XCLBIN := gemv-bf16-m8192-k2048.xclbin
BF16_M8192_K2048_INSTRUCTIONS := gemv-bf16-m8192-k2048.insts.bin
BF16_M8192_K2048_BUNDLE := gemv-bf16-m8192-k2048.ggmlxdna
BF16_M8192_K3072_PROGRAM := gemv-bf16-m8192-k3072.mlir
BF16_M8192_K3072_OBJECT := gemv-bf16-m8192-k3072.o
BF16_M8192_K3072_XCLBIN := gemv-bf16-m8192-k3072.xclbin
BF16_M8192_K3072_INSTRUCTIONS := gemv-bf16-m8192-k3072.insts.bin
BF16_M8192_K3072_BUNDLE := gemv-bf16-m8192-k3072.ggmlxdna
Q4_PROGRAM := gemv-q4_0-288.mlir
Q4_OBJECT := gemv-q4_0-288.o
Q4_XCLBIN := gemv-q4_0-288.xclbin
Q4_INSTRUCTIONS := gemv-q4_0-288.insts.bin
Q4_BUNDLE := gemv-q4_0-288.ggmlxdna
Q4_E4B_KV_PROGRAM := gemv-q4_0-m512-k2560.mlir
Q4_E4B_KV_OBJECT := gemv-q4_0-m512-k2560.o
Q4_E4B_KV_XCLBIN := gemv-q4_0-m512-k2560.xclbin
Q4_E4B_KV_INSTRUCTIONS := gemv-q4_0-m512-k2560.insts.bin
Q4_E4B_KV_BUNDLE := gemv-q4_0-m512-k2560.ggmlxdna
Q4_M10240_K2560_PROGRAM := gemv-q4_0-m10240-k2560.mlir
Q4_M10240_K2560_OBJECT := gemv-q4_0-m10240-k2560.o
Q4_M10240_K2560_XCLBIN := gemv-q4_0-m10240-k2560.xclbin
Q4_M10240_K2560_INSTRUCTIONS := gemv-q4_0-m10240-k2560.insts.bin
Q4_M10240_K2560_BUNDLE := gemv-q4_0-m10240-k2560.ggmlxdna
Q8_M9216_K2560_PROGRAM := gemv-q8_0-m9216-k2560.mlir
Q8_M9216_K2560_OBJECT := gemv-q8_0-m9216-k2560.o
Q8_M9216_K2560_XCLBIN := gemv-q8_0-m9216-k2560.xclbin
Q8_M9216_K2560_INSTRUCTIONS := gemv-q8_0-m9216-k2560.insts.bin
Q8_M9216_K2560_BUNDLE := gemv-q8_0-m9216-k2560.ggmlxdna

.PHONY: all check-environment clean
.DELETE_ON_ERROR:

all: $(Q4_BUNDLE) $(Q4_E4B_KV_BUNDLE) $(Q4_M10240_K2560_BUNDLE) $(Q8_M9216_K2560_BUNDLE) $(BF16_BUNDLE) $(BF16_M8192_K2048_BUNDLE) $(BF16_M8192_K3072_BUNDLE)

clean:
	$(RM) $(BF16_PROGRAM) $(BF16_OBJECT) $(BF16_XCLBIN) $(BF16_INSTRUCTIONS) $(BF16_BUNDLE)
	$(RM) $(BF16_M8192_K2048_PROGRAM) $(BF16_M8192_K2048_OBJECT) $(BF16_M8192_K2048_XCLBIN) $(BF16_M8192_K2048_INSTRUCTIONS) $(BF16_M8192_K2048_BUNDLE)
	$(RM) $(BF16_M8192_K3072_PROGRAM) $(BF16_M8192_K3072_OBJECT) $(BF16_M8192_K3072_XCLBIN) $(BF16_M8192_K3072_INSTRUCTIONS) $(BF16_M8192_K3072_BUNDLE)
	$(RM) $(Q4_PROGRAM) $(Q4_OBJECT) $(Q4_XCLBIN) $(Q4_INSTRUCTIONS) $(Q4_BUNDLE)
	$(RM) $(Q4_E4B_KV_PROGRAM) $(Q4_E4B_KV_OBJECT) $(Q4_E4B_KV_XCLBIN) $(Q4_E4B_KV_INSTRUCTIONS) $(Q4_E4B_KV_BUNDLE)
	$(RM) $(Q4_M10240_K2560_PROGRAM) $(Q4_M10240_K2560_OBJECT) $(Q4_M10240_K2560_XCLBIN) $(Q4_M10240_K2560_INSTRUCTIONS) $(Q4_M10240_K2560_BUNDLE)
	$(RM) $(Q8_M9216_K2560_PROGRAM) $(Q8_M9216_K2560_OBJECT) $(Q8_M9216_K2560_XCLBIN) $(Q8_M9216_K2560_INSTRUCTIONS) $(Q8_M9216_K2560_BUNDLE)
	$(RM) -r gemv-bf16-288.mlir.prj gemv-bf16-m8192-k2048.mlir.prj gemv-bf16-m8192-k3072.mlir.prj gemv-q4_0-288.mlir.prj gemv-q4_0-m512-k2560.mlir.prj gemv-q4_0-m10240-k2560.mlir.prj gemv-q8_0-m9216-k2560.mlir.prj

check-environment:
	@test -x "$(PYTHON)" || { echo "set IRON_ENV to an MLIR-AIE IRON environment" >&2; exit 1; }
	@test -x "$(AIECC)" || { echo "aiecc not found under MLIR_AIE=$(MLIR_AIE)" >&2; exit 1; }
	@test -x "$(PEANO_CLANGXX)" || { echo "Peano clang++ not found under PEANO=$(PEANO)" >&2; exit 1; }
	@command -v "$(XCLBINUTIL)" >/dev/null || { echo "xclbinutil not found; source the XRT setup script" >&2; exit 1; }

$(BF16_OBJECT): gemv-bf16-288.cpp | check-environment
	$(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG -c $< -o $@

$(BF16_PROGRAM): gemv-bf16-288.py | check-environment
	$(PYTHON) $< > $@

$(BF16_XCLBIN) $(BF16_INSTRUCTIONS) &: $(BF16_PROGRAM) $(BF16_OBJECT) | check-environment
	PATH="$(MLIR_AIE)/bin:$$PATH" $(AIECC) \
		--alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host --xclbin-name=$(BF16_XCLBIN) \
		--no-xchesscc --no-xbridge --peano "$(PEANO)" \
		--aie-generate-npu-insts --npu-insts-name=$(BF16_INSTRUCTIONS) $<
	@test -s "$(BF16_XCLBIN)" && test -s "$(BF16_INSTRUCTIONS)"

$(BF16_BUNDLE): pack-artifact.py $(BF16_XCLBIN) $(BF16_INSTRUCTIONS) | check-environment
	$(PYTHON) $< --abi-version 1 --architecture aie2p --kind-id 1 --m 288 --k 288 \
		--weight-bytes 165888 --activation-bytes 576 --output-bytes 1152 \
		--xclbin $(BF16_XCLBIN) \
		--instructions $(BF16_INSTRUCTIONS) --output $@

$(BF16_M8192_K2048_OBJECT): gemv-bf16-m8192.cpp | check-environment
	$(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
		-D__AIE_API_AIE_ADF_HPP__ -DGGML_XDNA_BF16_COLUMNS=2048 \
		-DGGML_XDNA_BF16_KERNEL_NAME=gemv_bf16_f32_m8_k2048 \
		-I"$(AIE_API_INCLUDE)" -c $< -o $@

$(BF16_M8192_K2048_PROGRAM): gemv-bf16-m8192.py | check-environment
	$(PYTHON) $< --columns 2048 > $@

$(BF16_M8192_K2048_XCLBIN) $(BF16_M8192_K2048_INSTRUCTIONS) &: $(BF16_M8192_K2048_PROGRAM) $(BF16_M8192_K2048_OBJECT) | check-environment
	PATH="$(MLIR_AIE)/bin:$$PATH" $(AIECC) \
		--alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host --xclbin-name=$(BF16_M8192_K2048_XCLBIN) \
		--no-xchesscc --no-xbridge --peano "$(PEANO)" \
		--aie-generate-npu-insts --npu-insts-name=$(BF16_M8192_K2048_INSTRUCTIONS) $<
	@test -s "$(BF16_M8192_K2048_XCLBIN)" && test -s "$(BF16_M8192_K2048_INSTRUCTIONS)"

$(BF16_M8192_K2048_BUNDLE): pack-artifact.py $(BF16_M8192_K2048_XCLBIN) $(BF16_M8192_K2048_INSTRUCTIONS) | check-environment
	$(PYTHON) $< --abi-version 1 --architecture aie2p --kind-id 1 --m 8192 --k 2048 \
		--weight-bytes 33554432 --activation-bytes 4096 --output-bytes 32768 \
		--xclbin $(BF16_M8192_K2048_XCLBIN) \
		--instructions $(BF16_M8192_K2048_INSTRUCTIONS) --output $@

$(BF16_M8192_K3072_OBJECT): gemv-bf16-m8192.cpp | check-environment
	$(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
		-D__AIE_API_AIE_ADF_HPP__ -DGGML_XDNA_BF16_COLUMNS=3072 \
		-DGGML_XDNA_BF16_KERNEL_NAME=gemv_bf16_f32_m8_k3072 \
		-I"$(AIE_API_INCLUDE)" -c $< -o $@

$(BF16_M8192_K3072_PROGRAM): gemv-bf16-m8192.py | check-environment
	$(PYTHON) $< --columns 3072 > $@

$(BF16_M8192_K3072_XCLBIN) $(BF16_M8192_K3072_INSTRUCTIONS) &: $(BF16_M8192_K3072_PROGRAM) $(BF16_M8192_K3072_OBJECT) | check-environment
	PATH="$(MLIR_AIE)/bin:$$PATH" $(AIECC) \
		--alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host --xclbin-name=$(BF16_M8192_K3072_XCLBIN) \
		--no-xchesscc --no-xbridge --peano "$(PEANO)" \
		--aie-generate-npu-insts --npu-insts-name=$(BF16_M8192_K3072_INSTRUCTIONS) $<
	@test -s "$(BF16_M8192_K3072_XCLBIN)" && test -s "$(BF16_M8192_K3072_INSTRUCTIONS)"

$(BF16_M8192_K3072_BUNDLE): pack-artifact.py $(BF16_M8192_K3072_XCLBIN) $(BF16_M8192_K3072_INSTRUCTIONS) | check-environment
	$(PYTHON) $< --abi-version 1 --architecture aie2p --kind-id 1 --m 8192 --k 3072 \
		--weight-bytes 50331648 --activation-bytes 6144 --output-bytes 32768 \
		--xclbin $(BF16_M8192_K3072_XCLBIN) \
		--instructions $(BF16_M8192_K3072_INSTRUCTIONS) --output $@

$(Q4_OBJECT): gemv-q4_0-288.cpp | check-environment
	$(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
		-D__AIE_API_AIE_ADF_HPP__ -I"$(AIE_API_INCLUDE)" -c $< -o $@

$(Q4_PROGRAM): gemv-q4_0-288.py | check-environment
	$(PYTHON) $< > $@

$(Q4_XCLBIN) $(Q4_INSTRUCTIONS) &: $(Q4_PROGRAM) $(Q4_OBJECT) | check-environment
	PATH="$(MLIR_AIE)/bin:$$PATH" $(AIECC) \
		--alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host --xclbin-name=$(Q4_XCLBIN) \
		--no-xchesscc --no-xbridge --peano "$(PEANO)" \
		--aie-generate-npu-insts --npu-insts-name=$(Q4_INSTRUCTIONS) $<
	@test -s "$(Q4_XCLBIN)" && test -s "$(Q4_INSTRUCTIONS)"

$(Q4_BUNDLE): pack-artifact.py $(Q4_XCLBIN) $(Q4_INSTRUCTIONS) | check-environment
	$(PYTHON) $< --abi-version 1 --architecture aie2p --kind-id 2 --m 288 --k 288 \
		--weight-bytes 46656 --activation-bytes 576 --output-bytes 1152 \
		--xclbin $(Q4_XCLBIN) \
		--instructions $(Q4_INSTRUCTIONS) --output $@

$(Q4_E4B_KV_OBJECT): gemv-q4_0-m512-k2560.cpp | check-environment
	$(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
		-D__AIE_API_AIE_ADF_HPP__ -I"$(AIE_API_INCLUDE)" -c $< -o $@

$(Q4_E4B_KV_PROGRAM): gemv-q4_0-m512-k2560.py | check-environment
	$(PYTHON) $< > $@

$(Q4_E4B_KV_XCLBIN) $(Q4_E4B_KV_INSTRUCTIONS) &: $(Q4_E4B_KV_PROGRAM) $(Q4_E4B_KV_OBJECT) | check-environment
	PATH="$(MLIR_AIE)/bin:$$PATH" $(AIECC) \
		--alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host --xclbin-name=$(Q4_E4B_KV_XCLBIN) \
		--no-xchesscc --no-xbridge --peano "$(PEANO)" \
		--aie-generate-npu-insts --npu-insts-name=$(Q4_E4B_KV_INSTRUCTIONS) $<
	@test -s "$(Q4_E4B_KV_XCLBIN)" && test -s "$(Q4_E4B_KV_INSTRUCTIONS)"

$(Q4_E4B_KV_BUNDLE): pack-artifact.py $(Q4_E4B_KV_XCLBIN) $(Q4_E4B_KV_INSTRUCTIONS) | check-environment
	$(PYTHON) $< --abi-version 1 --architecture aie2p --kind-id 2 --m 512 --k 2560 \
		--weight-bytes 737280 --activation-bytes 5120 --output-bytes 2048 \
		--xclbin $(Q4_E4B_KV_XCLBIN) \
		--instructions $(Q4_E4B_KV_INSTRUCTIONS) --output $@

$(Q4_M10240_K2560_OBJECT): gemv-q4_0-m512-k2560.cpp | check-environment
	$(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
		-D__AIE_API_AIE_ADF_HPP__ -I"$(AIE_API_INCLUDE)" -c $< -o $@

$(Q4_M10240_K2560_PROGRAM): gemv-q4_0-m10240-k2560.py | check-environment
	$(PYTHON) $< > $@

$(Q4_M10240_K2560_XCLBIN) $(Q4_M10240_K2560_INSTRUCTIONS) &: $(Q4_M10240_K2560_PROGRAM) $(Q4_M10240_K2560_OBJECT) | check-environment
	PATH="$(MLIR_AIE)/bin:$$PATH" $(AIECC) \
		--alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host --xclbin-name=$(Q4_M10240_K2560_XCLBIN) \
		--no-xchesscc --no-xbridge --peano "$(PEANO)" \
		--aie-generate-npu-insts --npu-insts-name=$(Q4_M10240_K2560_INSTRUCTIONS) $<
	@test -s "$(Q4_M10240_K2560_XCLBIN)" && test -s "$(Q4_M10240_K2560_INSTRUCTIONS)"

$(Q4_M10240_K2560_BUNDLE): pack-artifact.py $(Q4_M10240_K2560_XCLBIN) $(Q4_M10240_K2560_INSTRUCTIONS) | check-environment
	$(PYTHON) $< --abi-version 1 --architecture aie2p --kind-id 2 --m 10240 --k 2560 \
		--weight-bytes 14745600 --activation-bytes 5120 --output-bytes 40960 \
		--xclbin $(Q4_M10240_K2560_XCLBIN) \
		--instructions $(Q4_M10240_K2560_INSTRUCTIONS) --output $@

$(Q8_M9216_K2560_OBJECT): gemv-q8_0-m9216-k2560.cpp | check-environment
	$(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
		-D__AIE_API_AIE_ADF_HPP__ -I"$(AIE_API_INCLUDE)" -c $< -o $@

$(Q8_M9216_K2560_PROGRAM): gemv-q8_0-m9216-k2560.py | check-environment
	$(PYTHON) $< > $@

$(Q8_M9216_K2560_XCLBIN) $(Q8_M9216_K2560_INSTRUCTIONS) &: $(Q8_M9216_K2560_PROGRAM) $(Q8_M9216_K2560_OBJECT) | check-environment
	PATH="$(MLIR_AIE)/bin:$$PATH" $(AIECC) \
		--alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host --xclbin-name=$(Q8_M9216_K2560_XCLBIN) \
		--no-xchesscc --no-xbridge --peano "$(PEANO)" \
		--aie-generate-npu-insts --npu-insts-name=$(Q8_M9216_K2560_INSTRUCTIONS) $<
	@test -s "$(Q8_M9216_K2560_XCLBIN)" && test -s "$(Q8_M9216_K2560_INSTRUCTIONS)"

$(Q8_M9216_K2560_BUNDLE): pack-artifact.py $(Q8_M9216_K2560_XCLBIN) $(Q8_M9216_K2560_INSTRUCTIONS) | check-environment
	$(PYTHON) $< --abi-version 1 --architecture aie2p --kind-id 3 --m 9216 --k 2560 \
		--weight-bytes 25067520 --activation-bytes 5120 --output-bytes 36864 \
		--xclbin $(Q8_M9216_K2560_XCLBIN) \
		--instructions $(Q8_M9216_K2560_INSTRUCTIONS) --output $@
