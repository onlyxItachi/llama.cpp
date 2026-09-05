# Build-time dependencies are intentionally external to llama.cpp.
IRON_ENV ?=

PYTHON ?= $(IRON_ENV)/bin/python
HOST_PYTHON ?= python3
SITE_PACKAGES = $(shell "$(PYTHON)" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null)
MLIR_AIE ?= $(SITE_PACKAGES)/mlir_aie
PEANO ?= $(SITE_PACKAGES)/llvm-aie
AIECC ?= $(MLIR_AIE)/bin/aiecc
PEANO_CLANGXX ?= $(PEANO)/bin/clang++
AIE_API_INCLUDE ?= $(MLIR_AIE)/include
XCLBINUTIL ?= xclbinutil

KERNELS := gemv-q4_0-288 gemv-q4_0-m512-k2560 gemv-q4_0-m10240-k2560 gemv-q8_0-m9216-k2560 \
           gemv-bf16-288 gemv-bf16-m8192-k2048 gemv-bf16-m8192-k3072
GENERATED := $(foreach kernel,$(KERNELS),$(addprefix $(kernel),.mlir .o .xclbin .insts.bin .ggmlxdna))
BF16_M8192_KERNELS := gemv-bf16-m8192-k2048 gemv-bf16-m8192-k3072
BF16_M8192_ARTIFACTS := $(foreach kernel,$(BF16_M8192_KERNELS),$(addprefix $(kernel),.xclbin .insts.bin .ggmlxdna))
BF16_M8192_CONTROLLER_VERIFIER := verify-bf16-m8192-controller.py
BF16_M8192_HOST_TEST := test-bf16-m8192.py
CORE_FLAGS = -D__AIE_API_AIE_ADF_HPP__ -I"$(AIE_API_INCLUDE)"
COMPILE_CORE = $(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG $(CORE_FLAGS) -c $< -o $@

.PHONY: all check-bf16-m8192-host check-environment clean
.DELETE_ON_ERROR:

all: $(addsuffix .ggmlxdna,$(KERNELS))

# Explicit targets keep implicit-rule intermediates as reusable build artifacts.
$(GENERATED):

check-bf16-m8192-host:
	PYTHONDONTWRITEBYTECODE=1 "$(HOST_PYTHON)" $(BF16_M8192_HOST_TEST)

clean:
	$(RM) $(GENERATED)
	$(RM) -r $(addsuffix .mlir.prj,$(KERNELS))

check-environment:
	@test -x "$(PYTHON)" || { echo "set IRON_ENV to an MLIR-AIE IRON environment" >&2; exit 1; }
	@test -x "$(AIECC)" || { echo "aiecc not found under MLIR_AIE=$(MLIR_AIE)" >&2; exit 1; }
	@test -x "$(PEANO_CLANGXX)" || { echo "Peano clang++ not found under PEANO=$(PEANO)" >&2; exit 1; }
	@command -v "$(XCLBINUTIL)" >/dev/null || { echo "xclbinutil not found; source the XRT setup script" >&2; exit 1; }

gemv-bf16-288.o: CORE_FLAGS :=
%.o: %.cpp | check-environment
	$(COMPILE_CORE)

gemv-q4_0-m10240-k2560.o: gemv-q4_0-m512-k2560.cpp | check-environment
	$(COMPILE_CORE)

gemv-bf16-m8192-k%.o: gemv-bf16-m8192.cpp | check-environment
	$(PEANO_CLANGXX) -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
		-D__AIE_API_AIE_ADF_HPP__ -DGGML_XDNA_BF16_COLUMNS=$* \
		-DGGML_XDNA_BF16_KERNEL_NAME=gemv_bf16_f32_m8_k$* \
		-I"$(AIE_API_INCLUDE)" -c $< -o $@

%.mlir: %.py | check-environment
	$(PYTHON) $< > $@

gemv-bf16-m8192-k%.mlir: gemv-bf16-m8192.py | check-environment
	$(PYTHON) $< --columns $* > $@

$(addprefix gemv-bf16-m8192-k2048,.xclbin .insts.bin .ggmlxdna): CONTROLLER_COLUMNS := 2048
$(addprefix gemv-bf16-m8192-k3072,.xclbin .insts.bin .ggmlxdna): CONTROLLER_COLUMNS := 3072
$(BF16_M8192_ARTIFACTS): $(BF16_M8192_CONTROLLER_VERIFIER)

%.xclbin %.insts.bin &: %.mlir %.o | check-environment
	PATH="$(MLIR_AIE)/bin:$$PATH" $(AIECC) \
		--alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host --xclbin-name=$*.xclbin \
		--no-xchesscc --no-xbridge --peano "$(PEANO)" \
		--aie-generate-npu-insts --npu-insts-name=$*.insts.bin $*.mlir
	@test -s "$*.xclbin" && test -s "$*.insts.bin"
	$(if $(CONTROLLER_COLUMNS),$(PYTHON) $(BF16_M8192_CONTROLLER_VERIFIER) --columns $(CONTROLLER_COLUMNS) $*.insts.bin)

gemv-bf16-288.ggmlxdna: PACK_ARGS := --kind-id 1 --m 288 --k 288 --weight-bytes 165888 --activation-bytes 576 --output-bytes 1152
gemv-bf16-m8192-k2048.ggmlxdna: PACK_ARGS := --kind-id 1 --m 8192 --k 2048 --weight-bytes 33554432 --activation-bytes 4096 --output-bytes 32768
gemv-bf16-m8192-k3072.ggmlxdna: PACK_ARGS := --kind-id 1 --m 8192 --k 3072 --weight-bytes 50331648 --activation-bytes 6144 --output-bytes 32768
gemv-q4_0-288.ggmlxdna: PACK_ARGS := --kind-id 2 --m 288 --k 288 --weight-bytes 46656 --activation-bytes 576 --output-bytes 1152
gemv-q4_0-m512-k2560.ggmlxdna: PACK_ARGS := --kind-id 2 --m 512 --k 2560 --weight-bytes 737280 --activation-bytes 5120 --output-bytes 2048
gemv-q4_0-m10240-k2560.ggmlxdna: PACK_ARGS := --kind-id 2 --m 10240 --k 2560 --weight-bytes 14745600 --activation-bytes 5120 --output-bytes 40960
gemv-q8_0-m9216-k2560.ggmlxdna: PACK_ARGS := --kind-id 3 --m 9216 --k 2560 --weight-bytes 25067520 --activation-bytes 5120 --output-bytes 36864

%.ggmlxdna: pack-artifact.py %.xclbin %.insts.bin | check-environment
	$(if $(CONTROLLER_COLUMNS),$(PYTHON) $(BF16_M8192_CONTROLLER_VERIFIER) --columns $(CONTROLLER_COLUMNS) $*.insts.bin)
	$(PYTHON) pack-artifact.py --abi-version 1 --architecture aie2p $(PACK_ARGS) \
		--xclbin $*.xclbin --instructions $*.insts.bin --output $@
