# =============================================================================
# TSMM Course Project - cross-platform Makefile
#
# Targets two environments from one source tree:
#   * Local macOS (Apple Silicon): Apple clang + Homebrew libomp + Accelerate.
#   * Intel target cluster        : GCC/ICC + MKL + AVX-512 (use make TARGET=intel).
#
# Usage:
#   make                 # auto-detect platform
#   make TARGET=intel    # force Intel/MKL/AVX-512 build (on the cluster)
#   make run             # build + run benchmark (writes web/results.json)
#   make clean
# =============================================================================

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

SRC      := ops/v0_naive.cpp ops/v1_blocked.cpp ops/v2_openmp.cpp \
            ops/v3_avx512.cpp ops/v4_kreduce.cpp ops/v9_blas.cpp \
            ops/registry.cpp benchmark.cpp
INCLUDE  := -Iinclude
BIN      := tsmm_bench
STD      := -std=c++17
WARN     := -Wall -Wextra -Wno-unused-parameter

# ---- platform selection -----------------------------------------------------
ifeq ($(TARGET),intel)
  # Intel cluster (BSCC bscc-t6): MKL + AVX-512.
  # Supports both icpx (Intel oneAPI, preferred) and g++; the OpenMP flag and
  # the MKL threading layer differ between them, so detect from CXX.
  CXX      ?= icpx
  OPT      := -O3 -march=native -mavx512f -mavx512dq -mfma -funroll-loops
  BLAS_DEF := -DTSMM_BLAS_MKL
  BLAS_INC := -I$(MKLROOT)/include
  ifneq (,$(filter icpx icc icpc,$(notdir $(CXX))))
    # Intel compiler: -qopenmp + Intel OpenMP runtime (iomp5).
    OMP      := -qopenmp
    BLAS_LIB := -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed \
                -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
  else
    # GCC: -fopenmp + GNU OpenMP runtime (gomp).
    OMP      := -fopenmp
    BLAS_LIB := -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed \
                -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
  endif
else ifeq ($(UNAME_S),Darwin)
  # macOS / Apple Silicon: Apple clang + Homebrew libomp + Accelerate.
  CXX      ?= clang++
  BREW     := $(shell brew --prefix 2>/dev/null)
  LIBOMP   := $(BREW)/opt/libomp
  OPT      := -O3 -mcpu=native -funroll-loops
  OMP      := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  OMP_LIB  := -L$(LIBOMP)/lib -lomp
  BLAS_DEF := -DTSMM_BLAS_ACCELERATE
  BLAS_INC :=
  BLAS_LIB := -framework Accelerate
else
  # Generic Linux fallback: GCC + OpenBLAS.
  CXX      ?= g++
  OPT      := -O3 -march=native -funroll-loops
  OMP      := -fopenmp
  BLAS_DEF := -DTSMM_BLAS_OPENBLAS
  BLAS_INC := $(shell pkg-config --cflags openblas 2>/dev/null)
  BLAS_LIB := $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)
endif

CXXFLAGS := $(STD) $(WARN) $(OPT) $(OMP) $(INCLUDE) $(BLAS_INC) $(BLAS_DEF)
LDFLAGS  := $(OMP_LIB) $(BLAS_LIB)

.PHONY: all run run-required clean

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

run: $(BIN)
	mkdir -p web
	./$(BIN) --layout col --out web/results.json

run-required: $(BIN)
	mkdir -p web
	./$(BIN) --layout col --only required --out web/results.json

clean:
	rm -f $(BIN)

# Syntax/codegen check of the AVX-512 intrinsic kernel WITHOUT an Intel machine.
# We are on Apple arm64 where __AVX512F__ is undefined, so the intrinsic branch
# of v3_avx512.cpp normally never gets compiled. This target cross-compiles it
# for an x86-64 AVX-512 target (using the local macOS SDK headers) so we can
# verify the Intel code path actually builds and emits zmm/FMA instructions
# before shipping it to the cluster. Run:  make check-avx512
check-avx512:
	@echo "Cross-compiling v3_avx512.cpp for x86-64 + AVX-512 ..."
	clang++ -std=c++17 -O3 --target=x86_64-apple-darwin \
	    -march=skylake-avx512 -mavx512f -mavx512dq -mfma \
	    -Iinclude -DTSMM_BLAS_ACCELERATE -c ops/v3_avx512.cpp -o /tmp/v3_x86.o
	@echo "OK: AVX-512 intrinsic kernel compiles. zmm instruction count:"
	@otool -tvV /tmp/v3_x86.o | grep -c zmm
