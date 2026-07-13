#!/usr/bin/env bash
# colibrì — installazione su una macchina nuova (Linux x86-64, macOS, Windows/MinGW).
# Compila il motore e fa un self-test. Il MODELLO (~372 GB int4) va copiato a parte
# o rigenerato con: coli convert --model <dir-su-ext4/NVMe>
set -e
cd "$(dirname "$0")"
echo "🐦 colibrì — setup"

UNAME_S=$(uname -s)

# 1) dipendenze
command -v make >/dev/null || { echo "make is missing"; exit 1; }
case "$UNAME_S" in
Darwin)
    command -v clang >/dev/null || { echo "clang is missing (run: xcode-select --install)"; exit 1; }
    echo "  clang: $(clang --version | head -1) · $(sysctl -n hw.ncpu) core"
    echo -n "  OpenMP: "
    if [ -f "$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib" ]; then echo "ok (libomp)"
    else echo "libomp is missing -> single-threaded build (recommended: brew install libomp)"; fi
    ;;
MINGW*|MSYS*)
    command -v gcc  >/dev/null || { echo "gcc is missing (MinGW-w64). Install: pacman -S mingw-w64-x86_64-gcc make"; exit 1; }
    echo "  gcc: $(gcc -dumpversion) · MinGW-w64"
    echo -n "  OpenMP: "; echo 'int main(){return 0;}' | gcc -fopenmp -xc - -o /tmp/_omp 2>/dev/null && echo ok || { echo "libgomp is missing (pacman -S mingw-w64-x86_64-gcc)"; exit 1; }
    ;;
*)
    command -v gcc  >/dev/null || { echo "gcc is missing (for example: sudo apt install build-essential)"; exit 1; }
    echo "  gcc: $(gcc -dumpversion) · $(nproc) core"
    echo -n "  OpenMP: "; echo 'int main(){return 0;}' | gcc -fopenmp -xc - -o /tmp/_omp 2>/dev/null && echo ok || { echo "libgomp is missing"; exit 1; }
    ;;
esac

# 1b) CUDA (solo Linux): se c'è il toolkit E una GPU NVIDIA, compila con CUDA=1.
#     Override manuale: CUDA=0 ./setup.sh (solo CPU) · CUDA=1 ./setup.sh (forza CUDA).
#     Il runtime resta opt-in (COLI_CUDA=1): il binario linkato a CUDA di default gira su CPU.
case "$UNAME_S" in
Darwin|MINGW*|MSYS*) CUDA=0 ;;   # il Makefile rifiuta CUDA=1 fuori da Linux
*)
    if [ -z "${CUDA:-}" ]; then
        NVCC_BIN="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
        [ -x "$NVCC_BIN" ] || NVCC_BIN=$(command -v nvcc || true)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
        if [ -n "$NVCC_BIN" ] && [ -x "$NVCC_BIN" ] && [ -n "$GPU_NAME" ]; then
            CUDA=1
            echo "  CUDA: ok (nvcc $("$NVCC_BIN" --version | grep -oE 'release [0-9.]+' | cut -d' ' -f2), $GPU_NAME)"
        else
            CUDA=0
            echo "  CUDA: not found -> CPU-only build (need nvcc + NVIDIA GPU; override with CUDA=1)"
        fi
    fi
    ;;
esac

# 2) build: nativa (veloce, per QUESTA macchina). Per un binario da distribuire: make portable
echo "  building (ARCH=${ARCH:-native}${CUDA:+ CUDA=$CUDA})…"
make -s glm ARCH="${ARCH:-native}" CUDA="${CUDA:-0}"

# 3) self-test sull'oracolo tiny, se presente
if [ -d glm_tiny ] && [ -f ref_glm.json ]; then
    r=$(SNAP=./glm_tiny TF=1 ./glm 64 16 16 2>/dev/null | grep -oE "[0-9]+/[0-9]+ positions" || true)
    echo "  engine self-test: ${r:-?}  (expected 32/32)"
fi

# 4) info macchina (la velocità dipende da QUESTI due numeri, non dalla GPU)
case "$UNAME_S" in
Darwin)
    ram=$(( $(sysctl -n hw.memsize) / 1000000000 ))
    ;;
MINGW*|MSYS*)
    # MSYS2 fornisce /proc/meminfo come symlink (più affidabile di wmic, deprecato)
    ram=$(awk '/MemTotal/{printf "%.0f", $2/1e6}' /proc/meminfo 2>/dev/null || echo "?")
    ;;
*)
    ram=$(awk '/MemTotal/{printf "%.0f", $2/1e6}' /proc/meminfo 2>/dev/null || echo "?")
    ;;
esac
echo "  RAM: ${ram} GB   (more RAM = more cached experts = faster inference)"
echo
echo "ready. Next steps:"
echo "  ./coli build           # already done"
echo "  ./coli convert --model /path/on/NVMe/glm52_i4     # generate the int4 model (hours)"
echo "  ./coli info  --model /path/on/NVMe/glm52_i4"
echo "  ./coli chat  --model /path/on/NVMe/glm52_i4 --ram <GB>"
if [ "${CUDA:-0}" = "1" ]; then
    echo
    echo "GPU built in but OFF by default. Enable per run with:"
    echo "  COLI_CUDA=1 COLI_GPU=0 ./glm ...          # expert tier on GPU 0 (see README)"
    # Unified-memory hardware (GB10/DGX Spark, Grace): the backend auto-detects it and
    # goes zero-copy — ALL experts GPU-served, nothing duplicated into VRAM.
    case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
    *GB10*|*GB200*|*GH200*|*Grace*)
        echo "  unified memory detected: zero-copy is automatic (COLI_CUDA_UNIFIED=0 disables)"
        echo "  COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 ./glm ...   # recommended: dense part on GPU too"
        ;;
    esac
    echo
    echo "Key runtime variables (full reference: docs/gpu-cuda.md):"
    echo "  COLI_CUDA=1            enable the CUDA backend (off by default)"
    echo "  COLI_GPU=0|COLI_GPUS=0,1,..   device ordinal(s)"
    echo "  CUDA_DENSE=1           dense/attention tensors on GPU"
    echo "  CUDA_EXPERT_GB=<GB>    VRAM hot-expert tier (discrete GPUs; moot on unified memory)"
    echo "  COLI_CUDA_UNIFIED=0|1  force zero-copy mode off/on (default: auto-detect)"
    echo "  PIN=<stats> PIN_GB=<GB> RAM_GB=<GB>   expert pinning / memory budgets"
    echo "  Build: make CUDA=1 [CUDA_ARCH=native|sm_121] [CUDA_HOME=/usr/local/cuda]"
fi
echo
echo "IMPORTANT: keep the model on fast storage (NVMe/ext4), never on /mnt/c or a network mount."
