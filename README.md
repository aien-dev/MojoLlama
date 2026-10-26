# MojoLlama

> Archived 2026-10-15: AIEN dropped Mojo/MAX. Read-only history; the eval runners live on in aien-dev/aien-yardsticks (`yardsticks/mojollama-eval-runners`).

<p align="center">
  <img src="www/logo.png" alt="MojoLlama Logo" width="200">
</p>

**High‑throughput CPU LLM inference engine**: GGUF native, MoE-optimized, with a **pure Mojo inference pipeline** and full Studio suite.

![Mojo](https://img.shields.io/badge/Mojo-0.26.2-6C3CE1?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square) ![C](https://img.shields.io/badge/C-AVX2%2B%20OMP-green?style=flat-square) ![GGUF](https://img.shields.io/badge/Format-GGUF-green?style=flat-square) ![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)

---

## ✨ Overview

MojoLlama is a CPU-first inference engine that provides **two engines** for different needs:

### 1. Pure Mojo Engine (NEW)
A **zero-C, zero-Python** inference pipeline written entirely in Mojo. Uses Mojo's SIMD + `parallelize` for all compute kernels with C libc imports only for I/O (`malloc`, `open`, `read`). Features:
- **Q8_0 quantized matmul**: int8→float32 SIMD decode with FMA fusion
- **Batched inference**: batches of 2, 4 or 8 sequences, chosen at compile time (`comptime B`, set to 4 in the checked-in sources)
- **Full architecture support**: GQA attention, RoPE, RMS norm, SiLU activations
- **Running on real models**:
  - **TinyLlama-1.1B** f16: **27.2 tok/s** on one sequence (verified correct output); Q8_0 with a batch of 4: **88.1 tok/s**
  - **ZAYA1-8B** Q8_0 (MoE, 80 layers, 16 experts): **28.6 tok/s** (real English text output)

### 2. C Engine (Performance)
The C engine (`src/mojollama/kernels/cengine_batch_instr.c`) provides AVX2+OMP-accelerated operations. The numbers below were measured through the Python TurboEngine, which calls these C kernels:
- **GPT-OSS-20B** Q4_K_M: **25.5 tok/s** (0.94× llama.cpp)
- **Qwen3.6-35B** MXFP4: **25.5 tok/s** (1.57× llama.cpp) with concurrent serving at **47 tok/s aggregate**

---

## 🚀 Pure Mojo Quick Start

```bash
cd src/mojollama/mojo_engine

# Build and run ZAYA1-8B Q8_0 (28.6 tok/s); the argument is the worker thread count (default 32)
bash build_zaya.sh
LD_LIBRARY_PATH="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib" ./zaya_gen_q8 32

# Build TinyLlama-1.1B Q8_0 with a batch of 4 (88.1 tok/s)
bash build_q8.sh
```

Requirements, from the scripts in `src/mojollama/mojo_engine/`:
- Mojo 0.26.2 and `gcc` (used to link). The build scripts hard-code the Mojo library path shown above, which is where a `uv` tool install of Mojo with Python 3.11 puts it.
- Pre-extracted model files in fixed `/tmp` paths. ZAYA1-8B needs `/tmp/weights_zaya/` and `/tmp/vocab_zaya.bin`, which `extract_zaya_q8.py` and `extract_zaya_vocab.py` write from `/tmp/models/ZAYA1-8B-Q8_0.gguf`. TinyLlama needs `/tmp/weights_tl/` and `/tmp/vocab.bin` (and `/tmp/prompt_tokens.bin` for `tinyllama_benchmark`); see `REAL_BENCHMARK_RESULTS.md` in that folder.
- `tinyllama_benchmark.mojo` (the f16 27.2 tok/s figure) has no build script and takes no arguments. `build.sh` builds a different program (`mojo_engine_v6_final.mojo`) and needs a `read_helpers.c` that is not in the repository, so it does not work as is.

The matmul kernels are pure Mojo (no C bridge); `zaya_gen_q8.mojo` imports `malloc`, `open`, `read`, `lseek` and `close` from libc. See `zaya_gen_q8.mojo` (~700 lines) for the complete reference implementation. The model-extraction steps are Python scripts.

---

## 🏆 Performance Benchmarks

| Model | Engine | Config | tok/s | vs llama.cpp |
|-------|--------|--------|-------|-------------|
| **TinyLlama 1.1B** f16 | **Pure Mojo** | 1 sequence × 32 thr | **27.2** | - |
| **TinyLlama 1.1B** f16 | **Pure Mojo** | B=4 × 32 thr | **77.8** | - |
| **TinyLlama 1.1B** Q8_0 | **Pure Mojo** | B=4 × 32 thr | **88.1** | - |
| **ZAYA1-8B** Q8_0 | **Pure Mojo** | 80L, 16 experts, top-1 | **28.6** 🏆 | - |
| GPT-OSS-20B Q4_K_M | Python TurboEngine | 1 user | 25.5 | **0.94×** |
| Qwen3.6-35B MXFP4 | Python TurboEngine | 1 user | 25.5 | **1.57×** |
| Qwen3.6-35B MXFP4 | C batch_forward | B=10 × 128 prompt | 17.0 | 1.06× |
| Qwen3.6-35B MXFP4 | Concurrent server (spawn) | 10 workers × 3 thr | **47.0 agg** | - |
| TinyLlama 1.1B Q4_0 | Python TurboEngine | 1 user | 79.2 | 0.89× |
| Qwen3-30B-A3B Q4_K_M | Python TurboEngine | 1 user | 22.5 | 0.89× |

**Notes on these numbers.** Hardware is an AMD Threadripper 3970X (32 cores, 251 GB DDR4), 32 threads unless noted (`src/mojollama/README.md`). The 27.2 tok/s TinyLlama f16 row is the aggregate over 7 prompts of 10 generated tokens each. The Pure Mojo rows are recorded in `SUMMARY.md`, `src/mojollama/mojo_engine/TINYLLAMA_QUANT_BENCHMARK.md` and `src/mojollama/mojo_engine/REAL_BENCHMARK_RESULTS.md`. The TinyLlama Q4_0, GPT-OSS, Qwen3.6 TurboEngine and 47 aggregate concurrent figures match `src/mojollama/README.md`. No other file in the repository supports the Qwen3.6 C batch_forward row (17.0), the Qwen3-30B-A3B row (22.5, 0.89×), or the 10 × 1 and 8 × 4 concurrent rows; treat those as unverified. Later records disagree with some rows. `FINAL_STATUS.md` and `.hermes/debug-logs/2026-05-22_mojollama-opt-session.md` (22 May 2026) give GPT-OSS-20B Q4_K_M at 23.6 tok/s (0.86× llama.cpp) and Qwen3-30B Q4_K_M at 22.6 tok/s (0.81×), and list the C batch forward as not working (segfault on Qwen3.6, and a first-token mismatch with the Python path). Re-measure before relying on the TurboEngine and C batch_forward figures.

### Pure Mojo Details

| File | Model | Format | tok/s | Lines |
|------|-------|--------|-------|-------|
| `tinyllama_benchmark.mojo` | TinyLlama-1.1B | f16 | 27.2 | 488 |
| `tinyllama_gen_q8.mojo` | TinyLlama-1.1B | Q8_0 B=4 | 88.1 | 552 |
| `zaya_gen_q8.mojo` | ZAYA1-8B | Q8_0 | **28.6** | **~700** |

**Architecture support (pure Mojo):**
- **Dense models**: TinyLlama (22L, GQA, RoPE, SiLU): proven
- **MoE models**: ZAYA1-8B (80 alt attn/MoE layers, 16 experts, top-1 routing, CCA): proven
- 262K vocab with weight-tying, partial RoPE, learned residual scales: all in Mojo

---

## 🧠 C Engine (`src/mojollama/kernels/cengine_batch_instr.c`)

The C engine is the core performance layer: a single `.c` file (~1,900 lines) with AVX2+OMP-accelerated operations:

**Quantized Matmul Types:**
| Type | Block Size | Operations |
|------|-----------|------------|
| Q4_0 | 32 elems | 4-bit, 18B/block, symmetric |
| Q4_K | 256 elems | K-quant 4-bit, 144B/block |
| Q5_K | 256 elems | K-quant 5-bit, 176B/block |
| Q6_K | 256 elems | K-quant 6-bit, 210B/block |
| Q8_0 | 32 elems | 8-bit, 34B/block |
| **MXFP4** | 32 elems | 4-bit two's complement + E8M0 scale |

**Key Components:**
- `batch_forward()`: single C call for entire forward pass
- `moe_ffn()`: MoE FFN dispatch with per-expert quant matmul
- `ssm_decode_step()`: Mamba-2-like selective scan kernel
- `gqa()`: Grouped Query Attention with softmax + weighted sum
- `PagedAttention`: page table with KV block allocation

---

## 🔧 TurboEngine v7-MoE (`src/mojollama/kernels/turbo_engine_v7_moe.py`)

Zero-allocation Python engine with C-accelerated MoE support:

**Architecture Support:**
- GPT-OSS (24L, 32 experts, MXFP4 gate/up/down)
- Qwen3.6-35B (41L hybrid: 11 attn + 30 SSM, 256 experts, shared expert)
- Qwen3-30B, Qwen2-MoE, Llama, Mistral, Gemma
- Dense models (TinyLlama via `turbo_engine_v77.py`)

---

## 🔄 Concurrent Server

Two approaches for multi-user throughput:

### Multiprocessing spawn (best)
`src/mojollama/benchmarks/server_concurrent_qwen36.py`: workers share weight pages via OS page cache:

| Workers × OMP thr | Per-worker | Aggregate | CPU util |
|-------------------|-----------|-----------|----------|
| 10 × 1 | 3.6 t/s | 36 t/s | 10/32 cores |
| **10 × 3** | **4.7 t/s** | **47 t/s** 🏆 | **30/32 threads** |
| 8 × 4 | 5.7 t/s | 46 t/s | 32/32 threads |

---

## 🗺️ Future Plans

Status below is from the code in this repository.

### Near-term
- **Batch inference** for ZAYA1-8B (B=4, currently B=1): `zaya_gen_q8.mojo` still sets `comptime B = 1`
- **Q8_0 input quantization** in C engine: a Q8_0 by Q8_0 batch matmul exists in `src/mojollama/kernels/quant_kernels_omp.c`; its use across the C engine paths is not confirmed
- **True continuous batching**: merge B users' MoE routing (listed as planned)

### Medium-term
- **MAX GPU backend**: zero-copy tensor dispatch to NVIDIA GPUs. `MAXBackend` in `src/mojollama/backends.py` is documented there as a stub
- **Speculative decoding**: TinyLlama draft + ZAYA target. `src/mojollama/speculative.py` exists, with a TinyLlama draft and a Qwen3-30B-A3B target (not ZAYA)
- **Prefix caching**: LRU with SHA-256 hash. `src/mojollama/prefix_cache.py` implements an LRU cache keyed by a SHA-256 hash; its integration into the serving paths is not confirmed

### Long-term
- **Full GGUF quant pipeline**: imatrix → dynamic 2-bit quantization. `imatrix` and `dynamic-quantize` commands exist in `studio.py`; the lowest type they select is Q2_K (2.5 bits per weight)
- **MojoLlama Studio** as Unsloth+LLaMA-Factory replacement: `studio.py` exists (27 commands, including `train`); it is not shown to replace either tool
- **Perplexity evaluation**: not built (listed in `ROADMAP.md`)

---

## 📚 Project Structure

```
src/mojollama/
├── mojo_engine/             # Pure Mojo inference engines
│   ├── zaya_gen_q8.mojo     ZAYA1-8B Q8_0 (28.6 tok/s)
│   ├── tinyllama_benchmark.mojo  TinyLlama-1.1B f16
│   ├── tinyllama_gen_q8.mojo     TinyLlama-1.1B Q8_0 B=4
│   ├── extract_zaya_q8.py        GGUF→raw Q8_0 .bin
│   ├── extract_zaya_vocab.py     262K vocab extraction
│   └── build_zaya.sh             Build script
├── kernels/                 # C + Python engines
│   ├── cengine_batch_instr.c     C engine (~1,900 lines)
│   ├── turbo_engine_v7_moe.py    Python MoE engine
│   └── zaya_batch_forward.c      ZAYA C reference
├── server.py                API server
├── benchmarks/server_moe.py       MoE server used by the benchmarks
├── studio.py                CLI Studio (27 commands)
└── model/architectures.py   50+ arch detection
```

---

## 📄 License

Apache 2.0
