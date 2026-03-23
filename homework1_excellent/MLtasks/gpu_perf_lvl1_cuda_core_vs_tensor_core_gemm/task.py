"""
CUDA Core vs Tensor Core GEMM Benchmark - FC Layer Performance Comparison

Benchmarks the performance difference between:
  - Mode 1 (CUDA Core baseline): FP32 matmul with TF32 disabled
  - Mode 2 (Tensor Core): FP32 matmul with TF32 enabled

Also includes CUDA C++ cuBLAS benchmark code (embedded as strings for reference).
"""

import os
import sys
import json
import time
import subprocess
import textwrap
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Metadata & Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_task_metadata():
    return {
        "task_name": "cuda_core_vs_tensor_core_gemm",
        "task_type": "benchmark",
        "description": (
            "Measure and explain the performance difference between a "
            "'traditional CUDA-core' GEMM path (FP32, TF32 disabled) and a "
            "'modern Tensor Core' GEMM path (FP32 input/output, TF32 compute)."
        ),
        "modes": ["fp32_no_tf32 (CUDA Core baseline)", "fp32_tf32 (Tensor Core)"],
        "metrics": ["latency_ms", "throughput_tflops"],
    }


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires a CUDA-capable GPU. No CUDA device found."
        )
    return torch.device("cuda")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class FCModel(nn.Module):
    """Simple single fully-connected layer — isolates one GEMM operation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.fc(x)


def build_model(in_features: int, out_features: int, device: torch.device) -> FCModel:
    model = FCModel(in_features, out_features).to(device).float()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_forward(
    model: FCModel,
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    """Return average per-iteration latency in milliseconds."""
    # Warm-up
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    # Timed region
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        _ = model(x)
    end_evt.record()
    torch.cuda.synchronize()

    return start_evt.elapsed_time(end_evt) / iters  # ms


def compute_tflops(batch: int, in_f: int, out_f: int, latency_ms: float) -> float:
    """
    FLOPs for one linear (no bias): 2 * batch * in_f * out_f
    (multiply-add counts as 2 ops).
    """
    flops = 2.0 * batch * in_f * out_f
    tflops = flops / (latency_ms * 1e-3) / 1e12
    return tflops


# ─────────────────────────────────────────────────────────────────────────────
# Core Benchmark: two modes
# ─────────────────────────────────────────────────────────────────────────────

MATRIX_SIZES = [
    (512,  512,  512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]   # (batch, in_features, out_features)

WARMUP_ITERS = 20
BENCH_ITERS  = 100


def run_benchmark(device: torch.device, sizes=MATRIX_SIZES):
    """
    Returns a dict with results for both modes across all sizes.
    """
    results = {
        "sizes": [],
        "fp32_no_tf32": {"latency_ms": [], "tflops": []},
        "fp32_tf32":    {"latency_ms": [], "tflops": []},
    }

    print(f"\n{'Size (B×M×N)':<22} {'FP32 (ms)':>12} {'TF32 (ms)':>12} "
          f"{'FP32 TFLOPS':>14} {'TF32 TFLOPS':>14} {'Speedup':>10}")
    print("-" * 90)

    for (batch, in_f, out_f) in sizes:
        label = f"{batch}×{in_f}×{out_f}"
        x     = torch.randn(batch, in_f, device=device, dtype=torch.float32)

        # ── Mode 1: CUDA Core baseline (TF32 disabled) ──────────────────────
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32       = False
        model_fp32 = build_model(in_f, out_f, device)
        lat_fp32   = _time_forward(model_fp32, x, WARMUP_ITERS, BENCH_ITERS)
        tflops_fp32 = compute_tflops(batch, in_f, out_f, lat_fp32)

        # ── Mode 2: Tensor Core (TF32 enabled) ──────────────────────────────
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        model_tf32 = build_model(in_f, out_f, device)
        lat_tf32   = _time_forward(model_tf32, x, WARMUP_ITERS, BENCH_ITERS)
        tflops_tf32 = compute_tflops(batch, in_f, out_f, lat_tf32)

        speedup = lat_fp32 / lat_tf32

        results["sizes"].append(label)
        results["fp32_no_tf32"]["latency_ms"].append(round(lat_fp32,  4))
        results["fp32_no_tf32"]["tflops"].append(round(tflops_fp32, 4))
        results["fp32_tf32"]["latency_ms"].append(round(lat_tf32,   4))
        results["fp32_tf32"]["tflops"].append(round(tflops_tf32, 4))

        print(f"{label:<22} {lat_fp32:>12.4f} {lat_tf32:>12.4f} "
              f"{tflops_fp32:>14.4f} {tflops_tf32:>14.4f} {speedup:>10.2f}x")

    # Restore TF32 to default (True) after benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    return results


# ─────────────────────────────────────────────────────────────────────────────
# make_dataloaders / train / evaluate / predict stubs
# (required by pytorch_task_v1 protocol; benchmark task does not use them)
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(**kwargs):
    """Not used for benchmarking; stub for protocol compliance."""
    return None, None, None, None


def train(model, *args, **kwargs):
    """Not used for benchmarking; stub for protocol compliance."""
    return {}


def evaluate(model, data_loader, **kwargs):
    """Not used for benchmarking; stub for protocol compliance."""
    return {}


def predict(model, data_loader):
    """Not used for benchmarking; stub for protocol compliance."""
    return np.array([]), np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, output_dir: str):
    sizes    = results["sizes"]
    x_idx    = np.arange(len(sizes))
    bar_w    = 0.35

    lat_fp32  = results["fp32_no_tf32"]["latency_ms"]
    lat_tf32  = results["fp32_tf32"]["latency_ms"]
    tflops_fp32 = results["fp32_no_tf32"]["tflops"]
    tflops_tf32 = results["fp32_tf32"]["tflops"]
    speedups  = [a / b for a, b in zip(lat_fp32, lat_tf32)]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("CUDA Core (FP32) vs Tensor Core (TF32) — FC Layer GEMM Benchmark",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ── 1. Latency bar chart ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x_idx - bar_w/2, lat_fp32,  bar_w, label="FP32 / CUDA Core",  color="#4C72B0")
    ax1.bar(x_idx + bar_w/2, lat_tf32,  bar_w, label="TF32 / Tensor Core", color="#DD8452")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(sizes, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Average Forward-Pass Latency")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # ── 2. Throughput bar chart ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x_idx - bar_w/2, tflops_fp32, bar_w, label="FP32 / CUDA Core",  color="#4C72B0")
    ax2.bar(x_idx + bar_w/2, tflops_tf32, bar_w, label="TF32 / Tensor Core", color="#DD8452")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(sizes, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Throughput (TFLOPS)")
    ax2.set_title("Compute Throughput")
    ax2.legend(fontsize=8)

    # ── 3. Speedup line chart ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x_idx, speedups, marker="o", color="#55A868", linewidth=2, markersize=8)
    ax3.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(sizes, rotation=25, ha="right", fontsize=8)
    ax3.set_ylabel("Speedup (FP32 lat / TF32 lat)")
    ax3.set_title("TF32 Tensor Core Speedup over FP32 CUDA Core")
    for i, s in enumerate(speedups):
        ax3.annotate(f"{s:.2f}x", (x_idx[i], speedups[i]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#55A868")

    # ── 4. Latency line chart (both modes) ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x_idx, lat_fp32,  marker="s", label="FP32 / CUDA Core",  color="#4C72B0", linewidth=2)
    ax4.plot(x_idx, lat_tf32,  marker="o", label="TF32 / Tensor Core", color="#DD8452", linewidth=2)
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(sizes, rotation=25, ha="right", fontsize=8)
    ax4.set_ylabel("Latency (ms)")
    ax4.set_title("Latency Scaling with Matrix Size")
    ax4.legend(fontsize=8)
    ax4.set_yscale("log")

    path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# cuBLAS CUDA C++ reference code (embedded for reference / compilation)
# ─────────────────────────────────────────────────────────────────────────────

CUBLAS_BENCHMARK_CODE = r"""
// cublas_gemm_benchmark.cu
// Compile:
//   nvcc -O3 -arch=sm_80 cublas_gemm_benchmark.cu -lcublas -o cublas_gemm_bench
//
// Benchmarks:
//   Baseline : cublasSgemm  (FP32 inputs, FP32 math, NO Tensor Cores)
//   TF32 path: cublasGemmEx (FP32 inputs/outputs, TF32 compute via Tensor Cores)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define CHECK_CUBLAS(call)                                                \
    do {                                                                  \
        cublasStatus_t st = (call);                                       \
        if (st != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                  \
                    __FILE__, __LINE__, (int)st);                         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// Time a cuBLAS GEMM kernel (GPU timing via CUDA events)
// Returns average latency in milliseconds.
double time_gemm(
    cublasHandle_t handle,
    bool use_tf32,
    int M, int N, int K,
    const float* A, const float* B, float* C,
    int warmup, int iters)
{
    const float alpha = 1.0f, beta = 0.0f;

    // Set math mode
    cublasSetMathMode(handle,
        use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);

    // Warm-up
    for (int i = 0; i < warmup; ++i) {
        if (!use_tf32) {
            // Baseline: plain FP32
            CHECK_CUBLAS(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B, N, A, K, &beta, C, N));
        } else {
            // TF32 path via GemmEx
            CHECK_CUBLAS(cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B, CUDA_R_32F, N,
                A, CUDA_R_32F, K,
                &beta,
                C, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed region
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i) {
        if (!use_tf32) {
            CHECK_CUBLAS(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B, N, A, K, &beta, C, N));
        } else {
            CHECK_CUBLAS(cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B, CUDA_R_32F, N,
                A, CUDA_R_32F, K,
                &beta,
                C, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return (double)ms / iters;
}

int main()
{
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const int WARMUP = 20;
    const int ITERS  = 100;

    // Matrix sizes: M=batch, K=in_features, N=out_features
    struct Size { int M, K, N; };
    std::vector<Size> sizes = {
        { 512,  512,  512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
    };

    printf("%-22s %12s %12s %14s %14s %10s\n",
           "Size (M x K x N)", "FP32(ms)", "TF32(ms)",
           "FP32 TFLOPS", "TF32 TFLOPS", "Speedup");
    printf("%s\n", std::string(90, '-').c_str());

    for (auto& s : sizes) {
        long long total = (long long)s.M * s.K + (long long)s.K * s.N
                        + (long long)s.M * s.N;
        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, (long long)s.M * s.K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dB, (long long)s.K * s.N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dC, (long long)s.M * s.N * sizeof(float)));
        // Initialize with random data
        // (omitted for brevity – use cuRAND or host-side fill)

        double lat_fp32 = time_gemm(handle, false, s.M, s.N, s.K,
                                    dA, dB, dC, WARMUP, ITERS);
        double lat_tf32 = time_gemm(handle, true,  s.M, s.N, s.K,
                                    dA, dB, dC, WARMUP, ITERS);

        double flops = 2.0 * s.M * s.K * s.N;
        double tflops_fp32 = flops / (lat_fp32 * 1e-3) / 1e12;
        double tflops_tf32 = flops / (lat_tf32 * 1e-3) / 1e12;

        printf("%-22s %12.4f %12.4f %14.4f %14.4f %10.2fx\n",
               "", lat_fp32, lat_tf32, tflops_fp32, tflops_tf32,
               lat_fp32 / lat_tf32);

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
"""


def save_cublas_code(output_dir: str):
    path = os.path.join(output_dir, "cublas_gemm_benchmark.cu")
    with open(path, "w") as f:
        f.write(CUBLAS_BENCHMARK_CODE.strip())
    print(f"  Saved cuBLAS C++ benchmark → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# save_artifacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(results: dict, plot_path: str, output_dir: str = "./output"):
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics  → {metrics_path}")

    # Save cuBLAS C++ code
    save_cublas_code(output_dir)

    return metrics_path


# ─────────────────────────────────────────────────────────────────────────────
# GPU info helper
# ─────────────────────────────────────────────────────────────────────────────

def print_gpu_info():
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"  Compute Capability : {props.major}.{props.minor}")
    print(f"  Total Memory       : {props.total_memory / 1e9:.1f} GB")
    print(f"  SM Count           : {props.multi_processor_count}")
    has_tc = props.major >= 7  # Volta+ has Tensor Cores
    print(f"  Tensor Cores       : {'Yes (Volta+)' if has_tc else 'No (pre-Volta)'}")
    if not has_tc:
        print("\n  WARNING: This GPU predates Tensor Core hardware (Volta, sm_70+).")
        print("  TF32 mode will fall back to FP32 and speedup will be ~1x.")
    return has_tc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CUDA Core vs Tensor Core GEMM Benchmark")
    print("=" * 70)

    meta = get_task_metadata()
    print(f"\nTask : {meta['task_name']}")
    print(f"Desc : {meta['description']}")

    # Device
    device = get_device()
    has_tc = print_gpu_info()

    set_seed(42)
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # ── Run PyTorch benchmark ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PyTorch Benchmark  (torch.cuda.Event timing, GPU-side)")
    print(f"Warm-up iterations : {WARMUP_ITERS}")
    print(f"Bench  iterations  : {BENCH_ITERS}")
    print("─" * 70)

    results = run_benchmark(device)

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_path = plot_results(results, output_dir)

    # ── Save artefacts ────────────────────────────────────────────────────
    print("\nSaving artefacts...")
    save_artifacts(results, plot_path, output_dir)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    avg_speedup = np.mean(
        [a / b for a, b in zip(
            results["fp32_no_tf32"]["latency_ms"],
            results["fp32_tf32"]["latency_ms"]
        )]
    )
    peak_tflops_fp32 = max(results["fp32_no_tf32"]["tflops"])
    peak_tflops_tf32 = max(results["fp32_tf32"]["tflops"])

    print(f"  Average TF32 speedup over FP32 baseline : {avg_speedup:.2f}x")
    print(f"  Peak CUDA Core throughput (FP32)         : {peak_tflops_fp32:.3f} TFLOPS")
    print(f"  Peak Tensor Core throughput (TF32)       : {peak_tflops_tf32:.3f} TFLOPS")

    # ── Quality checks ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QUALITY CHECKS")
    print("=" * 70)

    checks_passed = True

    # Check 1: Benchmark completed for all sizes
    check1 = len(results["sizes"]) == len(MATRIX_SIZES)
    print(f"{'✓' if check1 else '✗'} All {len(MATRIX_SIZES)} matrix sizes benchmarked")
    checks_passed &= check1

    # Check 2: TF32 throughput >= FP32 throughput on average (for Tensor Core GPUs)
    mean_tf32 = np.mean(results["fp32_tf32"]["tflops"])
    mean_fp32 = np.mean(results["fp32_no_tf32"]["tflops"])
    check2 = (not has_tc) or (mean_tf32 >= mean_fp32)
    print(f"{'✓' if check2 else '✗'} TF32 throughput >= FP32 throughput "
          f"({mean_tf32:.3f} vs {mean_fp32:.3f} TFLOPS)")
    checks_passed &= check2

    # Check 3: Average speedup > 1 (or GPU has no Tensor Cores)
    check3 = (not has_tc) or (avg_speedup > 1.0)
    print(f"{'✓' if check3 else '✗'} TF32 speedup > 1.0x  ({avg_speedup:.2f}x)")
    checks_passed &= check3

    # Check 4: Largest size latency recorded sanely (> 0)
    check4 = results["fp32_tf32"]["latency_ms"][-1] > 0
    print(f"{'✓' if check4 else '✗'} Valid latency values recorded")
    checks_passed &= check4

    print("\n" + "=" * 70)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 70)

    return 0 if checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
