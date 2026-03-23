# CUDA Core vs Tensor Core GEMM Benchmark

## Overview

This project benchmarks the performance difference between two GPU matrix multiplication paths for a fully connected layer implemented in PyTorch: a baseline FP32 path with TF32 disabled and a Tensor Core path with TF32 enabled. The task measures average latency and throughput across several square GEMM sizes and saves plots, metrics, and a reference cuBLAS benchmark source file. The implementation follows the `pytorch_task_v1` protocol structure, with benchmark-specific stubs for training-related functions. 

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- NumPy
- Matplotlib
- NVIDIA GPU with CUDA support

## Task

### Task: FC Layer GEMM Benchmark

- **Task type:** Benchmark
- **Task name:** `cuda_core_vs_tensor_core_gemm`
- **Description:** Compare a traditional CUDA-core-style FP32 GEMM path against a Tensor Core TF32 GEMM path using the same single fully connected layer workload.
- **Modes:**
  - `fp32_no_tf32` — FP32 matmul with TF32 disabled
  - `fp32_tf32` — FP32 input/output with TF32-enabled compute on Tensor Cores
- **Metrics:**
  - `latency_ms`
  - `throughput_tflops`

### Benchmark Configuration

The benchmark evaluates these matrix sizes:

- `512 x 512 x 512`
- `1024 x 1024 x 1024`
- `2048 x 2048 x 2048`
- `4096 x 4096 x 4096`
- `8192 x 8192 x 8192`

Each configuration uses:

- `20` warm-up iterations
- `100` timed iterations

## Protocol

This task follows the `pytorch_task_v1` style layout with:

- Standardized metadata through `get_task_metadata()`
- A clearly defined model via `FCModel`
- Benchmark execution through `run_benchmark()`
- Stubbed `make_dataloaders`, `train`, `evaluate`, and `predict` functions for protocol compliance
- Self-verifiable exit codes via `sys.exit(...)`

## Outputs

Running the benchmark creates an `output/` directory containing:

- `benchmark_comparison.png` — comparison plots for latency, throughput, and speedup
- `metrics.json` — serialized benchmark results
- `cublas_gemm_benchmark.cu` — reference CUDA C++ cuBLAS benchmark implementation

## Usage

```bash
python task.py
```

## Exit Codes

- `0`: Success (all quality checks passed)
- `1`: Failure (one or more quality checks failed, or runtime error)

## Notes

- A CUDA-capable GPU is required.
- Tensor Core acceleration is expected on Volta-generation GPUs or newer.
- On pre-Volta GPUs, TF32/Tensor Core speedup may be negligible or unavailable.
