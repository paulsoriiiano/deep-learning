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