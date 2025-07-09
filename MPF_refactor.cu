#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <lapacke.h>
#include <cublas_v2.h>

using fp16 = __half;

__device__ inline void swap_fp16(fp16 &a, fp16 &b) {
    fp16 tmp = a;
    a = b;
    b = tmp;
}

fp16 double_to_fp16(double x) {
    float xf = static_cast<float>(x);
    constexpr float FP16_MAX = 65504.0f;
    constexpr float FP16_MIN_POS = 6.10352e-05f;
    if (xf > FP16_MAX) xf = FP16_MAX;
    else if (xf < -FP16_MAX) xf = -FP16_MAX;
    if (xf > -FP16_MIN_POS && xf < FP16_MIN_POS) xf = 0.0f;
    return __float2half_rn(xf);
}

double fp16_to_double(fp16 x) {
    return static_cast<double>(__half2float(x));
}

// CUDA kernel for HGETF2 (panel LU in FP16)
__global__ void HGETF2_kernel(fp16 *panel, int ld, int rows, int cols, int *ipiv_panel) {
    int tid = threadIdx.x;
    for (int j = 0; j < cols; ++j) {
        int piv = j;
        fp16 maxval = __habs(panel[j * ld + j]);
        if (tid == 0) {
            for (int i = j + 1; i < rows; ++i) {
                fp16 val = __habs(panel[j * ld + i]);
                if (__half2float(val) > __half2float(maxval)) {
                    maxval = val;
                    piv = i;
                }
            }
            ipiv_panel[j] = piv;
        }
        __syncthreads();
        if (tid == 0 && piv != j) {
            for (int k = 0; k < cols; ++k)
                swap_fp16(panel[k * ld + j], panel[k * ld + piv]);
        }
        __syncthreads();
        int i = j + 1 + tid;
        if (i < rows) {
            fp16 lij = __hdiv(panel[j * ld + i], panel[j * ld + j]);
            panel[j * ld + i] = lij;
            for (int k = j + 1; k < cols; ++k) {
                fp16 a = panel[k * ld + i];
                fp16 b = panel[k * ld + j];
                panel[k * ld + i] = __hsub(a, __hmul(b, lij));
            }
        }
        __syncthreads();
    }
}

// Device version of DGETF2_NATIVE_NPV (no pivoting, FP64)
__global__ void DGETF2_NATIVE_NPV_kernel(double* panel, int ld, int rows, int cols) {
    for (int j = 0; j < cols; ++j) {
        int i = threadIdx.x + j + 1;
        if (i < rows) {
            panel[j * ld + i] /= panel[j * ld + j];
            for (int k = j + 1; k < cols; ++k)
                panel[k * ld + i] -= panel[k * ld + j] * panel[j * ld + i];
        }
        __syncthreads();
    }
}

// Device version of LASWP (row swaps, FP64)
__global__ void LASWP_kernel(double* A, int n, int k, int cols, const int* ipiv_panel) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        int piv = ipiv_panel[j];
        if (piv != j) {
            for (int i = 0; i < n; ++i) {
                double tmp = A[j * n + i + k * n];
                A[j * n + i + k * n] = A[piv * n + i + k * n];
                A[piv * n + i + k * n] = tmp;
            }
        }
    }
}

// --- DTRSM: Triangular solve en FP64 usando cuBLAS ---
void DTRSM_cublas(cublasHandle_t handle, double* dA, int lda, double* dB, int ldb, int m, int n) {
    const double alpha = 1.0;
    cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                m, n, &alpha, dA, lda, dB, ldb);
}

// --- DGEMM: Multiplicación de matrices en FP64 usando cuBLAS ---
void DGEMM_cublas(cublasHandle_t handle, double* dA, int lda, double* dB, int ldb, double* dC, int ldc, int m, int n, int k) {
    const double alpha = -1.0;
    const double beta = 1.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
}

// --- MPF: All on GPU ---
void MPF(double* h_A, int N, int r, std::vector<int>& IPIV) {
    // Allocate device memory
    double* d_A;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);

    fp16* d_P_FP16_buffer;
    cudaMalloc(&d_P_FP16_buffer, N * r * sizeof(fp16));
    double* d_P_FP64_NPV_buffer;
    cudaMalloc(&d_P_FP64_NPV_buffer, N * r * sizeof(double));
    int* d_IPIV_panel;
    cudaMalloc(&d_IPIV_panel, r * sizeof(int));
    int* d_IPIV;
    cudaMalloc(&d_IPIV, N * sizeof(int));

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int k = 0; k < N; k += r) {
        int current_panel_cols = std::min(r, N - k);
        int panel_rows = N - k;

        // a. Copy panel to FP16 (kernel or thrust::transform for efficiency)
        // (for simplicity, do it on host and copy, but for full GPU, write a kernel)
        std::vector<fp16> h_panel(panel_rows * current_panel_cols);
        cudaMemcpy2D(h_panel.data(), panel_rows * sizeof(fp16),
                     d_A + k * N + k, N * sizeof(double),
                     current_panel_cols * sizeof(double), panel_rows,
                     cudaMemcpyDeviceToHost);
        for (int j = 0; j < current_panel_cols; ++j)
            for (int i = 0; i < panel_rows; ++i)
                h_panel[j * panel_rows + i] = double_to_fp16(h_A[(k + j) * N + (k + i)]);
        cudaMemcpy(d_P_FP16_buffer, h_panel.data(), panel_rows * current_panel_cols * sizeof(fp16), cudaMemcpyHostToDevice);

        // b.i. Panel LU in FP16 (kernel)
        int threads = std::min(1024, panel_rows - 1);
        HGETF2_kernel<<<1, threads>>>(d_P_FP16_buffer, panel_rows, panel_rows, current_panel_cols, d_IPIV_panel);
        cudaDeviceSynchronize();

        // b.ii. Apply permutations to FP64 matrix (kernel)
        LASWP_kernel<<<(current_panel_cols + 255) / 256, 256>>>(d_A, N, k, current_panel_cols, d_IPIV_panel);
        cudaDeviceSynchronize();

        // b.iii. Panel LU in FP64 (no pivoting, kernel)
        DGETF2_NATIVE_NPV_kernel<<<1, threads>>>(d_P_FP64_NPV_buffer, panel_rows, panel_rows, current_panel_cols);
        cudaDeviceSynchronize();

        // Copy back the panel to d_A
        cudaMemcpy2D(d_A + k * N + k, N * sizeof(double),
                     d_P_FP64_NPV_buffer, panel_rows * sizeof(double),
                     current_panel_cols * sizeof(double), panel_rows,
                     cudaMemcpyDeviceToDevice);

        // c. Trailing submatrix update (cuBLAS)
        if (k + current_panel_cols < N) {
            int m = panel_rows - current_panel_cols;
            int n = N - k - current_panel_cols;
            DTRSM_cublas(
                handle,
                d_A + k * N + k + current_panel_cols,
                N,
                d_A + (k + current_panel_cols) * N + k + current_panel_cols,
                N,
                m,
                current_panel_cols
            );
            DGEMM_cublas(
                handle,
                d_A + (k + current_panel_cols) * N + k + current_panel_cols,
                N,
                d_A + (k + current_panel_cols) * N + k,
                N,
                d_A + k * N + k + current_panel_cols,
                N,
                m,
                n,
                current_panel_cols
            );
        }
    }

    // Copy result back to host
    cudaMemcpy(h_A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    IPIV.resize(N);
    cudaMemcpy(IPIV.data(), d_IPIV, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_P_FP16_buffer);
    cudaFree(d_P_FP64_NPV_buffer);
    cudaFree(d_IPIV_panel);
    cudaFree(d_IPIV);
    cublasDestroy(handle);
}