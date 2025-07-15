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
        // Use fabsf(__half2float(x)) instead of __habs(x)
        fp16 maxval = __float2half(fabsf(__half2float(panel[j * ld + j])));
        if (tid == 0) {
            for (int i = j + 1; i < rows; ++i) {
                fp16 val = __float2half(fabsf(__half2float(panel[j * ld + i])));
                if (val > maxval) {
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
            // Use operator overloads instead of intrinsics
            fp16 lij = panel[j * ld + i] / panel[j * ld + j];
            panel[j * ld + i] = lij;
            for (int k = j + 1; k < cols; ++k) {
                fp16 a = panel[k * ld + i];
                fp16 b = panel[k * ld + j];
                panel[k * ld + i] = a - b * lij;
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
void MPF(double* h_A, int N, int r, int *IPIV) {
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

    // Initialize IPIV to identity permutation
    int* h_IPIV = new int[N];
    for (int i = 0; i < N; i++) h_IPIV[i] = i + 1; // 1-based indexing
    cudaMemcpy(d_IPIV, h_IPIV, N * sizeof(int), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int k = 0; k < N; k += r) {
        int current_panel_cols = std::min(r, N - k);
        int panel_rows = N - k;

        // a. Copy panel to FP16 buffer on device
        // First copy to FP64 buffer, then convert to FP16
        cudaMemcpy2D(d_P_FP64_NPV_buffer, panel_rows * sizeof(double),
                     d_A + k * N + k, N * sizeof(double),
                     current_panel_cols * sizeof(double), panel_rows,
                     cudaMemcpyDeviceToDevice);

        // Convert FP64 panel to FP16 (you need a conversion kernel here)
        // For now, do it on host as a temporary solution
        double* h_panel_fp64 = new double[panel_rows * current_panel_cols];
        cudaMemcpy(h_panel_fp64, d_P_FP64_NPV_buffer, 
                   panel_rows * current_panel_cols * sizeof(double), 
                   cudaMemcpyDeviceToHost);
        
        fp16* h_panel_fp16 = new fp16[panel_rows * current_panel_cols];
        for (int i = 0; i < panel_rows * current_panel_cols; ++i) {
            h_panel_fp16[i] = double_to_fp16(h_panel_fp64[i]);
        }
        cudaMemcpy(d_P_FP16_buffer, h_panel_fp16, 
                   panel_rows * current_panel_cols * sizeof(fp16), 
                   cudaMemcpyHostToDevice);

        // b.i. Panel LU in FP16 (kernel)
        int threads = std::min(1024, panel_rows - 1);
        if (threads > 0) {
            HGETF2_kernel<<<1, threads>>>(d_P_FP16_buffer, panel_rows, panel_rows, current_panel_cols, d_IPIV_panel);
            cudaDeviceSynchronize();
        }

        // b.ii. Apply permutations to FP64 matrix (kernel)
        LASWP_kernel<<<(current_panel_cols + 255) / 256, 256>>>(d_A, N, k, current_panel_cols, d_IPIV_panel);
        cudaDeviceSynchronize();

        // Update global IPIV array
        int* h_panel_ipiv = new int[current_panel_cols];
        cudaMemcpy(h_panel_ipiv, d_IPIV_panel, current_panel_cols * sizeof(int), cudaMemcpyDeviceToHost);
        for (int j = 0; j < current_panel_cols; ++j) {
            h_IPIV[k + j] = k + h_panel_ipiv[j];
        }

        // b.iii. Copy updated panel back for FP64 factorization
        cudaMemcpy2D(d_P_FP64_NPV_buffer, panel_rows * sizeof(double),
                     d_A + k * N + k, N * sizeof(double),
                     current_panel_cols * sizeof(double), panel_rows,
                     cudaMemcpyDeviceToDevice);

        // Panel LU in FP64 (no pivoting, kernel)
        if (threads > 0) {
            DGETF2_NATIVE_NPV_kernel<<<1, threads>>>(d_P_FP64_NPV_buffer, panel_rows, panel_rows, current_panel_cols);
            cudaDeviceSynchronize();
        }

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
    memcpy(IPIV, h_IPIV, N * sizeof(int));

    // Cleanup
    delete[] h_IPIV;
    cudaFree(d_A);
    cudaFree(d_P_FP16_buffer);
    cudaFree(d_P_FP64_NPV_buffer);
    cudaFree(d_IPIV_panel);
    cudaFree(d_IPIV);
    cublasDestroy(handle);
}