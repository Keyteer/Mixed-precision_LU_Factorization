#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <lapacke.h>
#include <cublas_v2.h>
#include "hgetf2_kernel.h"

// GPU kernel for FP64 to FP16 conversion
__global__ void double_to_fp16_block(const double* input, fp16* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = double_to_fp16(input[idx]);
    }
}

// GPU kernel for FP16 to FP64 conversion
__global__ void fp16_to_double_block(const fp16* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fp16_to_double(input[idx]);
    }
}

// Device version of DGETF2_NATIVE_NPV (no pivoting, FP64)
__global__ void DGETF2_NATIVE_NPV_kernel(double *panel, int ld, int rows, int cols) {
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
__global__ void LASWP_kernel(double *A, int n, int k, int cols, const int *ipiv_panel) {
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
void DTRSM_cublas(cublasHandle_t handle, double *dA, int lda, double *dB, int ldb, int m, int n) {
    const double alpha = 1.0;
    cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        m, n, &alpha, dA, lda, dB, ldb);
}

// --- DGEMM: Multiplicación de matrices en FP64 usando cuBLAS ---
void DGEMM_cublas(cublasHandle_t handle, double *dA, int lda, double *dB, int ldb, double *dC, int ldc, int m, int n, int k) {
    const double alpha = -1.0;
    const double beta = 1.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
}

// --- MPF: All on GPU ---
void MPF(double *h_A, int N, int r, int *IPIV) {
    // Allocate device memory
    double *d_A;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);

    fp16 *d_P_FP16_buffer;
    cudaMalloc(&d_P_FP16_buffer, N * r * sizeof(fp16));
    double *d_P_FP64_NPV_buffer;
    cudaMalloc(&d_P_FP64_NPV_buffer, N * r * sizeof(double));

    
    int *d_IPIV_panel;
    cudaMalloc(&d_IPIV_panel, r * sizeof(int));
    int ident_ipiv_panel[r]; // Identity permutation for panel
    for (int i = 0; i < r; ++i) {
        ident_ipiv_panel[i] = i + 1; // Initialize to identity permutation
    }
    cudaMemcpy(d_IPIV_panel, ident_ipiv_panel, r * sizeof(int), cudaMemcpyHostToDevice);
    
    
    int *d_IPIV;
    cudaMalloc(&d_IPIV, N * sizeof(int));

    // Initialize IPIV to identity permutation
    int *h_IPIV = new int[N];
    for (int i = 0; i < N; i++) h_IPIV[i] = i + 1; // 1-based indexing
    cudaMemcpy(d_IPIV, h_IPIV, N * sizeof(int), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int k = 0; k < N; k += r) {
        int current_panel_cols = std::min(r, N - k); // Number of columns in the current panel (r or N%r)
        int panel_rows = N - k; // Number of rows in the panel

        // a. Copy panel to FP16 buffer on device
        // First copy to FP64 buffer, then convert to FP16
        cudaMemcpy2D(d_P_FP64_NPV_buffer, panel_rows * sizeof(double),
            d_A + k * N + k, N * sizeof(double),
            current_panel_cols * sizeof(double), panel_rows,
            cudaMemcpyDeviceToDevice);

        // Convert FP64 panel to FP16 (you need a conversion kernel here)
        // For now, do it on host as a temporary solution
        int total_elements = panel_rows * current_panel_cols;
        dim3 block(256);
        dim3 grid((total_elements + 255) / 256);
        double_to_fp16_block<<<grid, block>>>(d_P_FP64_NPV_buffer, d_P_FP16_buffer, total_elements);
        cudaDeviceSynchronize();

        int ipiv_panel_print[r];
        cudaMemcpy(ipiv_panel_print, d_IPIV_panel, r * sizeof(int), cudaMemcpyDeviceToHost);
        // Debug print for IPIV panel
        std::cout << "IPIV panel: ";
        for (int i = 0; i < current_panel_cols; ++i)
            std::cout << ipiv_panel_print[i] << " ";
        std::cout << std::endl;

        // b.i. Panel LU in FP16 (kernel)
        int threads = std::min(1024, panel_rows - 1);
        if (threads > 0) {
            HGETF2_kernel << <1, threads >> > (d_P_FP16_buffer, panel_rows, panel_rows, current_panel_cols, d_IPIV_panel);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(ipiv_panel_print, d_IPIV_panel, r * sizeof(int), cudaMemcpyDeviceToHost);
        // Debug print for IPIV panel
        std::cout << "IPIV panel: ";
        for (int i = 0; i < current_panel_cols; ++i)
            std::cout << ipiv_panel_print[i] << " ";
        std::cout << std::endl;

        // b.ii. Apply permutations to FP64 matrix (kernel)
        LASWP_kernel << <(current_panel_cols + 255) / 256, 256 >> > (d_A, N, k, current_panel_cols, d_IPIV_panel);
        cudaDeviceSynchronize();

        // Update global IPIV array
        int *h_panel_ipiv = new int[current_panel_cols];
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
            DGETF2_NATIVE_NPV_kernel << <1, threads >> > (d_P_FP64_NPV_buffer, panel_rows, panel_rows, current_panel_cols);
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
}