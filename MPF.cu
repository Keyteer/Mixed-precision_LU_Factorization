#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <lapacke.h>
#include <cublas_v2.h>
#include "fp16_utils.h"
#include "hgetf2_kernel.h"
#include "dgetf2_native_npv.h"
#include "cuda_debug.h"

#define __threads_per_block__ 256

// Quick calculation of blocks needed based on the number of threads needed
int inline grid_size(int threads_needed) {
    return (threads_needed + __threads_per_block__ - 1) / __threads_per_block__;
}

// GPU kernel for FP64 to FP16 conversion
__global__ void double_to_fp16_block(const double *input, fp16 *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = double_to_fp16(input[idx]);
    }
}

// GPU kernel for FP16 to FP64 conversion
__global__ void fp16_to_double_block(const fp16 *input, double *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fp16_to_double(input[idx]);
    }
}

// Efficient device version of LASWP (row swaps, FP64) - Column-major order
// Apply swaps sequentially for each panel column
// A [in/out] pointer to the matrix A
// lda [in] leading dimension of A
// k [in] starting row index for the panel
// cols [in] number of columns in the panel
// ipiv_panel [in] array of pivot indices for the panel (1-based global indexing)
__global__ void LASWP_kernel(double *A, int lda, int k, int cols, const int *ipiv_panel) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (col < lda) {
        // Apply swaps sequentially for this column
        for (int panel_col = 0; panel_col < cols; ++panel_col) {
            int current_row = k + panel_col;              // Current row being processed
            int pivot_row = ipiv_panel[panel_col] - 1;    // Convert to 0-based global index

            if (pivot_row != current_row) {
                // Swap A[col * lda + current_row] <-> A[col * lda + pivot_row]
                double tmp = A[col * lda + current_row];
                A[col * lda + current_row] = A[col * lda + pivot_row];
                A[col * lda + pivot_row] = tmp;
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

// --- MPF: Mixed‑precision Pre‑pivoting Factorization ---
// A [in/out] pointer to the matrix A
// N [in] size of the matrix A (N x N)
// r [in] panel size for mixed-precision factorization
// IPIV [out] array to store pivot indices (1-based global indexing)
void MPF(double *A, int N, int r, int *IPIV) {

    CUDA_CHECK("ENTRY");

    // Check CUDA device availability
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices available." << std::endl;
        return;
    }

    cudaSetDevice(0);  // Explicitly set device


    // Allocate device memory
    double *d_A;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);

    fp16 *d_P_FP16_buffer;
    cudaMalloc(&d_P_FP16_buffer, N * r * sizeof(fp16));

    double *d_P_FP64_NPV_buffer;
    cudaMalloc(&d_P_FP64_NPV_buffer, N * r * sizeof(double));

    int *d_IPIV_panel;
    cudaMalloc(&d_IPIV_panel, r * sizeof(int));

    int *d_IPIV;
    cudaMalloc(&d_IPIV, N * sizeof(int));
    
    // Initialize IPIV to identity permutation (1-based indexing)
    for (int i = 0; i < N; i++) {
        IPIV[i] = i + 1;
    }

    cublasHandle_t handle;
    cublasStatus_t cublasStatus = cublasCreate(&handle);


    // Panel iteration
    for (int k = 0; k < N; k += r) {
        int current_panel_cols = std::min(r, N - k); // Number of columns in the current panel (r or N%r)
        int panel_rows = N - k; // Number of rows in the panel

        if (panel_rows > 1) {

            // 1.1 Extract panel from matrix A to FP64 buffer
            cudaMemcpy2D(d_P_FP64_NPV_buffer, panel_rows * sizeof(double),
                d_A + k * N + k, N * sizeof(double),
                current_panel_cols * sizeof(double), panel_rows,
                cudaMemcpyDeviceToDevice);

            // 1.2 Convert and copy FP64 panel to FP16 panel
            int total_elements = panel_rows * current_panel_cols;
            double_to_fp16_block << <grid_size(total_elements), __threads_per_block__ >> > (d_P_FP64_NPV_buffer, d_P_FP16_buffer, total_elements);
            cudaDeviceSynchronize();



            // 2 Panel LU factorization in FP16
            HGETF2_kernel << <grid_size(panel_rows), __threads_per_block__ >> > (d_P_FP16_buffer, panel_rows, panel_rows, current_panel_cols, d_IPIV_panel);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cout << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "Kernel completed successfully with " << grid_size(panel_rows) << " blocks" << std::endl;
            }



            // 3.1 Apply permutations to FP64 matrix (kernel)
            LASWP_kernel << <grid_size(N), __threads_per_block__ >> > (d_A, N, k, current_panel_cols, d_IPIV_panel);
            cudaDeviceSynchronize();

            // 3.2 Update global IPIV array
            int *h_panel_ipiv = new int[current_panel_cols];
            cudaMemcpy(h_panel_ipiv, d_IPIV_panel, current_panel_cols * sizeof(int), cudaMemcpyDeviceToHost);

            for (int j = 0; j < current_panel_cols; ++j) {
                // h_panel_ipiv[j] is already a global 1-based index from HGETF2_kernel
                // No need to add k again
                IPIV[k + j] = h_panel_ipiv[j] + k;
            }
            delete[] h_panel_ipiv;


            // 4.1 Copy updated panel back for FP64 factorization
            cudaMemcpy2D(d_P_FP64_NPV_buffer, panel_rows * sizeof(double),
                d_A + k * N + k, N * sizeof(double),
                current_panel_cols * sizeof(double), panel_rows,
                cudaMemcpyDeviceToDevice);

            // 4.2 Panel LU factorization in FP64 withot pivoting (kernel)
            dgetf2_native_npv << <grid_size(panel_rows), __threads_per_block__ >> > (panel_rows, current_panel_cols, d_P_FP64_NPV_buffer, panel_rows);
            cudaDeviceSynchronize();

            // 4.3 Copy back the panel to matrix A
            cudaMemcpy2D(d_A + k * N + k, N * sizeof(double),
                d_P_FP64_NPV_buffer, panel_rows * sizeof(double),
                current_panel_cols * sizeof(double), panel_rows,
                cudaMemcpyDeviceToDevice);



            // 5 Trailing submatrix update (cuBLAS)
            if (k + current_panel_cols < N) {
                int m = panel_rows - current_panel_cols;
                int n = N - k - current_panel_cols;
                // 5.1 Solve triangular system (DTRSM)
                DTRSM_cublas(
                    handle,
                    d_A + k * N + k + current_panel_cols,
                    N,
                    d_A + (k + current_panel_cols) * N + k + current_panel_cols,
                    N,
                    m,
                    current_panel_cols
                );
                // 5.2 Update trailing submatrix (DGEMM)
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
    }

    // Copy matrix back to host
    cudaMemcpy(A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Note: IPIV is already updated on host side during panel processing

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_P_FP16_buffer);
    cudaFree(d_P_FP64_NPV_buffer);
    cudaFree(d_IPIV_panel);
    cudaFree(d_IPIV);
}