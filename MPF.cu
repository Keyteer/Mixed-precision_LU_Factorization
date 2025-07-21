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

    cublasHandle_t handle;
    cublasStatus_t cublasStatus = cublasCreate(&handle);

    // Panel iteration
    for (int k = 0; k < N; k += r) {
        int panel_cols = std::min(r, N - k); // Number of columns in the current panel (r or N%r)
        int panel_rows = N - k; // Number of rows in the panel

        if (panel_rows > 1) {

            // 1.1 Extract panel from matrix A to FP64 buffer
            // Copy panel column by column using cudaMemcpy
            for (int col = 0; col < panel_cols; ++col) {
                cudaMemcpy(
                    d_P_FP64_NPV_buffer + col * panel_rows,
                    d_A + (k + col) * N + k,
                    panel_rows * sizeof(double),
                    cudaMemcpyDeviceToDevice
                );
            }
            cudaDeviceSynchronize();
            
            // 1.2 Convert and copy FP64 panel to FP16 panel
            int total_elements = panel_rows * panel_cols;
            double_to_fp16_block << <grid_size(total_elements), __threads_per_block__ >> > (d_P_FP64_NPV_buffer, d_P_FP16_buffer, total_elements);
            cudaDeviceSynchronize();



            // 2 Panel LU factorization in FP16 using Cooperative Groups
            int num_blocks = grid_size(panel_rows);
            int threads_per_block = __threads_per_block__;
            
            void* args[] = {&d_P_FP16_buffer, &panel_rows, &panel_rows, &panel_cols, &d_IPIV_panel};
            
            cudaError_t err = cudaLaunchCooperativeKernel((void*)HGETF2_kernel, 
                                                        dim3(num_blocks), dim3(threads_per_block), 
                                                        args, 0, 0);
            if (err != cudaSuccess) {
                std::cout << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "Kernel completed successfully with " << num_blocks << " blocks" << std::endl;
            }
            
            cudaDeviceSynchronize();



            // 3.2 Update global IPIV array and prepare indices for LASWP
            int *h_panel_ipiv = new int[panel_cols];
            cudaMemcpy(h_panel_ipiv, d_IPIV_panel, panel_cols * sizeof(int), cudaMemcpyDeviceToHost);

            // Convert local panel indices to global indices
            for (int j = 0; j < panel_cols; ++j) {
                // h_panel_ipiv[j] is 1-based local index within panel
                // Convert to global 1-based index for final IPIV output
                IPIV[k + j] = h_panel_ipiv[j] + k;
                // Also convert for LASWP kernel which expects global indices
                h_panel_ipiv[j] = h_panel_ipiv[j] + k;  
            }
            
            // Update the device array with global indices for LASWP
            cudaMemcpy(d_IPIV_panel, h_panel_ipiv, panel_cols * sizeof(int), cudaMemcpyHostToDevice);
            delete[] h_panel_ipiv;

            // 3.1 Apply permutations to FP64 matrix (kernel)
            LASWP_kernel << <grid_size(N), __threads_per_block__ >> > (d_A, N, k, panel_cols, d_IPIV_panel);
            cudaDeviceSynchronize();


            // 4.1 Copy updated panel back for FP64 factorization
            // Copy updated panel from d_A back to d_P_FP64_NPV_buffer column by column
            for (int col = 0; col < panel_cols; ++col) {
                cudaMemcpy(
                    d_P_FP64_NPV_buffer + col * panel_rows,
                    d_A + (k + col) * N + k,
                    panel_rows * sizeof(double),
                    cudaMemcpyDeviceToDevice
                );
            }

            // 4.2 Panel LU factorization in FP64 without pivoting (kernel)
            int num_blocks_dgetf2 = grid_size(panel_rows);
            int threads_per_block_dgetf2 = __threads_per_block__;
            
            void* args_dgetf2[] = {&panel_rows, &panel_cols, &d_P_FP64_NPV_buffer, &panel_rows};
            
            cudaError_t err_dgetf2 = cudaLaunchCooperativeKernel((void*)dgetf2_native_npv, 
                                                              dim3(num_blocks_dgetf2), dim3(threads_per_block_dgetf2), 
                                                              args_dgetf2, 0, 0);
            if (err_dgetf2 != cudaSuccess) {
                std::cout << "CUDA dgetf2 kernel error: " << cudaGetErrorString(err_dgetf2) << std::endl;
            }
            cudaDeviceSynchronize();

            // 4.3 Copy back the panel to matrix A
            // Copy back the panel to matrix A column by column
            for (int col = 0; col < panel_cols; ++col) {
                cudaMemcpy(
                    d_A + (k + col) * N + k,
                    d_P_FP64_NPV_buffer + col * panel_rows,
                    panel_rows * sizeof(double),
                    cudaMemcpyDeviceToDevice
                );
            }
            
            // 5 Trailing submatrix update (cuBLAS)
            if (k + panel_cols < N) {
                int n = N - k - panel_cols;  // Number of columns in trailing matrix
                int m = N - k - panel_cols;  // Number of rows in trailing matrix
                //
                //      |   L11\U11   |   U12   | }-panel_cols              
                //      |     L21     |   A22   | }-n
                //      I--panel_cols-I--- n ---I
                //
                // 5.1 Solve triangular system L21 * U12 = A12 (where A12 is the top-right block)
                // We need to solve L^-1 * A12 = U12, which is equivalent to L * U12 = A12
                // Since L is unit lower triangular, we use triangular solve
                double alpha = 1.0;
                cublasDtrsm(
                    handle,
                    CUBLAS_SIDE_LEFT,           // L is on the left
                    CUBLAS_FILL_MODE_LOWER,     // L is lower triangular
                    CUBLAS_OP_N,                // No transpose of L
                    CUBLAS_DIAG_UNIT,           // Unit diagonal
                    panel_cols, n,              // dimensions: m=panel_cols, n=trailing_cols
                    &alpha,                     // alpha = 1.0
                    d_A + k * N + k, N,         // L11 (panel_cols x panel_cols)
                    d_A + (k + panel_cols) * N + k, N  // A12 -> U12 (panel_cols x n)
                );
                
                // 5.2 Update trailing submatrix A22 = A22 - L21 * U12
                alpha = -1.0; 
                double beta = 1.0;
                cublasDgemm(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,                                      // no transpose L21, no transpose U12
                    m, n, panel_cols,                                              // dimensions: m, n, k
                    &alpha,                                                        // -1.0
                    d_A + k * N + k + panel_cols, N,                               // L21 (m x panel_cols)
                    d_A + (k + panel_cols) * N + k, N,                             // U12 (panel_cols x n)
                    &beta,                                                         // 1.0
                    d_A + (k + panel_cols) * N + k + panel_cols, N                 // A22 (m x n)
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