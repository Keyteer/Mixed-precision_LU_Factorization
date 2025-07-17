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

// Device version of LASWP (row swaps, FP64) - Column-major order
__global__ void LASWP_kernel(double *A, int lda, int k, int cols, const int *ipiv_panel) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        int piv = ipiv_panel[j] - 1; // Convert from 1-based to 0-based
        if (piv != j) {
            // Swap rows j+k and piv+k for all columns (column-major)
            for (int col = 0; col < lda; ++col) {
                double tmp = A[col * lda + (j + k)];
                A[col * lda + (j + k)] = A[col * lda + (piv + k)];
                A[col * lda + (piv + k)] = tmp;
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
// h_A [in] 
void MPF(double *A, int N, int r, int *IPIV) {
    
    CUDA_CHECK("ENTRY");
    
    // Check CUDA device availability
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA GetDeviceCount status: " << cudaGetErrorString(cudaStatus) << std::endl;
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return;
    }
    
    CUDA_CHECK("AFTER_DEVICE_COUNT");
    
    cudaSetDevice(0);  // Explicitly set device
    CUDA_CHECK("AFTER_SET_DEVICE");
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using device: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    
    CUDA_CHECK("AFTER_DEVICE_PROPERTIES");
    
    // Initialize host memory
    double *h_A = new double[N * N];
    std::memcpy(h_A, A, N * N * sizeof(double));

    CUDA_CHECK("AFTER_HOST_ALLOC");

    // Allocate device memory
    double *d_A;
    cudaMalloc(&d_A, N * N * sizeof(double));
    CUDA_CHECK("AFTER_D_A_MALLOC");
    
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK("AFTER_D_A_MEMCPY");

    fp16 *d_P_FP16_buffer;
    cudaMalloc(&d_P_FP16_buffer, N * r * sizeof(fp16));
    CUDA_CHECK("AFTER_FP16_MALLOC");
    
    double *d_P_FP64_NPV_buffer;
    cudaMalloc(&d_P_FP64_NPV_buffer, N * r * sizeof(double));
    CUDA_CHECK("AFTER_FP64_MALLOC");

    int *d_IPIV_panel;
    cudaMalloc(&d_IPIV_panel, r * sizeof(int));
    CUDA_CHECK("AFTER_IPIV_PANEL_MALLOC");
    
    int *d_IPIV;
    cudaMalloc(&d_IPIV, N * sizeof(int));
    CUDA_CHECK("AFTER_IPIV_MALLOC");



    // Initialize IPIV to identity permutation
    int *h_IPIV = new int[N];
    for (int i = 0; i < N; i++) h_IPIV[i] = i + 1; // 1-based indexing
    cudaMemcpy(d_IPIV, h_IPIV, N * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK("AFTER_IPIV_INIT");

    // Check CUDA context after memory operations
    cudaError_t contextCheck = cudaGetLastError();
    std::cout << "CUDA context after memory ops: " << cudaGetErrorString(contextCheck) << std::endl;

    CUDA_CHECK("BEFORE_CUBLAS_CREATE");
    cublasHandle_t handle;
    cublasStatus_t cublasStatus = cublasCreate(&handle);
    std::cout << "cuBLAS create status: " << cublasStatus << std::endl;
    CUDA_CHECK("AFTER_CUBLAS_CREATE");

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

        // Initialize IPIV panel to identity permutation for this panel
        int ident_ipiv_panel[r];
        for (int i = 0; i < current_panel_cols; i++) ident_ipiv_panel[i] = i + 1; // 1-based indexing
        cudaMemcpy(d_IPIV_panel, ident_ipiv_panel, current_panel_cols * sizeof(int), cudaMemcpyHostToDevice);

        int ipiv_panel_print[r];
        // Initialize the print array to avoid garbage values
        for (int i = 0; i < r; i++) ipiv_panel_print[i] = 0;
        cudaMemcpy(ipiv_panel_print, d_IPIV_panel, current_panel_cols * sizeof(int), cudaMemcpyDeviceToHost);
        // Debug print for IPIV panel
        std::cout << "IPIV panel: ";
        for (int i = 0; i < current_panel_cols; ++i)
            std::cout << ipiv_panel_print[i] << " ";
        std::cout << std::endl;

        // b.i. Panel LU in FP16 (kernel)
        int threads = std::min(1024, panel_rows - 1);
        if (threads > 0) {
            std::cout << "Launching HGETF2_kernel with " << threads << " threads" << std::endl;
            HGETF2_kernel << <1, threads >> > (d_P_FP16_buffer, panel_rows, panel_rows, current_panel_cols, d_IPIV_panel);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cout << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
                std::cout << "Using CPU fallback for IPIV calculation" << std::endl;
                
                // CPU fallback: simple pivoting logic
                int *h_fallback_ipiv = new int[current_panel_cols];
                for (int i = 0; i < current_panel_cols; i++) {
                    h_fallback_ipiv[i] = i + 1; // Identity for now (no actual pivoting)
                }
                cudaMemcpy(d_IPIV_panel, h_fallback_ipiv, current_panel_cols * sizeof(int), cudaMemcpyHostToDevice);
                delete[] h_fallback_ipiv;
            } else {
                std::cout << "Kernel completed successfully" << std::endl;
            }
        }

        cudaMemcpy(ipiv_panel_print, d_IPIV_panel, current_panel_cols * sizeof(int), cudaMemcpyDeviceToHost);
        // Debug print for IPIV panel
        std::cout << "IPIV panel: ";
        for (int i = 0; i < current_panel_cols; ++i)
            std::cout << ipiv_panel_print[i] << " ";
        std::cout << std::endl;

        // b.ii. Apply permutations to FP64 matrix (kernel)
        // LASWP_kernel << <(current_panel_cols + 255) / 256, 256 >> > (d_A, N, k, current_panel_cols, d_IPIV_panel);
        // cudaDeviceSynchronize();


        // Copy panel from device to host
        cudaMemcpy2D(h_A, N * sizeof(double),
                     d_A + k * N + k, N * sizeof(double),
                     N * sizeof(double), N,
                     cudaMemcpyDeviceToHost);

        // Copy IPIV panel from device to host for LAPACKE_dlaswp
        int* h_ipiv_panel = new int[current_panel_cols];
        cudaMemcpy(h_ipiv_panel, d_IPIV_panel, current_panel_cols * sizeof(int), cudaMemcpyDeviceToHost);

        // Apply row swaps using LAPACKE_dlaswp (1-based ipiv)
        LAPACKE_dlaswp(LAPACK_COL_MAJOR, panel_rows, h_A, N, 1, current_panel_cols, h_ipiv_panel, 1);

        delete[] h_ipiv_panel;

        // Copy updated panel back to device
        cudaMemcpy2D(d_A + k * N + k, N * sizeof(double),
                     h_A, N * sizeof(double),
                     N * sizeof(double), N,
                     cudaMemcpyHostToDevice);

        // Update global IPIV array
        int *h_panel_ipiv = new int[current_panel_cols];
        cudaMemcpy(h_panel_ipiv, d_IPIV_panel, current_panel_cols * sizeof(int), cudaMemcpyDeviceToHost);
        
        std::cout << "Panel IPIV from kernel: ";
        for (int j = 0; j < current_panel_cols; ++j) {
            std::cout << h_panel_ipiv[j] << " ";
        }
        std::cout << std::endl;
        
        for (int j = 0; j < current_panel_cols; ++j) {
            // h_panel_ipiv[j] is 1-based relative to panel start
            // Convert to global 1-based index: panel_start + panel_relative_index
            h_IPIV[k + j] = h_panel_ipiv[j] + k;
            std::cout << "Global IPIV[" << (k + j) << "] = " << h_IPIV[k + j] << std::endl;
        }
        delete[] h_panel_ipiv;

        // b.iii. Copy updated panel back for FP64 factorization
        cudaMemcpy2D(d_P_FP64_NPV_buffer, panel_rows * sizeof(double),
            d_A + k * N + k, N * sizeof(double),
            current_panel_cols * sizeof(double), panel_rows,
            cudaMemcpyDeviceToDevice);



            
        // Panel LU in FP64 (no pivoting, kernel)
        if (threads > 0) {
            // DGETF2_NATIVE_NPV_kernel << <1, threads >> > (d_P_FP64_NPV_buffer, panel_rows, panel_rows, current_panel_cols);
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
    memcpy(A, h_A, N * N * sizeof(double));

    // Copy IPIV back to host
    memcpy(IPIV, h_IPIV, N * sizeof(int));
    // Cleanup
    // cublasDestroy(handle);
    delete[] h_A;
    delete[] h_IPIV;
    cudaFree(d_A);
    cudaFree(d_P_FP16_buffer);
    cudaFree(d_P_FP64_NPV_buffer);
    cudaFree(d_IPIV_panel);
    cudaFree(d_IPIV);
}