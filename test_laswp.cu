#include <iostream>
#include <cuda_runtime.h>

// Copy of the LASWP_kernel for testing
__global__ void LASWP_kernel_test(double *A, int lda, int k, int cols, const int *ipiv_panel) {
    int panel_row = blockIdx.y * blockDim.y + threadIdx.y; // 0 <= panel_row < cols
    int col = blockIdx.x * blockDim.x + threadIdx.x;        // 0 <= col < lda
    if (panel_row < cols && col < lda) {
        int piv = ipiv_panel[panel_row] - 1; // Convert 1-based global index to 0-based
        int current_row = panel_row + k;     // Current row in global coordinates
        if (piv != current_row) {
            // Swap A[col * lda + current_row] <-> A[col * lda + piv]
            double tmp = A[col * lda + current_row];
            A[col * lda + current_row] = A[col * lda + piv];
            A[col * lda + piv] = tmp;
        }
    }
}

int main() {
    const int N = 4;
    
    // Test matrix (column-major)
    double h_A[] = {
        8.3, 8.6, 7.7, 1.5,  // Column 0
        9.3, 3.5, 8.6, 9.2,  // Column 1  
        4.9, 2.1, 6.2, 2.7,  // Column 2
        9.0, 5.9, 6.3, 2.6   // Column 3
    };
    
    // IPIV from HGETF2_kernel: [2, 4, 3, 4]
    int h_ipiv[] = {2, 4, 3, 4};
    
    std::cout << "Original matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_A[j * N + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Allocate device memory
    double *d_A;
    int *d_ipiv;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_ipiv, N * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ipiv, h_ipiv, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Test LASWP for panel starting at k=0 with 4 columns
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    
    std::cout << "Applying LASWP with IPIV [2, 4, 3, 4] for k=0...\n";
    LASWP_kernel_test<<<grid, block>>>(d_A, N, 0, N, d_ipiv);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "Matrix after LASWP:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_A[j * N + i] << " ";
        }
        std::cout << "\n";
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_ipiv);
    
    return 0;
}
