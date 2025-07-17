#include <iostream>
#include <cuda_runtime.h>
#include "fp16_utils.h"
#include "hgetf2_kernel.h"

void print_matrix_fp16(const char* name, fp16* matrix, int rows, int cols, int ld) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << fp16_to_double(matrix[j * ld + i]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Test matrix: 3x3 panel for LU factorization
    const int rows = 3, cols = 3;
    
    // Input matrix (column-major):
    // 3  0 -5
    // -1 -1  3  
    // -8  1  0
    double input[9] = {3, -1, -8, 0, -1, 1, -5, 3, 0};
    
    // Convert to FP16
    fp16 h_panel[9];
    for (int i = 0; i < 9; i++) {
        h_panel[i] = double_to_fp16(input[i]);
    }
    
    std::cout << "=== HGETF2_KERNEL DEBUG TEST ===\n";
    print_matrix_fp16("Input matrix", h_panel, rows, cols, rows);
    
    // Allocate device memory
    fp16 *d_panel;
    int *d_ipiv;
    cudaMalloc(&d_panel, 9 * sizeof(fp16));
    cudaMalloc(&d_ipiv, cols * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_panel, h_panel, 9 * sizeof(fp16), cudaMemcpyHostToDevice);
    
    // Initialize IPIV to identity
    int h_ipiv[3] = {1, 2, 3};
    cudaMemcpy(d_ipiv, h_ipiv, cols * sizeof(int), cudaMemcpyHostToDevice);
    
    std::cout << "Launching HGETF2_kernel...\n";
    
    // Launch kernel
    int threads = std::min(1024, rows - 1);
    HGETF2_kernel<<<1, threads>>>(d_panel, rows, rows, cols, d_ipiv);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    
    std::cout << "Kernel completed successfully!\n\n";
    
    // Copy results back
    cudaMemcpy(h_panel, d_panel, 9 * sizeof(fp16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ipiv, d_ipiv, cols * sizeof(int), cudaMemcpyDeviceToHost);
    
    print_matrix_fp16("Output matrix (LU factorized)", h_panel, rows, cols, rows);
    
    std::cout << "IPIV (pivot indices): ";
    for (int i = 0; i < cols; i++) {
        std::cout << h_ipiv[i] << " ";
    }
    std::cout << "\n\n";
    
    // Verify: Extract L and U matrices
    std::cout << "L matrix (lower triangular):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i > j) {
                std::cout << fp16_to_double(h_panel[j * rows + i]) << " ";
            } else if (i == j) {
                std::cout << "1 ";
            } else {
                std::cout << "0 ";
            }
        }
        std::cout << "\n";
    }
    
    std::cout << "\nU matrix (upper triangular):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i <= j) {
                std::cout << fp16_to_double(h_panel[j * rows + i]) << " ";
            } else {
                std::cout << "0 ";
            }
        }
        std::cout << "\n";
    }
    
    // Cleanup
    cudaFree(d_panel);
    cudaFree(d_ipiv);
    
    return 0;
}
