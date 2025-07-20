#include <iostream>
#include <cuda_runtime.h>
#include "fp16_utils.h"
#include "hgetf2_kernel.h"

int main() {
    // Test with the same 4x4 matrix from matrix_4x4.txt
    const int rows = 4, cols = 4;
    double matrix_data[] = {
        8.3, 8.6, 7.7, 1.5,  // Column 0
        9.3, 3.5, 8.6, 9.2,  // Column 1  
        4.9, 2.1, 6.2, 2.7,  // Column 2
        9.0, 5.9, 6.3, 2.6   // Column 3
    };
    
    // Print original matrix
    std::cout << "Original matrix (column-major):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix_data[j * rows + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Allocate host memory
    fp16 *h_panel = new fp16[rows * cols];
    int *h_ipiv = new int[cols];
    
    // Convert to FP16
    for (int i = 0; i < rows * cols; ++i) {
        h_panel[i] = double_to_fp16(matrix_data[i]);
    }
    
    // Allocate device memory
    fp16 *d_panel;
    int *d_ipiv;
    cudaMalloc(&d_panel, rows * cols * sizeof(fp16));
    cudaMalloc(&d_ipiv, cols * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_panel, h_panel, rows * cols * sizeof(fp16), cudaMemcpyHostToDevice);
    
    // Launch kernel with 1 block, multiple threads
    std::cout << "Launching HGETF2_kernel...\n";
    HGETF2_kernel<<<1, rows>>>(d_panel, rows, rows, cols, d_ipiv);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Copy results back
    cudaMemcpy(h_panel, d_panel, rows * cols * sizeof(fp16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ipiv, d_ipiv, cols * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "IPIV from HGETF2_kernel: ";
    for (int i = 0; i < cols; ++i) {
        std::cout << h_ipiv[i] << " ";
    }
    std::cout << "\n\n";
    
    std::cout << "Matrix after HGETF2_kernel:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << fp16_to_double(h_panel[j * rows + i]) << " ";
        }
        std::cout << "\n";
    }
    
    // Cleanup
    delete[] h_panel;
    delete[] h_ipiv;
    cudaFree(d_panel);
    cudaFree(d_ipiv);
    
    return 0;
}
