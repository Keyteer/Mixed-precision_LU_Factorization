#include <iostream>
#include <cuda_runtime.h>
#include "fp16_utils.h"
#include "hgetf2_kernel.h"

int main() {
    const int rows = 3, cols = 2;
    const int total_elements = rows * cols;
    
    // Allocate host memory
    fp16 *h_panel = new fp16[total_elements];
    int *h_ipiv = new int[cols];
    
    // Initialize test data
    for (int i = 0; i < total_elements; ++i) {
        h_panel[i] = double_to_fp16(i + 1.0); // Simple test values
    }
    
    // Print input
    std::cout << "Input panel:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << fp16_to_double(h_panel[j * rows + i]) << " ";
        }
        std::cout << "\n";
    }
    
    // Allocate device memory
    fp16 *d_panel;
    int *d_ipiv;
    cudaMalloc(&d_panel, total_elements * sizeof(fp16));
    cudaMalloc(&d_ipiv, cols * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_panel, h_panel, total_elements * sizeof(fp16), cudaMemcpyHostToDevice);
    
    // Launch kernel with fewer threads for easier debugging
    int threads = std::min(32, rows - 1); // Use fewer threads
    std::cout << "Launching kernel with " << threads << " threads\n";
    
    if (threads > 0) {
        HGETF2_kernel<<<1, threads>>>(d_panel, rows, rows, cols, d_ipiv);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
    }
    
    // Copy results back
    cudaMemcpy(h_panel, d_panel, total_elements * sizeof(fp16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ipiv, d_ipiv, cols * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Output panel:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << fp16_to_double(h_panel[j * rows + i]) << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "IPIV: ";
    for (int i = 0; i < cols; ++i) {
        std::cout << h_ipiv[i] << " ";
    }
    std::cout << "\n";
    
    // Cleanup
    delete[] h_panel;
    delete[] h_ipiv;
    cudaFree(d_panel);
    cudaFree(d_ipiv);
    
    return 0;
}
