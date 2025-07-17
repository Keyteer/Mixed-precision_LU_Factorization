#include <cuda_runtime.h>
#include <iostream>

// CUDA corruption detector function
void checkCudaCorruption(const char* checkpoint) {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    std::cout << "[CHECKPOINT " << checkpoint << "] ";
    std::cout << "CUDA Status: " << cudaGetErrorString(error) << ", ";
    std::cout << "Device Count: " << deviceCount;
    
    if (error == cudaSuccess && deviceCount > 0) {
        cudaDeviceProp prop;
        cudaError_t propError = cudaGetDeviceProperties(&prop, 0);
        if (propError == cudaSuccess) {
            std::cout << ", Device: " << prop.name;
        } else {
            std::cout << ", Device prop error: " << cudaGetErrorString(propError);
        }
    }
    
    // Check last error
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        std::cout << ", Last error: " << cudaGetErrorString(lastError);
    }
    
    std::cout << std::endl;
}

#define CUDA_CHECK(name) checkCudaCorruption(name)
