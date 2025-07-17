#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    std::cout << "Device count: " << deviceCount << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    }
    
    return 0;
}
