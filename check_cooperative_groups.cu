#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        std::cout << "=== Device " << dev << ": " << prop.name << " ===" << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        // Check cooperative groups support
        int cooperative_launch = 0;
        cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, dev);
        
        int cooperative_multi_device = 0;
        cudaDeviceGetAttribute(&cooperative_multi_device, cudaDevAttrCooperativeMultiDeviceLaunch, dev);
        
        std::cout << "Cooperative Launch Support: " << (cooperative_launch ? "YES" : "NO") << std::endl;
        std::cout << "Cooperative Multi-Device: " << (cooperative_multi_device ? "YES" : "NO") << std::endl;
        
        // Additional useful info
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
        std::cout << "Max Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
        
        // Calculate max cooperative blocks
        if (cooperative_launch) {
            int max_blocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, nullptr, 256, 0);
            int total_max_blocks = max_blocks * prop.multiProcessorCount;
            std::cout << "Max Cooperative Blocks (estimated): " << total_max_blocks << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    return 0;
}
