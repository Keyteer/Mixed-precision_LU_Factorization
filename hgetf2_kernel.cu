#include "hgetf2_kernel.h"
#include <cmath>

// Global memory for inter-block communication - increase size for safety
__device__ fp16 g_block_max_vals[4096];  // Support up to 4096 blocks
__device__ int g_block_max_indices[4096];
__device__ int g_blocks_done[256];       // Support up to 256 columns (more than enough)

// CUDA kernel for HGETF2 (panel LU in FP16) - Single-block version (safer)
// panel: [in/out] pointer to the panel matrix in FP16
// ld: [in] leading dimension of the panel matrix
// rows: [in] number of rows in the panel
// cols: [in] number of columns in the panel
// ipiv_panel: [out] pivot indices for the panel
__global__ void HGETF2_kernel(fp16 *panel, int ld, int rows, int cols, int *ipiv_panel) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Only use block 0 to avoid inter-block synchronization issues
    if (bid != 0) return;
    
    // Process each column sequentially (this is the nature of LU factorization)
    for (int j = 0; j < cols; ++j) {
        __shared__ fp16 max_vals[1024];
        __shared__ int piv_indices[1024];
        __shared__ int shared_piv;

        // Bounds check for safety
        if (tid >= 1024) return;

        // Initialize shared memory
        max_vals[tid] = __float2half(0.0f);
        piv_indices[tid] = j;

        // Each thread checks one element for maximum
        int row_idx = tid + j;
        if (row_idx < rows) {
            fp16 val = __habs(panel[j * ld + row_idx]);
            max_vals[tid] = val;
            piv_indices[tid] = row_idx;
        }
        __syncthreads();

        // Block-level reduction to find maximum
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride && tid + stride < blockDim.x) {
                if (max_vals[tid + stride] > max_vals[tid]) {
                    max_vals[tid] = max_vals[tid + stride];
                    piv_indices[tid] = piv_indices[tid + stride];
                }
            }
            __syncthreads();
        }

        // Thread 0 sets the pivot
        if (tid == 0) {
            shared_piv = piv_indices[0] + 1; // Store 1-based index
            ipiv_panel[j] = shared_piv;
        }
        __syncthreads();

        // Step 2: Perform row swap if needed (parallel across columns)
        if (shared_piv != (j + 1)) { // Compare 1-based indices
            int col_idx = tid;
            if (col_idx < cols) {
                // Convert 1-based to 0-based for access
                swap_fp16(panel[col_idx * ld + j], panel[col_idx * ld + (shared_piv - 1)]);
            }
        }
        __syncthreads();

        // Step 3: Gaussian elimination - compute multipliers and update (parallel)
        row_idx = tid + j + 1;
        if (row_idx < rows) {
            // Compute multiplier
            fp16 pivot_val = panel[j * ld + j];
            fp16 multiplier = panel[j * ld + row_idx] / pivot_val;
            panel[j * ld + row_idx] = multiplier;

            // Update remaining columns
            for (int k = j + 1; k < cols; ++k) {
                panel[k * ld + row_idx] -= multiplier * panel[k * ld + j];
            }
        }
        __syncthreads();
    }
}
