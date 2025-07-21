#include "hgetf2_kernel.h"
#include <cmath>
#include <cooperative_groups.h>

// Global memory for inter-block communication
__device__ fp16 g_block_max_vals[1024];  // Max 1024 blocks
__device__ int g_block_max_indices[1024];

// CUDA kernel for HGETF2 (panel LU in FP16) - Cooperative Groups version
// panel: [in/out] pointer to the panel matrix in FP16
// ld: [in] leading dimension of the panel matrix
// rows: [in] number of rows in the panel
// cols: [in] number of columns in the panel
// ipiv_panel: [out] pivot indices for the panel
__global__ void HGETF2_kernel(fp16 *panel, int ld, int rows, int cols, int *ipiv_panel) {
    auto grid = cooperative_groups::this_grid();
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    // Process each column sequentially (this is the nature of LU factorization)
    for (int j = 0; j < cols; ++j) {


        grid.sync();
        __shared__ int shared_piv;

        // Step 1: Find pivot element using multi-block reduction
        __shared__ fp16 max_vals[256];  // Reduced size for better occupancy
        __shared__ int piv_indices[256];

        // Initialize shared memory
        if (tid < 256) {
            max_vals[tid] = __float2half(0.0f);
            piv_indices[tid] = j;
        }

        // Each thread checks one element for maximum
        int row_idx = global_tid + j;
        if (row_idx < rows && tid < 256) {
            fp16 val = __habs(panel[j * ld + row_idx]);
            max_vals[tid] = val;
            piv_indices[tid] = row_idx;
        }
        __syncthreads();

        // Block-level reduction to find maximum
        for (int stride = min(blockDim.x, 256) / 2; stride > 0; stride /= 2) {
            if (tid < stride && tid + stride < min(blockDim.x, 256)) {
                if (max_vals[tid + stride] > max_vals[tid]) {
                    max_vals[tid] = max_vals[tid + stride];
                    piv_indices[tid] = piv_indices[tid + stride];
                }
            }
            __syncthreads();
        }

        // Store block result in global memory
        if (tid == 0) {
            g_block_max_vals[bid] = max_vals[0];
            g_block_max_indices[bid] = piv_indices[0];
        }

        // Synchronize all blocks - cooperative groups magic!
        grid.sync();

        // Block 0 performs inter-block reduction
        if (bid == 0 && tid == 0) {
            // Find global maximum across all blocks
            fp16 global_max = g_block_max_vals[0];
            int global_max_idx = g_block_max_indices[0];
            
            for (int b = 1; b < gridDim.x; ++b) {
                if (g_block_max_vals[b] > global_max) {
                    global_max = g_block_max_vals[b];
                    global_max_idx = g_block_max_indices[b];
                }
            }
            
            shared_piv = global_max_idx + 1; // Store 1-based index
            ipiv_panel[j] = shared_piv;
        }

        // Synchronize all blocks again - pivot is ready
        grid.sync();

        // All blocks read the computed pivot
        shared_piv = ipiv_panel[j];
        __syncthreads();

        // Step 2: Perform row swap if needed (parallel across columns)
        if (shared_piv != (j + 1)) { // Compare 1-based indices
            int col_idx = global_tid;
            if (col_idx < cols) {
                // Convert 1-based to 0-based for access
                swap_fp16(panel[col_idx * ld + j], panel[col_idx * ld + (shared_piv - 1)]);
            }
        }

        // Synchronize all blocks before Gaussian elimination
        grid.sync();

        // Step 3: Gaussian elimination - compute multipliers and update (parallel)
        row_idx = global_tid + j + 1;
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

        // Synchronize all blocks before next column
        grid.sync();
    }
}
