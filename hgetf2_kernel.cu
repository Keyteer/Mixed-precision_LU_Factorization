#include "hgetf2_kernel.h"
#include <cmath>

// CUDA kernel for HGETF2 (panel LU in FP16)
__global__ void HGETF2_kernel(fp16 *panel, int ld, int rows, int cols, int *ipiv_panel) {
    int tid = threadIdx.x;

    // Process each column sequentially (this is the nature of LU factorization)
    for (int j = 0; j < cols; ++j) {
        __shared__ int shared_piv;

        // Step 1: Find pivot element in parallel using reduction
        __shared__ fp16 max_vals[1024];
        __shared__ int piv_indices[1024];

        // Initialize shared memory
        max_vals[tid] = __float2half(0.0f);
        piv_indices[tid] = j;

        // Each thread checks one element for maximum
        if (tid + j < rows) {
            int row_idx = tid + j;
            fp16 val = __habs(panel[j * ld + row_idx]);
            max_vals[tid] = val;
            piv_indices[tid] = row_idx;
        }
        __syncthreads();

        // Parallel reduction to find maximum
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride && tid + stride < blockDim.x) {
                if (max_vals[tid + stride] > max_vals[tid]) {
                    max_vals[tid] = max_vals[tid + stride];
                    piv_indices[tid] = piv_indices[tid + stride];
                }
            }
            __syncthreads();
        }

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
        int row_idx = tid + j + 1;
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
