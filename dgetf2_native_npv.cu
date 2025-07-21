#include "dgetf2_native_npv.h"
#include <cmath>
#include <cooperative_groups.h>

// CUDA kernel for DGETF2 (panel LU in double, no pivoting) - Multi-block version with cooperative groups
// panel: [in/out] pointer to the panel matrix in double
// ld: [in] leading dimension of the panel matrix
// m: [in] number of rows in the panel
// n: [in] number of columns in the panel

__global__ void dgetf2_native_npv(int m, int n, double *panel, int ld) {
    auto grid = cooperative_groups::this_grid();
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    // Process each column sequentially (no pivoting, panel is pre-pivoted)
    for (int j = 0; j < n; ++j) {
        // Gaussian elimination - compute multipliers and update (parallel)
        int row_idx = global_tid + j + 1;
        if (row_idx < m) {
            // Compute multiplier
            double pivot_val = panel[j * ld + j];
            double multiplier = panel[j * ld + row_idx] / pivot_val;
            panel[j * ld + row_idx] = multiplier;

            // Update remaining columns
            for (int k = j + 1; k < n; ++k) {
                panel[k * ld + row_idx] -= multiplier * panel[k * ld + j];
            }
        }
        
        // Synchronize all blocks before proceeding to next column
        grid.sync();
    }
}