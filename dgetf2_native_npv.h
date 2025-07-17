#pragma once

// CUDA kernel for DGETF2 (panel LU in double, no pivoting)
// panel: [in/out] pointer to the panel matrix in double
// ld: [in] leading dimension of the panel matrix
// m: [in] number of rows in the panel
// n: [in] number of columns in the panel
__global__ void dgetf2_native_npv(int m, int n, double *panel, int ld);