#pragma once
#include "fp16_utils.h"

// CUDA kernel for HGETF2 (panel LU in FP16)
// panel: [in/out] pointer to the panel matrix in FP16
// ld: [in] leading dimension of the panel matrix
// rows: [in] number of rows in the panel
// cols: [in] number of columns in the panel
// ipiv_panel: [out] pivot indices for the panel
__global__ void HGETF2_kernel(fp16 *panel, int ld, int rows, int cols, int *ipiv_panel);
