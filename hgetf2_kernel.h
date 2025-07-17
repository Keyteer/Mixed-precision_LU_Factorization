#pragma once
#include "fp16_utils.h"

// CUDA kernel for HGETF2 (panel LU in FP16)
__global__ void HGETF2_kernel(fp16 *panel, int ld, int rows, int cols, int *ipiv_panel);
