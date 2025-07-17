#pragma once
#include "fp16_utils.h"

void dgetf2_native_npv(int m, int n, double *A, int lda, int *ipiv, int &info);