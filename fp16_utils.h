#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using fp16 = __half;

// Device function for swapping FP16 values
__device__ inline void swap_fp16(fp16 &a, fp16 &b) {
    fp16 tmp = a;
    a = b;
    b = tmp;
}

// Host and device conversion functions
__host__ __device__ inline fp16 double_to_fp16(double x) {
    float xf = static_cast<float>(x);
    constexpr float FP16_MAX = 65504.0f;
    constexpr float FP16_MIN_POS = 6.10352e-05f;
    if (xf > FP16_MAX) xf = FP16_MAX;
    else if (xf < -FP16_MAX) xf = -FP16_MAX;
    if (xf > -FP16_MIN_POS && xf < FP16_MIN_POS) xf = 0.0f;
    return __float2half_rn(xf);
}

__host__ __device__ inline double fp16_to_double(fp16 x) {
    return static_cast<double>(__half2float(x));
}
