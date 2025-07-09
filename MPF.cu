#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <cuda_fp16.h> // Para __half y funciones de conversión FP16
#include <cuda_runtime.h>
#include <lapacke.h>
#include <cublas_v2.h> // Para cuBLAS


// Simulación de FP16 (usa __half como tipo real de CUDA)
using fp16 = __half;

__device__ inline void swap_fp16(fp16 &a, fp16 &b) {
    fp16 tmp = a;
    a = b;
    b = tmp;
}

// Conversión robusta de FP64 a FP16 con control de overflow/underflow
fp16 double_to_fp16(double x) {
    // 1. Conversion to single precision
    float xf = static_cast<float>(x);

    // 2. Overflow/underflow thresholds for FP16
    constexpr float FP16_MAX = 65504.0f;
    constexpr float FP16_MIN_POS = 6.10352e-05f; // Smallest positive normal FP16

    // 3. Overflow handling
    if (xf > FP16_MAX)
        xf = FP16_MAX;
    else if (xf < -FP16_MAX)
        xf = -FP16_MAX;

    // 4. Underflow handling
    if (xf > -FP16_MIN_POS && xf < FP16_MIN_POS)
        xf = 0.0f;

    // 5. Conversion to half precision with rounding to nearest even
    return __float2half_rn(xf);
}

// Conversión simple FP16 -> FP64 (placeholder)
double fp16_to_double(fp16 x) {
    // Aquí deberías usar una conversión real a FP64
    return static_cast<double>(static_cast<float>(x));
}

// HGETF2: Factorización LU con pivoteo parcial en FP16 (simulada)
/*void HGETF2(fp16 *panel, int ld, int rows, int cols, int *ipiv_panel) {
    for (int j = 0; j < cols; ++j) {

        // Búsqueda de pivote (máximo valor absoluto en la columna j)
        int piv = j;
        fp16 maxval = __habs(panel[j * ld + j]);
        for (int i = j + 1; i < rows; ++i) {
            fp16 val = __habs((panel[j * ld + i]));
            if (val > maxval) {
                maxval = val;
                piv = i;
            }
        }
        ipiv_panel[j] = piv;

        // Intercambio de filas si es necesario
        if (piv != j) {
            for (int k = 0; k < cols; ++k)
                swap_fp16(panel[k * ld + j], panel[k * ld + piv]);
        }

        // Factorización
        for (int i = j + 1; i < rows; ++i) {
            fp16 lij = panel[j * ld + i] / panel[j * ld + j];
            panel[j * ld + i] = lij;
            for (int k = j + 1; k < cols; ++k) {
                fp16 a = panel[k * ld + i];
                fp16 b = panel[k * ld + j];
                panel[k * ld + i] = a - b * lij;
            }
        }
    }
}*/

// Kernel de CUDA para HGETF2
__global__ void HGETF2_kernel(fp16 *panel, int ld, int rows, int cols, int *ipiv_panel) {
    int tid = threadIdx.x;
    for (int j = 0; j < cols; ++j) {
        // 1. Búsqueda de pivote (serial, por el thread 0)
        int piv = j;
        fp16 maxval = __habs(panel[j * ld + j]);
        if (tid == 0) {
            for (int i = j + 1; i < rows; ++i) {
                fp16 val = __habs(panel[j * ld + i]);
                if (val > maxval) {
                    maxval = val;
                    piv = i;
                }
            }
            ipiv_panel[j] = piv;
        }
        __syncthreads();

        // 2. Intercambio de filas (serial, por el thread 0)
        if (tid == 0 && piv != j) {
            for (int k = 0; k < cols; ++k)
                swap_fp16(panel[k * ld + j], panel[k * ld + piv]);
        }
        __syncthreads();

        // 3. Eliminación (paralelo: cada thread maneja una fila)
        int i = j + 1 + tid;
        if (i < rows) {
            fp16 lij = panel[j * ld + i] / panel[j * ld + j];
            panel[j * ld + i] = lij;
            for (int k = j + 1; k < cols; ++k) {
                fp16 a = panel[k * ld + i];
                fp16 b = panel[k * ld + j];
                panel[k * ld + i] = a - b * lij;
            }
        }
        __syncthreads();
    }
}

// DGETF2_NATIVE_NPV: Factorización LU sin pivoteo en FP64
void DGETF2_NATIVE_NPV(double* panel, int ld, int rows, int cols) {
    for (int j = 0; j < cols; ++j) {
        for (int i = j + 1; i < rows; ++i) {
            panel[j * ld + i] /= panel[j * ld + j];
            for (int k = j + 1; k < cols; ++k)
                panel[k * ld + i] -= panel[k * ld + j] * panel[j * ld + i];
        }
    }
}

// --- LASWP: Intercambio de filas en FP64 usando LAPACK ---
void LASWP(double* A, int n, int k, int cols, const int* ipiv_panel) {
    // Usar LAPACKE_dlaswp para mayor robustez
    LAPACKE_dlaswp(LAPACK_COL_MAJOR, n, A, n, k + 1, k + cols, ipiv_panel, 1);
}

// --- DTRSM: Triangular solve en FP64 usando cuBLAS ---
void DTRSM_cublas(cublasHandle_t handle, double* dA, int lda, double* dB, int ldb, int m, int n) {
    const double alpha = 1.0;
    // Lado izquierdo, L, no transpuesta, no unitaria
    cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                m, n, &alpha, dA, lda, dB, ldb);
}

// --- DGEMM: Multiplicación de matrices en FP64 usando cuBLAS ---
void DGEMM_cublas(cublasHandle_t handle, double* dA, int lda, double* dB, int ldb, double* dC, int ldc, int m, int n, int k) {
    const double alpha = -1.0;
    const double beta = 1.0;
    // C = beta*C + alpha*A*B
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
}

// Algoritmo MPF
void MPF(double* A, int N, int r, std::vector<int>& IPIV) {
    // Buffers auxiliares
    std::vector<fp16> P_FP16_buffer(N * r);
    std::vector<double> P_FP64_NPV_buffer(N * r);
    std::vector<int> IPIV_panel(r);
    IPIV.resize(N);

    // --- Create cuBLAS handle ---
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int k = 0; k < N; k += r) {
        int current_panel_cols = std::min(r, N - k);
        int panel_rows = N - k;

        // a. Copiar panel a FP16
        for (int j = 0; j < current_panel_cols; ++j)
            for (int i = 0; i < panel_rows; ++i)
                P_FP16_buffer[j * panel_rows + i] = double_to_fp16(A[(k + j) * N + (k + i)]);

        // b.i. Factorización LU con pivoteo parcial en FP16
        HGETF2_kernel(P_FP16_buffer.data(), panel_rows, panel_rows, current_panel_cols, IPIV_panel.data());

        // b.ii. Aplicar permutaciones a la matriz original FP64
        for (int j = 0; j < current_panel_cols; ++j)
            IPIV[k + j] = k + IPIV_panel[j];
        LASWP(A, N, k, current_panel_cols, IPIV_panel.data());

        // b.iii. Factorización de panel sin pivoteo en FP64
        for (int j = 0; j < current_panel_cols; ++j)
            for (int i = 0; i < panel_rows; ++i)
                P_FP64_NPV_buffer[j * panel_rows + i] = A[(k + j) * N + (k + i)];
        DGETF2_NATIVE_NPV(P_FP64_NPV_buffer.data(), panel_rows, panel_rows, current_panel_cols);
        // Copiar de vuelta el panel factorizado
        for (int j = 0; j < current_panel_cols; ++j)
            for (int i = 0; i < panel_rows; ++i)
                A[(k + j) * N + (k + i)] = P_FP64_NPV_buffer[j * panel_rows + i];

        // c. Actualización de trailing submatrix usando cuBLAS
        if (k + current_panel_cols < N) {
            int m = panel_rows - current_panel_cols;
            int n = N - k - current_panel_cols;
            // DTRSM: resolver parte inferior del panel
            DTRSM_cublas(
                handle,
                &A[k * N + k + current_panel_cols], // dA (L)
                N,
                &A[(k + current_panel_cols) * N + k + current_panel_cols], // dB (parte inferior del panel)
                N,
                m,
                current_panel_cols
            );
            // DGEMM: actualizar trailing submatrix
            DGEMM_cublas(
                handle,
                &A[(k + current_panel_cols) * N + k + current_panel_cols], // dA
                N,
                &A[(k + current_panel_cols) * N + k], // dB
                N,
                &A[k * N + k + current_panel_cols], // dC
                N,
                m,
                n,
                current_panel_cols
            );
        }
    }

    // --- Destroy cuBLAS handle ---
    cublasDestroy(handle);
}