#include "dgetf2_native_npv.h"
#include <cmath>
#include <algorithm>

void dgetf2_native_npv(int m, int n, double *A, int lda, int *ipiv, int &info) {
    const double ONE = 1.0;
    const double ZERO = 0.0;

    info = 0;

    if (m < 0) {
        info = -1;
        return;
    } else if (n < 0) {
        info = -2;
        return;
    } else if (lda < std::max(1, m)) {
        info = -4;
        return;
    }

    if (m == 0 || n == 0)
        return;

    double sfmin = 2.2250738585072014e-308; // Safe minimum for double (approximate DBL_MIN)
    int min_mn = std::min(m, n);
    for (int j = 0; j < min_mn; ++j) {
        int jp = ipiv[j];

        // Swap rows jp and j if needed
        if (jp != j) {
            for (int k = 0; k < n; ++k) {
                std::swap(A[j + k * lda], A[jp + k * lda]);
            }
        }

        // Check for zero pivot
        if (A[j + j * lda] == ZERO && info == 0) {
            info = j + 1; // Fortran 1-based indexing
        }

        // Scale sub-column below pivot
        if (j < m - 1) {
            if (std::abs(A[j + j * lda]) >= sfmin) {
                double inv_pivot = ONE / A[j + j * lda];
                for (int i = j + 1; i < m; ++i) {
                    A[i + j * lda] *= inv_pivot;
                }
            } else {
                for (int i = j + 1; i < m; ++i) {
                    A[i + j * lda] /= A[j + j * lda];
                }
            }
        }

        // Rank-1 update to trailing submatrix
        if (j < min_mn - 1) {
            for (int i = j + 1; i < m; ++i) {
                double mult = A[i + j * lda];
                for (int k = j + 1; k < n; ++k) {
                    A[i + k * lda] -= mult * A[j + k * lda];
                }
            }
        }
    }
}