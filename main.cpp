#include <iostream>
#include <iomanip>
#include <fstream>
#include <lapacke.h>
#include <vector>
#include <chrono>
#include <cstring>
#include <cblas.h>
#include <iostream>
#include "dgetf2_native_npv.h"

using namespace std;

void print_sqrMatrix(const char *msg, double *mat, int n, bool verbose = true) {
    if (verbose && n < 10) {
        cout << msg << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << mat[j * n + i] << " "; // Column-major: mat[col * lda + row]
            }
            cout << endl;
        }
        cout << endl;
    }
}

void print_LU(const double *lu, int n, bool verbose = true) {
    if (verbose && n < 10) {
        // Print L (column-major order)
        cout << "L matrix:" << endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > j)
                    cout << lu[j * n + i] << " "; // Column-major: lu[col * lda + row]
                else if (i == j)
                    cout << "1 ";
                else
                    cout << "0 ";
            }
            cout << endl;
        }
        cout << endl;

        // Print U (column-major order)
        cout << "U matrix:" << endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i <= j)
                    cout << lu[j * n + i] << " "; // Column-major: lu[col * lda + row]
                else
                    cout << "0 ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void get_LU(const double *A, double *L, double *U, int n) {
    // Extract L and U from the LU factorization
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) {
                L[i * n + j] = A[i * n + j];
                U[i * n + j] = 0.0;
            } else if (i == j) {
                L[i * n + j] = 1.0;
                U[i * n + j] = A[i * n + j];
            } else {
                L[i * n + j] = 0.0;
                U[i * n + j] = A[i * n + j];
            }
        }
    }
}

void multiply_sqrMatrices(const double *A, const double *B, double *C, int n) {
    // C = A * B using BLAS
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        n, n, n,
        1.0, A, n, B, n, 0.0, C, n);
}

void row_permute(double *A, const int *ipiv, int n) {
    // Apply the pivot swaps
    for (int i = n - 1; i >= 0; --i) {
        int piv = ipiv[i] - 1; // Convert to 0-based
        if (piv != i) {
            // Swap rows i and piv
            for (int j = 0; j < n; ++j) {
                std::swap(A[i * n + j], A[piv * n + j]);
            }
        }
    }
}

bool check_sqrMatrix_equality(double *A, double *B, int n, double tol = 1e-10) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(A[i] - B[i]) > tol) {
            return false;
        }
    }
    return true;
}

bool check_correctitude(double *A, double *Data, int ipiv[], int n, bool verbose = false) {

    // Verify results
    // get lu
    double *L = new double[n * n];
    double *U = new double[n * n];
    get_LU(Data, L, U, n);

    print_LU(Data, n, verbose);

    if (verbose && n < 10) {
        cout << "ipiv:";
        for (int i = 0; i < n; i++) {
            cout << " " << ipiv[i];
        }
        cout << "\n" << endl;
    }

    // get L*U
    double *LU = new double[n * n];
    multiply_sqrMatrices(L, U, LU, n);

    print_sqrMatrix("LU matrix:", LU, n, verbose);

    double *PLU = new double[n * n];
    memcpy(PLU, LU, n * n * sizeof(double));
    row_permute(PLU, ipiv, n);

    print_sqrMatrix("PLU matrix:", PLU, n, verbose);

    bool correctitude = check_sqrMatrix_equality(A, PLU, n);

    delete[] L;
    delete[] U;
    delete[] LU;
    delete[] PLU;

    return correctitude;
}


int main() {
    const int m = 3, n = 3;
    double A[m * n] = {
        2, 4, 6,
        1, 5, 7,
        3, 8, 9
    };  // Column-major order!

    int ipiv[3] = {3,2,3};
    int info;

    dgetf2_native_npv(m, n, A, m, ipiv, info);

    std::cout << "LU Factorized Matrix:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << A[i + j * m] << " ";
        std::cout << "\n";
    }

    std::cout << "Pivot indices:\n";
    for (int i = 0; i < std::min(m, n); ++i)
        std::cout << ipiv[i] << " ";
    std::cout << "\n";

    std::cout << "Info: " << info << "\n";

    // Print L and U using print_LU
    print_LU(A, n, true);

    std::cout << "LU Factorized Matrix:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << A[i + j * m] << " ";
        std::cout << "\n";
    }

    std::cout << "Pivot indices:\n";
    for (int i = 0; i < std::min(m, n); ++i)
        std::cout << ipiv[i] << " ";
    std::cout << "\n";

    std::cout << "Info: " << info << "\n";

    // Print L and U using print_LU
    print_LU(A, n, true);

    // Check correctitude using check_correctitude
    bool correct = check_correctitude(A, A, ipiv, n, true);
    if (correct)
        std::cout << "PA = LU check PASSED.\n";
    else
        std::cout << "PA = LU check FAILED.\n";

    
}
