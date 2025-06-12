#include <iostream>
#include <fstream>
#include <lapacke.h>
#include "MPF.h"
#include <chrono>
#include <cstring>

using namespace std;
/*
 * Mixed Precision Panel Factorization (MPF)
 * Performs LU factorization of an n x n matrix A (in-place) using mixed precision.
 * For demonstration, this function uses single precision for the panel factorization
 * and double precision for the trailing submatrix update.
 */
void MPF(int n, double* A) {
    const int panel_size = 32; // You can tune this value
    for (int k = 0; k < n; k += panel_size) {
        int pb = std::min(panel_size, n - k);

        // 1. Panel factorization in single precision
        float* panel = new float[pb * (n - k)];
        // Copy panel to single precision
        for (int j = 0; j < pb; ++j)
            for (int i = k; i < n; ++i)
                panel[j * (n - k) + (i - k)] = static_cast<float>(A[(k + j) * n + i]);

        // Simple LU factorization (Doolittle) in single precision
        for (int j = 0; j < pb; ++j) {
            for (int i = j + 1; i < n - k; ++i) {
                panel[j * (n - k) + i] /= panel[j * (n - k) + j];
                for (int l = j + 1; l < pb; ++l)
                    panel[l * (n - k) + i] -= panel[l * (n - k) + j] * panel[j * (n - k) + i];
            }
        }

        // Copy back to double precision
        for (int j = 0; j < pb; ++j)
            for (int i = k; i < n; ++i)
                A[(k + j) * n + i] = static_cast<double>(panel[j * (n - k) + (i - k)]);

        delete[] panel;

        // 2. Trailing submatrix update in double precision
        if (k + pb < n) {
            for (int i = k + pb; i < n; ++i) {
                for (int j = k + pb; j < n; ++j) {
                    double sum = 0.0;
                    for (int l = k; l < k + pb; ++l) {
                        sum += A[l * n + i] * A[j * n + l];
                    }
                    A[j * n + i] -= sum;
                }
            }
        }
    }
}
// Helper function to print a matrix
void print_matrix(const char* msg, double* mat, int n) {
    cout << msg << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << mat[j * n + i] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " filename [-v]" << endl;
        return -1;
    }

    bool verbose = false;
    if (argc > 2 && string(argv[2]) == "-v") verbose = true;

    int n;
    double *data;

    ifstream fin(argv[1]);
    if (!fin.is_open()) {
        cout << "Failed to open " << argv[1] << endl;
        return -1;
    }

    fin >> n ; 
    data = new double[n * n];
    for (int i = 0;i < n;i++) {
        for (int j = 0;j < n;j++) {
            fin >> data[j * n + i];
        }
    }
    if (fin.fail() || fin.eof()) {
        cout << "Error while reading " << argv[1] << endl;
        return -1;
    }
    fin.close();

    // Make a copy of the data for fair benchmarking
    double* data_copy = new double[n * n];
    memcpy(data_copy, data, n * n * sizeof(double));

    if (verbose && n < 10) {
        print_matrix("Original matrix:", data, n);
    }

    // Benchmark MPF (your LU factorization)
    auto start = std::chrono::high_resolution_clock::now();
    MPF(n, data);
    auto end = std::chrono::high_resolution_clock::now();
    double mpf_time = std::chrono::duration<double>(end - start).count();

    if (verbose && n < 10) {
        print_matrix("After MPF (LU):", data, n);
    }

    // Benchmark LAPACKE_dgetrf
    int* ipiv = new int[n];
    start = std::chrono::high_resolution_clock::now();
    int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, data_copy, n, ipiv);
    end = std::chrono::high_resolution_clock::now();
    double lapack_time = std::chrono::duration<double>(end - start).count();

    if (verbose && n < 10) {
        print_matrix("After LAPACKE_dgetrf (LU):", data_copy, n);
    }

    if (info != 0) {
        cout << "LAPACKE_dgetrf failed with error code " << info << endl;
    }

    cout << "MPF() time: " << mpf_time << " seconds" << endl;
    cout << "LAPACKE_dgetrf time: " << lapack_time << " seconds" << endl;

    delete[] data;
    delete[] data_copy;
    delete[] ipiv;

    return 0;
}