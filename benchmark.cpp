#include <iostream>
#include <fstream>
#include <lapacke.h>
#include "MPF.h"
#include <chrono>
#include <cstring>

using namespace std;

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