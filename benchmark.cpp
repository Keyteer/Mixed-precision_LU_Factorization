#include <iostream>
#include <fstream>
#include <lapacke.h>
#include "MPF.h"
#include <chrono>
#include <cstring>

using namespace std;

// Helper function to print a matrix
void print_matrix(const char* msg, double* mat, int n, int m) {
    cout << msg << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << mat[i * m + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void print_LU(const double* lu, int n) {
    // Print L
    cout << "L matrix:" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j)
                cout << lu[i * n + j] << " ";
            else if (i == j)
                cout << "1 ";
            else
                cout << "0 ";
        }
        cout << endl;
    }
    cout << endl;

    // Print U
    cout << "U matrix:" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j)
                cout << lu[i * n + j] << " ";
            else
                cout << "0 ";
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

    ifstream fin(argv[1]);
    if (!fin.is_open()) {
        cout << "Failed to open " << argv[1] << endl;
        return -1;
    }

    int num_matrices;
    fin >> num_matrices;
    if (fin.fail() || num_matrices <= 0) {
        cout << "Invalid number of matrices in " << argv[1] << endl;
        return -1;
    }
    if (verbose) {
        cout << "Number of matrices: " << num_matrices << endl;
    }

    for (int mNum = 0; mNum < num_matrices; mNum++) {

        int n, m;
        double *data;

        fin >> n;
        fin >> m;
        if (fin.fail() || n <= 0 || m <= 0) {
            cout << "Invalid matrix size in " << argv[1] << endl;
            return -1;
        }
        data = new double[n * m];
        for (int i = 0; i < n * m; i++) {
            fin >> data[i];
        }
        if (fin.fail() || fin.eof()) {
            cout << "Error while reading matrix data in " << argv[1] << endl;
            delete[] data;
            return -1;
        }
        fin.close();

        // Make a copy of the data for fair benchmarking
        double *data_copy = new double[n * m];
        memcpy(data_copy, data, n * m * sizeof(double));

        if (verbose && n < 10) {
            print_matrix("Original matrix:", data, n, m);
        }

        // Benchmark MPF (your LU factorization)
        auto start = chrono::high_resolution_clock::now();
        MPF(n, data);
        auto end = chrono::high_resolution_clock::now();
        double mpf_time = chrono::duration<double>(end - start).count();

        if (verbose && n < 10) {
            print_matrix("After MPF (LU):", data, n, m);
        }

        // Benchmark LAPACKE_dgetrf
        int *ipiv = new int[n];
        start = chrono::high_resolution_clock::now();
        int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, m, data_copy, n, ipiv);
        end = chrono::high_resolution_clock::now();
        double lapack_time = chrono::duration<double>(end - start).count();


        if (info != 0) {
            cout << "LAPACKE_dgetrf failed with error code " << info << endl;
        } else if (verbose && n < 10) {
            print_matrix("After LAPACKE_dgetrf (LU):", data_copy, n, m);
            print_LU(data_copy, n);

            // Print pivoting
            cout << "Pivoting (ipiv): ";
            for (int i = 0; i < n; ++i) {
                cout << ipiv[i] << " ";
            }
            cout << "\n" << endl;
        }

        cout << "MPF() time: " << mpf_time << " seconds" << endl;
        cout << "LAPACKE_dgetrf time: " << lapack_time << " seconds" << endl;

        delete[] data;
        delete[] data_copy;
        delete[] ipiv;
    }

    return 0;
}