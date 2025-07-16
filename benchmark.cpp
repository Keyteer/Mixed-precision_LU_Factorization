#include <iostream>
#include <iomanip>
#include <fstream>
#include <lapacke.h>
#include <vector>
#include "MPF.h"
#include <chrono>
#include <cstring>
#include <cblas.h>

using namespace std;

// Helper function to print a matrix
void print_sqrMatrix(const char *msg, double *mat, int n, bool verbose = true) {
    if (verbose && n < 10) {
        cout << msg << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << mat[i * n + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void print_LU(const double *lu, int n, bool verbose = true) {
    if (verbose && n < 10) {
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
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
                swap(A[i * n + j], A[piv * n + j]);
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

int main(int argc, char **argv) {

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " filename [-v] [--no-check]" << endl;
        return -1;
    }

    bool verbose = false;
    bool check_correct = true;
    for (int i = 2; i < argc; ++i) {
        if (string(argv[i]) == "-v") verbose = true;
        if (string(argv[i]) == "--no-check") check_correct = false;
    }

    ifstream fin(argv[1]);
    if (!fin.is_open()) {
        cout << "Failed to open " << argv[1] << endl;
        return -1;
    }

    ofstream csv("benchmark_times.csv");
    csv << "matrix_size,mpf_time,lapack_time\n" << fixed << setprecision(10);

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

        int n;
        double *data_original;

        fin >> n;
        if (fin.fail() || n <= 0) {
            cout << "Invalid matrix size in " << argv[1] << " n: " << n << endl;
            return -1;
        }
        data_original = new double[n * n];
        for (int i = 0; i < n * n; i++) {
            fin >> data_original[i];
        }
        if (fin.fail()) {
            cout << "Error while reading matrix data in " << argv[1] << endl;
            delete[] data_original;
            return -1;
        }

        // Make copys of A for fair benchmarking
        double *data_dgetrf = new double[n * n];
        double *data_mpf = new double[n * n];
        memcpy(data_dgetrf, data_original, n * n * sizeof(double));
        memcpy(data_mpf, data_original, n * n * sizeof(double));


        print_sqrMatrix("Original matrix:", data_mpf, n);


        // Benchmark MPF (your LU factorization)
        int* ipiv_mpf = new int[n];
        auto start = chrono::high_resolution_clock::now();
        MPF(data_mpf, n, 32, ipiv_mpf);
        auto end = chrono::high_resolution_clock::now();
        double mpf_time = chrono::duration<double>(end - start).count();

        if (verbose) {
            cout << "MPF() time: " << mpf_time << " seconds\n" << endl;
        }

        if (check_correct) {
            if (!check_correctitude(data_original, data_mpf, ipiv_mpf, n, verbose)) {
                cout << "MPF produced incorrect results." << endl;
            }
        }

        // Benchmark LAPACKE_dgetrf
        int *ipiv = new int[n];
        start = chrono::high_resolution_clock::now();
        int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, data_dgetrf, n, ipiv);
        end = chrono::high_resolution_clock::now();
        double lapack_time = chrono::duration<double>(end - start).count();


        if (info != 0) {
            cout << "LAPACKE_dgetrf failed with error code " << info << endl;
        }

        if (verbose) {
            cout << "LAPACKE_dgetrf time: " << lapack_time << " seconds\n" << endl;
        }

        if (check_correct) {
            if (!check_correctitude(data_original, data_dgetrf, ipiv, n, verbose)) {
                cout << "LAPACKE_dgetrf produced incorrect results." << endl;
            }
        }

        delete[] data_original;
        delete[] data_mpf;
        delete[] data_dgetrf;
        delete[] ipiv;

        csv << n << "," << mpf_time << "," << lapack_time << endl;
    }
    csv.close();

    return 0;
}