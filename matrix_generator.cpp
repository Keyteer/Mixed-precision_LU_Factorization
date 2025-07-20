#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char **argv) {

    if (argc < 3) {
        cout << "Usage: " << argv[0] << " filename maxSize [step=2] [function=exp (exp/lin)] [sparsity=0.0]" << endl;
        cout << "  sparsity: fraction of zeros in the matrix (0.0 = dense, 0.9 = 90% zeros)" << endl;
        return -1;
    }

    ofstream fout(argv[1]);
    if (!fout.is_open()) {
        cout << "Failed to open " << argv[1] << endl;
        return -1;
    }
    int maxSize = atoi(argv[2]);
    if (maxSize <= 0) {
        cout << "Invalid maxSize: " << maxSize << endl;
        return -1;
    }
    int step = 2;
    if (argc > 3) {
        step = atoi(argv[3]);
        if (step <= 0) {
            cout << "Invalid step: " << step << endl;
            return -1;
        }
    }

    bool expStepFunction = true;
    if (argc > 4) {
        string function = argv[4];
        if (function == "lin") {
            expStepFunction = false;
        } else if (function != "exp") {
            cout << "Invalid function: " << function << ". Use 'exp' or 'lin'." << endl;
            return -1;
        }
    }

    double sparsity = 0.0;
    if (argc > 5) {
        sparsity = atof(argv[5]);
        if (sparsity < 0.0 || sparsity >= 1.0) {
            cout << "Invalid sparsity: " << sparsity << ". Must be in [0.0, 1.0)." << endl;
            return -1;
        }
    }

    fout << "                " << endl;

    int size = 2;
    int numMatrices = 0;
    while (size <= maxSize) {
        fout << size << endl;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double val;
                if (sparsity > 0.0 && (static_cast<double>(rand()) / (RAND_MAX + 1.0)) < sparsity) {
                    val = 0.0;
                } else {
                    val = static_cast<double>(rand() % 100) / 10.0;
                }
                fout << val << " ";
            }
            fout << endl;
        }
        fout << endl;

        if (expStepFunction) {
            size *= step;
        } else {
            size += step;
        }
        numMatrices++;

        cout << "Generating matrix of size " << size << "\r" << flush;
    }
    // Move the file pointer to the beginning and write the number of matrices
    fout.seekp(0, ios::beg);
    fout << numMatrices;
    fout.close();

    cout << "\nnumber of matrices: " << numMatrices << endl;
    return 0;
}
