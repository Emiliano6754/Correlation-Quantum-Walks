#include "disc_qfunc.h"
#include<map>
#include<bit>
#include<cmath>
#include "omp.h"

// Calculates the field-wise trace of alpha by calculating its hamming weight and returning the last bit (modulo 2)
inline unsigned int trace(unsigned int alpha) {
    return std::popcount(alpha) & 0x1;
}

// Calculates the trace of the product by doing bitwise and. Equivalent to calling trace(alpha&beta)
inline unsigned int trace(unsigned int alpha, unsigned int beta) {
    return std::popcount(alpha & beta) & 0x1;
}

// Generates map of powers of xi conjugate from 0 to n_qubits. Defaults to symmetric xi. Probably this shouldnt be specialized to generate the complex buffer, but to a map of powers of xi only.
inline void generate_xi_buffer(std::map<unsigned int, std::complex<double>> &xi_buffer, const unsigned int &n_qubits, const std::complex<double> &xi) {
    std::complex<double> xi_conj = std::conj(xi);
    for (unsigned int n = 0; n <= n_qubits; n++) {
        xi_buffer[n] = std::pow(xi_conj,n);
    }
}

// Returns the fiducial as the direct product in fiducial. Assumes fiducial is sized correctly to qubitstate_size. Maximum of 32 qubits because of qubitstate_size
void generate_fiducial(Eigen::VectorXcd &fiducial, const unsigned int n_qubits, const unsigned int qubitstate_size, const std::complex<double> &xi) {
    std::map<unsigned int, std::complex<double>> xi_buffer{};
    generate_xi_buffer(xi_buffer,n_qubits,std::conj(xi)); // Conjugates xi because xi_buffer conjugates xi automatically.
    const double denom = 1.0 / std::pow(1+std::norm(xi), n_qubits/2);
    for (unsigned int n = 0; n < qubitstate_size; n++) {
        fiducial[n] = denom * xi_buffer[std::popcount(n)];
    }
}

void sym_Qfunc(Eigen::MatrixXcd &Qfunc, const unsigned int &n_qubits, const Eigen::VectorXcd &state) {

}

// Returns the sum of Q^2 for state in sum. Requires state to be normalized. Else, it returns squarednorm^2 * sum Q^2. Supports max 32 qubits, limited by qubitstate_size and the variables for the loops
void sym_sumQ2(double &sum, const unsigned int &n_qubits, const Eigen::VectorXcd &state) {
    sum = 0;
    const unsigned int qubitstate_size = 1 << n_qubits;
    const std::complex<double> xi = 0.5*(sqrt(3)-1)*std::complex<double>(1.0,1.0);
    std::map<unsigned int, std::complex<double>> xi_buffer{};
    generate_xi_buffer(xi_buffer,n_qubits,xi);
    std::map<unsigned int, std::complex<double>> im_buffer{{0,std::complex<double>(1.0,0.0)},{1,std::complex<double>(0.0,-1.0)}}; // Required for (-i)^tr(ab)
    std::map<unsigned int, double> sign_buffer{{0,1.0},{1,-1.0}}; // Required for (-1)^tr(ab). std::complex requires double to perform multiplication :|
    const double denom = 1.0 / std::pow(1+std::norm(xi), n_qubits);
    std::complex<double> coeff;
    #pragma omp parallel for
    for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
        for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
            coeff = 0;
            for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
                coeff += sign_buffer[trace(alpha,eta)] * xi_buffer[std::popcount(beta ^ eta)] * state[eta];
            }
            coeff = im_buffer[trace(alpha,beta)] * coeff;
            sum += std::pow(std::norm(coeff),2);
        }
    }
    sum *= denom*denom;
}