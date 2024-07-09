#ifndef DISC_QFUNC_H
#define DISC_QFUNC_H

#include<Eigen/Dense>

void sym_Qfunc(Eigen::MatrixXd &Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXcd &state);
void sym_squared_Qfunc(Eigen::MatrixXd &squared_Qfunc, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXcd &state);
void sym_sumQ2(double &sum, const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXcd &state);
void generate_fiducial(Eigen::VectorXcd &fiducial, const unsigned int n_qubits, const unsigned int qubitstate_size, const std::complex<double> &xi);

#endif