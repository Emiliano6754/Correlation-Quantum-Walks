#ifndef DISC_QFUNC_H
#define DISC_QFUNC_H

#include<Eigen/Dense>

inline void sym_Qfunc(Eigen::MatrixXcd &Qfunc, const unsigned int &n_qubits, const Eigen::VectorXcd &state);
inline void sym_sumQ2(double &sum, const unsigned int &n_qubits, const Eigen::VectorXcd &state);

#endif