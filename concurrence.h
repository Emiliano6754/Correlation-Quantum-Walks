#ifndef CONCURRENCE_H
#define CONCURRENCE_H

#include<Eigen/Dense>

// Calculates concurrence of a set of 2 qubit states with state_number elements
void compressed_states_concurrence(const Eigen::VectorXcd &state, const unsigned int &state_number, Eigen::VectorXd &weighted_concurrences);

#endif