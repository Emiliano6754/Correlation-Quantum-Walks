#include "concurrence.h"
#include "omp.h"
#include<unsupported/Eigen/CXX11/Tensor>

void compressed_states_concurrence(const Eigen::VectorXcd &state, const unsigned int &state_number, Eigen::VectorXd &weighted_concurrences) {   
    Eigen::TensorMap<const Eigen::Tensor<Eigen::dcomplex,3>> qubit_state(state.data(),state_number,2,2);
    #pragma omp simd
    for (unsigned int t = 0; t < state_number; t++) {
        weighted_concurrences[t] = 2*std::abs(qubit_state(t,0,0)*qubit_state(t,1,1) - qubit_state(t,0,1) * qubit_state(t,1,0));
    }
}