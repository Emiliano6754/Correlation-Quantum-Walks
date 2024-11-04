#include "concurrence.h"
#include "omp.h"
#include<unsupported/Eigen/CXX11/Tensor>

unsigned int qubitstate_size = 1 << 2;

void compressed_states_concurrence(const Eigen::VectorXcd &state, const unsigned int &state_number, Eigen::VectorXd &weighted_concurrences) {   
    unsigned int time_pos = 0;
    #pragma omp simd
    for (unsigned int t = 0; t < state_number; t++) {
        weighted_concurrences[t] = 2*std::abs(state(time_pos)*state(time_pos+3) - state(time_pos+1) * state(time_pos+2));
        time_pos += qubitstate_size;
    }
}


// void compressed_states_concurrence(const Eigen::VectorXcd &state, const unsigned int &state_number, Eigen::VectorXd &weighted_concurrences) {   
//     Eigen::TensorMap<const Eigen::Tensor<Eigen::dcomplex,3>,Eigen::RowMajor> qubit_state(state.data(),state_number,2,2);
//     #pragma omp simd
//     for (unsigned int t = 0; t < state_number; t++) {
//         weighted_concurrences[t] = 2*std::abs(qubit_state(t,0,0)*qubit_state(t,1,1) - qubit_state(t,0,1) * qubit_state(t,1,0));
//     }
// }