#include<iostream>
#include<fstream>
#include<vector>
#include<unsupported/Eigen/MatrixFunctions>
#include<unsupported/Eigen/CXX11/Tensor>
// #include<cstdlib>
// #include<ctime>
#include<algorithm>
#include<string>
#include<filesystem>
#include "seed_generator.h"

// Cap of 28 qubits for sure, unless unsigned int are changed for unsigned long

const unsigned int D = 7;
const unsigned int n_qubits = 2;
const unsigned int Max_Time = 2;
const float w = 2*EIGEN_PI/D;
const unsigned int qubitstate_size = 1 << n_qubits;
const float sqrt2 = sqrt(2);
const std::complex<float> im(0.0,1.0);
const Eigen::Vector2cf qubit_instate(std::complex(1.0/sqrt2,0.0),std::complex(0.0,1.0/sqrt2)); // Assumes all qubits are initialized to this state
const Eigen::VectorXcf pinstate = Eigen::VectorXcf::Constant(D,1.0/sqrt(D)); // It also assumes position initialized at |0>

inline std::complex<float> omega(const unsigned int &p) {
    return std::exp(w*p*im);
}

inline std::complex<float> omega(const int &p) {
    return std::exp(w*p*im);
}

// Assumes powers_buffer already allocated enough space. Minimum value of 1 for max_power
inline void generate_matrix_powers_buffer(std::vector<Eigen::Matrix2cf> &powers_buffer,const unsigned int &max_power, const Eigen::Matrix2cf &matrix) {
    Eigen::MatrixPower<Eigen::Matrix2cf> matrix_power(matrix);
    powers_buffer[0] = Eigen::Matrix2cf::Identity();
    powers_buffer[1] = matrix;
    for (unsigned int j = 2; j <= max_power; j++) {
        powers_buffer[j] = matrix_power(j);
    }
}

// Positions elements parsed between ini_time and fin_time in those positions inside interaction_seed 
inline void parse_interaction_seed(const std::string &seed_filename, const unsigned int &ini_time, const unsigned int &fin_time, std::vector<unsigned int> &interaction_seed) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ifstream seed_file(cwd.string()+"/"+seed_filename,std::ifstream::in);
    if (seed_file.is_open()) {
        const unsigned int length = std::count(std::istreambuf_iterator<char>(seed_file),std::istreambuf_iterator<char>(),'\n');
        std::cout << length << std::endl;
        if (ini_time >= fin_time || fin_time > length) {
            std::cout << "Time selection outside bounds or reversed" << std::endl;
            return;
        }
        interaction_seed.reserve(fin_time+1);
        if (fin_time >= interaction_seed.size()){
            interaction_seed.resize(fin_time+1);
        }
        unsigned int pos = 0;
        std::string line;
        seed_file.seekg(seed_file.beg);
        while(seed_file.good() && pos < ini_time) {
            std::getline(seed_file,line);
            pos++;
        }
        while(seed_file.good() && pos <= fin_time) {
            std::getline(seed_file,line);
            if (line != "") {
                interaction_seed[pos] = std::stoi(line);
            }
            pos++;
        }
    } else {
        std::cout << "Failed to open file" << std::endl;
    }
    return;
}

// Overwrites everything in the file
inline void write_state_as_is(const std::vector<Eigen::VectorXcf> &state, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (output_file.is_open()) {
        for (unsigned int j = 0; j < state.size(); j++) {
            output_file << j << std::endl;
            output_file << state[j] << std::endl;
        }
    } else {
            std::cout << "Failed to write file" << std::endl;
    }
    output_file.close();
}

// Assumes state is of size D, and that a number TimeSize of full qubitstates is compressed in each VectorXcf
inline void write_state_reordered(const std::vector<Eigen::VectorXcf> &state, const unsigned int &TimeSize, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (output_file.is_open()) {
        for (unsigned int t = 0; t <= TimeSize; t++) {
            for (unsigned int m = 0; m < D; m++) {
                // output_file << m << std::endl;
                for (unsigned int n = 0; n < qubitstate_size; n++) {
                    output_file << state[m](t*qubitstate_size + n).real() << "," << state[m](t*qubitstate_size + n).imag() << std::endl;
                }
            }
        }
    } else {
            std::cout << "Failed to write file" << std::endl;
    }
    output_file.close();
}

// Assumes interaction_counts has been adequately allocated and sized
inline void count_interactions(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> &interaction_counts, const std::vector<unsigned int> &interaction_seed, const unsigned int max_time) {
    for (unsigned int n = 0; n < n_qubits; n++) {
            interaction_counts(0,n) = 0;
        }
    for (unsigned int t = 0; t < max_time; t++) {
        for (unsigned int n = 0; n < n_qubits; n++) {
            interaction_counts(t+1,n) = std::count(interaction_seed.begin(),interaction_seed.begin()+t,n);
        }
    }
}

// Evolved state only contains the evolution for p from t (starting at t+1) to T and is compressed. Evolved_state should be sized adequately.
inline void Pevolution_fromt_toT(const unsigned int &p, const unsigned int &t, const unsigned int &T, const Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> &interaction_counts, 
                                 Eigen::VectorXcf &evolved_state) {
    Eigen::Matrix2cf evolution_matrix;
    evolution_matrix << -1/sqrt2 * omega(p), 1/sqrt2 * omega(p),
                        1/sqrt2 * omega(-p), 1/sqrt2 * omega(-p);
    const unsigned int max_power = interaction_counts.maxCoeff();
    std::vector<Eigen::Matrix2cf> powers_buffer(max_power+1);
    generate_matrix_powers_buffer(powers_buffer,max_power,evolution_matrix);
    unsigned int time = 0;
    std::vector<Eigen::Vector2cf> states(n_qubits);
    const std::complex<float> in_pcoeff = pinstate(p);
    std::complex<float> coeff = 0;
    for (unsigned int s = t+1; s <= T; s++) {
        for (unsigned int n = 0; n < n_qubits; n++) {
            states[n] = powers_buffer[interaction_counts(s,n)] * qubit_instate;
        }
        // Loop for all possible qubit states, the binary representation of n gives the corresponding qubits' states. Multiply initial coefficient of the spatial state by
        // the evolved state of each qubit.
        for (unsigned int n = 0; n < qubitstate_size; n++) {
            coeff = in_pcoeff;
            for (unsigned int qubit = 0; qubit < n_qubits; qubit++) {
                coeff *= states[qubit]( (n >> qubit) & 0x1 );
            }
            evolved_state[time * qubitstate_size + n] = coeff;
        }
        time++;
    }
}

inline void evolution_fromt_toT(const unsigned int &t, const unsigned int &T, const Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> &interaction_counts, std::vector<Eigen::VectorXcf> &full_evolved_state) {
    // Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> interaction_counts(T,n_qubits);
    // count_interactions(interaction_counts,interaction_seed,T);
    // Eigen::Tensor<std::complex<float>, 3> full_evolved_state(T-t+1, D, qubitstate_size);
    for (unsigned int p = 0; p < D; p++) {
        full_evolved_state[p].resize((T-t+1)*qubitstate_size);
        // Before threading this, p should be passed by value, not by reference.
        Pevolution_fromt_toT(p,t,T,interaction_counts,full_evolved_state[p]);
    }
}

// Transforms state to usual basis. Initializes VectorXcfs to the adequate size automatically. Minimum value of D = 2 required. Makes a full copy.
inline void transform_evolved_state(const std::vector<Eigen::VectorXcf> &pevolved_state, std::vector<Eigen::VectorXcf> &finstate) {
    // finstate.reserve(D);
    // finstate.resize(D);
    const float sqrtDInv = 1/sqrt(D);
    for (unsigned int m = 0; m < D; m++) {
        // std::cout << 0 << std::endl;
        // std::cout << sqrtDInv * pevolved_state[0] << std::endl;
        finstate[m] = sqrtDInv * pevolved_state[0];
        for (unsigned int p = 1; p < D; p++) {
            // std::cout << p << std::endl;
            // std::cout << sqrtDInv * omega(-m*p) * pevolved_state[p] << std::endl;
            finstate[m] += sqrtDInv * omega(-int(m*p)) * pevolved_state[p];
        }
    }
}

// For now, it just goes from 0 to T stupidly and it doesn't parallelize anything
inline void generate_and_evolve_seed_toT(const unsigned int T, const std::string &seed_filename, const std::string &output_filename) {
    std::cout << "Generating seed" << std::endl;
    generate_seed(n_qubits,T,seed_filename);
    std::vector<unsigned int> interaction_seed(T);
    std::cout << "Parsing seed" << std::endl;
    parse_interaction_seed(seed_filename,0,T,interaction_seed);
    std::cout << "Counting interactions" << std::endl;
    Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> interaction_counts(T+1,n_qubits);
    count_interactions(interaction_counts,interaction_seed,T);
    std::cout << "Evolving state" << std::endl;
    std::vector<Eigen::VectorXcf> full_evolved_state(D);
    evolution_fromt_toT(0,T,interaction_counts, full_evolved_state);
    std::cout << "Transforming state" << std::endl;
    std::vector<Eigen::VectorXcf> finstate(D);
    transform_evolved_state(full_evolved_state,finstate);
    std::cout << "Writing reordered state" << std::endl;
    write_state_reordered(full_evolved_state,T,"estadop.txt");
    write_state_reordered(finstate,T,output_filename);
}

int main() {
    std::string seed_filename = "seedtest.txt";
    std::string output_filename = "testo.txt";
    generate_and_evolve_seed_toT(Max_Time,seed_filename,output_filename);

    return 0;
}