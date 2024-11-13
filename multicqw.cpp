#include<iostream>
#include<fstream>
#include<vector>
#include<Eigen/Dense>
#include<unsupported/Eigen/MatrixFunctions>
// #include<unsupported/Eigen/CXX11/Tensor>
// #include<cstdlib>
// #include<ctime>
#include<algorithm>
#include<string>
#include<filesystem>
#include<stdexcept>  // std::invalid_argument, std::out_of_range
#include<limits>     // std::numeric_limits
#include<chrono> // Timing
#include<cmath> // Trigonometry, exp
#include "omp.h"
#include "seed_generator.h"
#include "disc_qfunc.h"
#include "concurrence.h"
#include <nlopt.hpp>

# define M_PI 3.14159265358979323846  /* pi */

// Consider compiling with  -O3 -ffast-math to optimize powers

// Cap of 28 qubits for sure, unless unsigned int in qubitstate_size and related are changed for unsigned long

struct opt_params {
    unsigned int n_qubits;
    unsigned int qubitstate_size;
    unsigned int max_time;
    std::vector<unsigned int> interaction_seed;
};

unsigned int D = 101;
double w = 2*EIGEN_PI/D;
const double sqrt2 = sqrt(2);
const std::complex<double> im(0.0,1.0);
Eigen::Vector2cd qubit_instate(std::complex(1.0/sqrt2,0.0),std::complex(0.0,1.0/sqrt2)); // Assumes all qubits are initialized to this state
Eigen::VectorXcd pinstate = Eigen::VectorXcd::Constant(D,1.0/sqrt(D)); // It also assumes position initialized at |0>

double alpha = 0;
double beta = M_PI/2; // for balanced coin beta = pi/2 
double gamma = M_PI; // alpha = 0, beta = pi/2, gamma = pi returns Hadamard

// Will cause overflow issues when D > 43,000 because of instances omega(p*q)
inline std::complex<double> omega(const int &p) {
    return std::exp(w*p*im);
}

// Assumes powers_buffer already allocated enough space. Minimum value of 1 for max_power
void generate_matrix_powers_buffer(std::vector<Eigen::Matrix2cd> &powers_buffer,const unsigned int &max_power, const Eigen::Matrix2cd &matrix) {
    Eigen::MatrixPower<Eigen::Matrix2cd> matrix_power(matrix);
    powers_buffer[0] = Eigen::Matrix2cd::Identity();
    powers_buffer[1] = matrix;
    for (unsigned int j = 2; j <= max_power; j++) {
        powers_buffer[j] = matrix_power(j);
    }
}

// Positions elements parsed between ini_time and fin_time in those positions inside interaction_seed 
void parse_interaction_seed(const std::string &seed_filename, const unsigned int &ini_time, const unsigned int &fin_time, std::vector<unsigned int> &interaction_seed) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ifstream seed_file(cwd.string()+"/data/seeds/"+seed_filename,std::ifstream::in);
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
}

// Overwrites everything in the file
void write_state_as_is(const std::vector<Eigen::VectorXcd> &state, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/states/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
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

// Assumes state is of size D, and that a number TimeSize of full qubitstates is compressed in each VectorXcd
void write_state_reordered(const unsigned int &qubitstate_size, const std::vector<Eigen::VectorXcd> &state, const unsigned int &time_size, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/states/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (output_file.is_open()) {
        for (unsigned int t = 0; t <= time_size; t++) {
            for (unsigned int m = 0; m < D; m++) {
                // output_file << m << std::endl;
                for (unsigned int n = 0; n < qubitstate_size; n++) {
                    output_file << state[m](t*qubitstate_size + n).real() << "," << state[m](t*qubitstate_size + n).imag() << "\n";
                }
            }
        }
        output_file << std::flush;
    } else {
            std::cout << "Failed to write file" << std::endl;
    }
    output_file.close();
}

void save_average_concurrences(const Eigen::VectorXd &average_concurrences, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/concurrences/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    Eigen::IOFormat FullPrecision(Eigen::FullPrecision,0,"\n");
    if (output_file.is_open()) {
        output_file << average_concurrences.format(FullPrecision) << std::endl;
    } else {
        std::cout << "Could not save average concurrences" << std::endl;
    }
}

// Does nothing if more than 2 qubits, as concurrence is not relevant for a walk of that kind. Saves average ensemble concurrences as function of time.
void calculate_save_average_concurrences(const unsigned int &n_qubits, const std::vector<Eigen::VectorXcd> &state, const unsigned int &time_size, const std::string &filename) {
    if (n_qubits != 2) {
        std::cout << "Concurrence is only relevant for 2 qubit walks, omitting concurrence calculation" << std::endl;
        return;
    }
    std::vector<Eigen::VectorXd> weighted_concurrences(D,Eigen::VectorXd::Zero(time_size+1));
    #pragma omp parallel for
    for (unsigned int m = 0; m < D; m++) {
        compressed_states_concurrence(state[m],time_size+1,weighted_concurrences[m]);
    }

    for (unsigned int m = 1; m < D; m++) {
        weighted_concurrences[0] += weighted_concurrences[m];
    }
    save_average_concurrences(weighted_concurrences[0],filename);
}

double steady_ensemble_concurrence(const unsigned int &n_qubits, const std::vector<Eigen::VectorXcd> &state, const unsigned int &time_size) {
    if (n_qubits != 2) {
        std::cout << "Concurrence is only relevant for 2 qubit walks, omitting concurrence calculation" << std::endl;
        return 0;
    }
    std::vector<Eigen::VectorXd> weighted_concurrences(D,Eigen::VectorXd::Zero(time_size+1));
    #pragma omp parallel for
    for (unsigned int m = 0; m < D; m++) {
        compressed_states_concurrence(state[m],time_size+1,weighted_concurrences[m]);
    }

    for (unsigned int m = 1; m < D; m++) {
        weighted_concurrences[0] += weighted_concurrences[m];
    }
    return weighted_concurrences[0].mean();
}

void save_steady_concurrences(Eigen::VectorXd &steady_concurrences,const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/concurrences/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    Eigen::IOFormat FullPrecision(Eigen::FullPrecision,0,"\n");
    if (output_file.is_open()) {
        output_file << steady_concurrences.format(FullPrecision);
    } else {
        std::cout << "Could not save steady concurrences" << std::endl;
    }
}

// Assumes interaction_counts has been adequately allocated and sized
inline void count_interactions(const unsigned int &n_qubits, Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> &interaction_counts, const std::vector<unsigned int> &interaction_seed, const unsigned int max_time) {
    for (unsigned int n = 0; n < n_qubits; n++) {
            interaction_counts(0,n) = 0;
        }
    for (unsigned int t = 0; t < max_time; t++) {
        for (unsigned int n = 0; n < n_qubits; n++) {
            interaction_counts(t+1,n) = std::count(interaction_seed.begin(),interaction_seed.begin()+t,n);
        }
    }
}

// Initializes all qubits to their correct state at time t. Assumes the full buffer contains D buffers, but generates each of them to the appropriate 
inline void initialize_qubitstates_buffer(const unsigned int &n_qubits, const unsigned int &t, const std::vector<unsigned int> &interaction_seed, std::vector<std::vector<Eigen::Vector2cd>> &fullqubitstates_buffers) {
    if (t == 0) {
        for (unsigned int p = 0; p < D; p++) {
            fullqubitstates_buffers[p].assign(n_qubits,qubit_instate);
        }
    } else {
        std::cout << "Initialization to arbitrary times not yet implemented" << std::endl; // Should calculate each qubit state at that point of the evolution by exponentiating the corresponding evolution matrices, counting interactions with every qubit until that point.
    }
}

// Evolved state only contains the evolution for p from t (starting at t+1) to T and is compressed. Evolved_state should be sized adequately. Limited in memory by (T-t)*qubitstatesize * 16 bytes (size of complex<double>)
void Pevolution_fromt_toT(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &p, const unsigned int &t, const unsigned int &T, const std::vector<unsigned int> &interaction_seed, std::vector<Eigen::Vector2cd> &qubitstates, Eigen::VectorXcd &evolved_state) {
    Eigen::Matrix2cd evolution_matrix;
    // evolution_matrix << -1/sqrt2 * omega(p), 1/sqrt2 * omega(p),
    //                    1/sqrt2 * omega(-p), 1/sqrt2 * omega(-p); // Could probably initiallize all of them at the beginning as a const vector. It would only take D*4*16 bytes of memory, and the compiler could potentially insert the matrix manually instead of searching for it. Only if D is compile time constant
    evolution_matrix << std::cos(beta/2) * omega(p), - std::sin(beta/2) * std::exp(im*gamma) * omega(p),
                        std::sin(beta/2) * std::exp(im*alpha) * omega(-p), std::cos(beta/2) * std::exp(im*(alpha+gamma)) * omega(-p);
    unsigned int time_pos = 0;
    const std::complex<double> in_pcoeff = pinstate(p);
    std::complex<double> coeff = 0;
    for (unsigned int s = t; s < T; s++) {
        qubitstates[interaction_seed[s]] = evolution_matrix*qubitstates[interaction_seed[s]];
        // Loop for all possible qubit states, the binary representation of n gives the corresponding qubits' states. Multiply initial coefficient of the spatial state by the evolved state of each qubit.
        for (unsigned int n = 0; n < qubitstate_size; n++) {
            coeff = in_pcoeff;
            for (unsigned int qubit = 0; qubit < n_qubits; qubit++) {
                coeff *= qubitstates[qubit]( (n >> qubit) & 0x1 );
            }
            evolved_state[time_pos + n] = coeff;
        }
        time_pos += qubitstate_size;
    }
}

inline void evolution_fromt_toT(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &t, const unsigned int &T, const std::vector<unsigned int> &interaction_seed, std::vector<std::vector<Eigen::Vector2cd>> &fullqubitstates_buffers, std::vector<Eigen::VectorXcd> &full_evolved_state) {
    #pragma omp parallel for
    for (unsigned int p = 0; p < D; p++) {
        full_evolved_state[p].resize((T-t+1)*qubitstate_size);
        Pevolution_fromt_toT(n_qubits,qubitstate_size,p,t,T,interaction_seed,fullqubitstates_buffers[p],full_evolved_state[p]);
    }
}

// Transforms state to usual basis. Initializes VectorXcds to the adequate size automatically. Minimum value of D = 2 required. Makes a full copy.
inline void transform_evolved_state(const std::vector<Eigen::VectorXcd> &pevolved_state, std::vector<Eigen::VectorXcd> &fin_state) {
    const double sqrtDInv = 1/sqrt(D);
    #pragma omp parallel for simd
    for (unsigned int m = 0; m < D; m++) {
        fin_state[m] = sqrtDInv * pevolved_state[0];
        for (unsigned int p = 1; p < D; p++) {
            fin_state[m] += sqrtDInv * omega(-int(m*p)) * pevolved_state[p];
        }
    }
}

// Calculates all sums of Q^2 and returns P(m)*sum Q^2 in Qsums, with shape (D,steps_taken). Assumes Qsums already contains the correct number of vectors, but initializes each vector to a steps_taken number of 0s. Can be parallelized by taking each m independently. Maybe could be optimized changing Qsums to an Eigen matrix/vectors.
inline void calculate_sumofQ2(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const std::vector<Eigen::VectorXcd> &full_state, const unsigned int &steps_taken, std::vector<Eigen::VectorXd> &Qsums) {
    for (unsigned int m = 0; m < D; m++) {
        Qsums[m] = Eigen::VectorXd::Zero(steps_taken);
        unsigned int delta_t = 0;
        for (unsigned int t = 0; t < steps_taken; t++) {
            Eigen::Map<const Eigen::VectorXcd> qubit_state(full_state[m].data()+delta_t,qubitstate_size);
            sym_sumQ2(Qsums[m][t],n_qubits,qubitstate_size,qubit_state);
            Qsums[m][t] /= qubit_state.squaredNorm(); // squaredNorm = P(m)
            delta_t += qubitstate_size;
        }
    }
}

// Equivalent to calculate_sumofQ2, but stores the probabilities P(m,t) in probs. Calculates all sums of Q^2 and returns P(m)*sum Q^2 in Qsums, with shape (D,steps_taken). Assumes Qsums already contains the correct number of vectors, but initializes each vector to a steps_taken number of 0s. Can be parallelized by taking each m independently. Maybe could be optimized changind Qsums to an Eigen matrix/vectors.
inline void calculate_sumofQ2_with_probs(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const std::vector<Eigen::VectorXcd> &full_state, const unsigned int &steps_taken, std::vector<Eigen::VectorXd> &Qsums, std::vector<Eigen::VectorXd> &probs) {
    for (unsigned int m = 0; m < D; m++) {
        Qsums[m] = Eigen::VectorXd::Zero(steps_taken);
        probs[m] = Eigen::VectorXd::Zero(steps_taken);
        unsigned int delta_t = 0;
        for (unsigned int t = 0; t < steps_taken; t++) {
            Eigen::Map<const Eigen::VectorXcd> qubit_state(full_state[m].data()+delta_t,qubitstate_size);
            sym_sumQ2(Qsums[m][t],n_qubits,qubitstate_size,qubit_state);
            probs[m][t] = qubit_state.squaredNorm();
            Qsums[m][t] /= probs[m][t]; // squaredNorm = P(m)
            delta_t += qubitstate_size;
        }
    }
}

// Assumes probs is initialized to have D VectorXds, but resizes each accordingly.
inline void calculate_probs(const unsigned int &qubitstate_size, const unsigned int &steps_taken, const std::vector<Eigen::VectorXcd> &full_state, std::vector<Eigen::VectorXd> &probs) {
    for (unsigned int m = 0; m < D; m++) {
        probs[m] = Eigen::VectorXd::Zero(steps_taken);
        unsigned int delta_t = 0;
        for (unsigned int t = 0; t < steps_taken; t++) {
            Eigen::Map<const Eigen::VectorXcd> qubit_state(full_state[m].data()+delta_t,qubitstate_size);
            probs[m][t] = qubit_state.squaredNorm();
            delta_t += qubitstate_size;
        }
    }
}

// Returns the average sum of Q^2 in the first entry of the vector of sums
inline void average_sumofQ2(std::vector<Eigen::VectorXd> &Qsums) {
    for (unsigned int p = 1; p < D; p++) {
        Qsums[0] += Qsums[p];
    }
}

/*
    Returns the different Q(alpha,beta) for all measurements in Qfuncs when measuring at time T.

    Assumes Qfuncs already has D MatrixXds, each resized to (qubitstate_size,qubitstate_size).
*/
void calculate_squared_Qfuncs_at_T(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &T, const std::vector<Eigen::VectorXcd> &full_state, std::vector<Eigen::VectorXd> &probs, std::vector<Eigen::MatrixXd> &squared_Qfuncs) {
    const unsigned int delta_t = T * qubitstate_size;
    for (unsigned int m = 0; m < D; m++) {
        Eigen::Map<const Eigen::VectorXcd> qubit_state(full_state[m].data()+delta_t,qubitstate_size);
        sym_squared_Qfunc(squared_Qfuncs[m],n_qubits,qubitstate_size,qubit_state);
        squared_Qfuncs[m] /= probs[m][T];
    }
}

// Returns the average Q^2 function in the first entry of the vector of matrices
inline void average_squared_Qfuncs(std::vector<Eigen::MatrixXd> &squared_Qfuncs) {
    for (unsigned int m = 1; m < D; m++) {
        squared_Qfuncs[0] += squared_Qfuncs[m];
    }
}

void save_sums(const Eigen::VectorXd &Qsums, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/sums/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    Eigen::IOFormat FullPrecision(Eigen::FullPrecision,0,"\n");
    if (output_file.is_open()) {
        output_file << Qsums.format(FullPrecision);
    } else {
        std::cout << "Could not save Qsums" << std::endl;
    }
}

void save_probs(const std::vector<Eigen::VectorXd> &probs, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/probs/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    Eigen::IOFormat FullPrecision(Eigen::FullPrecision,0,"\n");
    if (output_file.is_open()) {
        for (unsigned int m = 0; m < D; m++) {
            output_file << probs[m].format(FullPrecision) << std::endl;
        }
    } else {
        std::cout << "Could not save probs" << std::endl;
    }
}

void save_squared_Qfuncs(const std::vector<Eigen::MatrixXd> &squared_Qfuncs, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/squared_Qfuncs/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    Eigen::IOFormat FullPrecision(Eigen::FullPrecision,0,"\n");
    if (output_file.is_open()) {
        for (unsigned int m = 0; m < D; m++) {
            output_file << squared_Qfuncs[m].format(FullPrecision) << std::endl;
        }
    } else {
        std::cout << "Could not save squared Qfuncs" << std::endl;
    }
}

void evolve_seed_to_T(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &T,const std::vector<unsigned int> &interaction_seed, std::vector<Eigen::VectorXcd> &fin_state) {
    std::cout << "Initializing qubits" << std::endl;
    std::vector<std::vector<Eigen::Vector2cd>> fullqubitstates_buffers(D);
    initialize_qubitstates_buffer(n_qubits,0,interaction_seed,fullqubitstates_buffers);
    std::cout << "Evolving state" << std::endl;
    std::vector<Eigen::VectorXcd> full_pevolved_state(D);
    evolution_fromt_toT(n_qubits,qubitstate_size,0,T,interaction_seed,fullqubitstates_buffers,full_pevolved_state);
    std::cout << "Transforming state" << std::endl;
    transform_evolved_state(full_pevolved_state,fin_state);
}

void generate_seed_to_T(const unsigned int &n_qubits, const unsigned int &T, const std::string &seed_filename,const unsigned int &interaction_pattern, std::vector<unsigned int> &interaction_seed) {
    std::cout << "Generating seed" << std::endl;
    switch(interaction_pattern) {
        case 0:
            generate_random_seed(n_qubits,T,seed_filename);
            break;
        case 1:
            generate_ordered_seed(n_qubits,T,seed_filename);
            break;
        case 2:
            generate_biased_seed(n_qubits,T,seed_filename);
            break;
        case 3:
            generate_completely_biased_seed(n_qubits,T,seed_filename);
    }
    std::cout << "Parsing seed" << std::endl;
    parse_interaction_seed(seed_filename,0,T,interaction_seed);
}

void generate_and_evolve_seed_to_T(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &T, const std::string &seed_filename,const unsigned int &interaction_pattern, std::vector<Eigen::VectorXcd> &fin_state) {
    std::vector<unsigned int> interaction_seed(T);
    generate_seed_to_T(n_qubits,T,seed_filename,interaction_pattern,interaction_seed);
    
    evolve_seed_to_T(n_qubits,qubitstate_size,T,interaction_seed,fin_state);
}

// For now, it just goes from 0 to T. Should transfer informational prints to their respective functions to declutter
inline void general_multicqw(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &T, const std::string &seed_filename, const std::string &Qsums_filename, const std::string &probs_filename, const std::string &squared_Qfuncs_filename, const std::string &output_filename, const bool &save_state, const bool &calculate_qsums, const bool &calculate_probabilities, const unsigned int &calculate_final_Qfuncs, const unsigned int &interaction_pattern) {
    std::vector<Eigen::VectorXcd> fin_state(D);
    generate_and_evolve_seed_to_T(n_qubits,qubitstate_size,T,seed_filename,interaction_pattern,fin_state);
    if (calculate_qsums) {
        std::vector<Eigen::VectorXd> Qsums(D);
        if (calculate_probabilities) {
            std::cout << "Calculating sums of Q^2 and probabilities" << std::endl;
            std::vector<Eigen::VectorXd> probs(D);
            auto start = std::chrono::high_resolution_clock::now();
            calculate_sumofQ2_with_probs(n_qubits,qubitstate_size,fin_state,T,Qsums,probs);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> duration = end - start;
            std::cout << "Calculating took " << duration.count() << "s" << std::endl;
            std::cout << "Saving probabilies" << std::endl;
            save_probs(probs,probs_filename);
            if (calculate_final_Qfuncs) {
                std::cout << "Calculating final Q^2" << std::endl;
                std::vector<Eigen::MatrixXd> squared_Qfuncs(D,Eigen::MatrixXd::Zero(qubitstate_size,qubitstate_size));
                calculate_squared_Qfuncs_at_T(n_qubits,qubitstate_size,T-1,fin_state,probs,squared_Qfuncs); // Calculated at T-1, since T is not evaluated
                average_squared_Qfuncs(squared_Qfuncs);
                std::cout << "Saving final Q^2" << std::endl;
                save_squared_Qfuncs(squared_Qfuncs,squared_Qfuncs_filename);
            }
        } else{
            std::cout << "Calculating sums of Q^2" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            calculate_sumofQ2(n_qubits,qubitstate_size,fin_state,T,Qsums);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> duration = end - start;
            std::cout << "Calculating took " << duration.count() << "s" << std::endl;
        }
        std::cout << "Averaging sums of Q^2" << std::endl;
        average_sumofQ2(Qsums);
        std::cout << "Writing average sums of Q^2" << std::endl;
        save_sums(Qsums[0],Qsums_filename);
    } else if (calculate_probabilities) {
        std::vector<Eigen::VectorXd> probs(D);
        std::cout << "Calculating probabilies" << std::endl;
        calculate_probs(qubitstate_size,T,fin_state,probs);
        std::cout << "Saving probabilies" << std::endl;
        save_probs(probs,probs_filename);
        if (calculate_final_Qfuncs) {
            std::cout << "Calculating final Q^2" << std::endl;
            std::vector<Eigen::MatrixXd> squared_Qfuncs(D,Eigen::MatrixXd::Zero(qubitstate_size,qubitstate_size));
            calculate_squared_Qfuncs_at_T(n_qubits,qubitstate_size,T-1,fin_state,probs,squared_Qfuncs);
            average_squared_Qfuncs(squared_Qfuncs);
            std::cout << "Saving final Q^2" << std::endl;
            save_squared_Qfuncs(squared_Qfuncs,squared_Qfuncs_filename);
        }
    }
    if (save_state) {
        std::cout << "Writing reordered state" << std::endl;
        write_state_reordered(qubitstate_size,fin_state,T,output_filename);
    }
}

void parse_unsignedint(const std::string &input, unsigned int &parsed_input) {
    try {
        unsigned long u = std::stoul(input);
        if (u > std::numeric_limits<unsigned int>::max())
            throw std::out_of_range(input);

        parsed_input = u;
    } catch (const std::invalid_argument& e) {
        std::cout << "Input could not be parsed: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Input out of range: " << e.what() << std::endl;
    }
}

void ask_for_uint(const std::string message, unsigned int &output) {
    std::string input = "";
    std::cout << message << std::endl;
    std::cin >> input;
    parse_unsignedint(input,output);
}

void ask_for_uint(const std::string message, unsigned int &output, const unsigned int &def) {
    std::string input;
    std::cout << message << " [" << def << "]" << std::endl;
    if (std::cin.peek() != '\n') {
        std::cin >> input;
        parse_unsignedint(input,output);
    } else {
        output = def;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void ask_for_pimult(const std::string message, double &output, const double &frac_def) {
    double frac;
    std::cout << message << " [" << frac_def << "]" << std::endl;
    if (std::cin.peek() != '\n') {
        std::cin >> frac;
        output = frac * M_PI;
    } else {
        output = frac_def * M_PI;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void single_multicqw(){
    std::string input;
    std::cout << "Enter the center dimension [101]" << std::endl;
    if (std::cin.peek() != '\n') {
        std::cin >> input;
        parse_unsignedint(input,D);
        w = 2*EIGEN_PI/D;
        pinstate = Eigen::VectorXcd::Constant(D,1.0/sqrt(D));
    }
    input = "";
    std::cout << "Enter the number of qubits" << std::endl;
    std::cin >> input;
    unsigned int n_qubits;
    parse_unsignedint(input,n_qubits);
    input = "";
    std::cout << "Enter the number of steps to take" << std::endl;
    std::cin >> input;
    unsigned int max_time;
    parse_unsignedint(input,max_time);
    const unsigned int qubitstate_size = 1 << n_qubits; // Supports max 31 qubits
    std::string seed_filename = "seed_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    std::string Qsums_filename = "qsums_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    std::string probs_filename = "probs_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    std::string squared_Qfuncs_filename = "sqfuncs_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    std::string output_filename = "state_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    bool selected = false;
    unsigned int interaction_pattern;
    char prefix = 'r';
    while (!selected) {
        std::cout << "Enter the interaction pattern to take [r(andom),o(rdered),b(iased),c(ompletely biased)]" << std::endl;
        input = "";
        std::cin >> input;
        if (input == "random" || input == "r") {
            interaction_pattern = 0;
            prefix = 'r';
            selected = true;
        } else if (input == "ordered" || input == "o") {
            interaction_pattern = 1;
            prefix = 'o';
            selected = true;
        } else if (input == "biased" || input == "b") {
            interaction_pattern = 2;
            prefix = 'b';
            selected = true;
        } else if (input == "completely biased" || input == "c") {
            interaction_pattern = 3;
            prefix = 'c';
            selected = true;
        }
    }
    seed_filename.insert(seed_filename.begin(),prefix);
    Qsums_filename.insert(Qsums_filename.begin(),prefix);
    probs_filename.insert(probs_filename.begin(),prefix);
    squared_Qfuncs_filename.insert(squared_Qfuncs_filename.begin(),prefix);
    output_filename.insert(output_filename.begin(),prefix);

    std::getline(std::cin, input); // Needed to clean cin before taking more input
    bool calculate_qsums = true;
    std::cout << "Calculate and save sums of Q^2? [Y,n]" << std::endl;
    input = "";
    std::getline(std::cin, input);
    if (input == "no" || input == "n" || input == "N") {
        calculate_qsums = false;
    }
    bool calculate_probabilities = true;
    input = "";
    std::cout << "Calculate and save probability distributions? [Y,n]" << std::endl;
    std::getline(std::cin, input);
    if (input == "no" || input == "n" || input == "N") {
        calculate_probabilities = false;
    }
    bool calculate_squared_Qfuncs = false;
    if (calculate_probs) {
        input = "";
        std::cout << "Calculate the final Q^2? [y/N]" << std::endl;
        std::getline(std::cin, input);
        if (input == "yes" || input == "y" || input == "Y") {
            calculate_squared_Qfuncs = true;
        }
    }
    bool save_state = false;
    input = "";
    std::cout << "Save state? [y/N]" << std::endl;
    std::getline(std::cin, input);
    if (input == "yes" || input == "y" || input == "Y") {
        save_state = true;
    }
    general_multicqw(n_qubits,qubitstate_size,max_time,seed_filename,Qsums_filename,probs_filename,squared_Qfuncs_filename,output_filename,save_state,calculate_qsums,calculate_probabilities,calculate_squared_Qfuncs,interaction_pattern);
}

void name_interaction_seed(const unsigned int &interaction_pattern, std::string &suffix, std::string &seed_filename) {
    switch(interaction_pattern) {
        case 0:
            suffix = suffix + "_r";
            seed_filename = "random_seed.txt";
            break;
        case 1:
            seed_filename = "ordered_seed.txt";
            suffix = suffix + "_o";
            break;
        case 2:
        case 3:
            seed_filename = "biased_seed.txt";
            suffix = suffix + "_b";
    }
}

void generate_or_parse_seed(const unsigned int &n_qubits, const std::string &save_file, const unsigned int &max_time, const unsigned int &interaction_pattern, std::vector<unsigned int> &interaction_seed) {
    unsigned int generate_seed = 0;
    ask_for_uint("Generate new seed? (0 = no, 1 = yes)",generate_seed,0);
    switch(generate_seed) {
        case 0:
            parse_interaction_seed(save_file,0,max_time,interaction_seed);
            break;
        case 1:
            generate_seed_to_T(n_qubits,max_time,save_file,interaction_pattern,interaction_seed);
            break;
    }
}

void set_angles_pattern(unsigned int &interaction_pattern) {
    double theta = M_PI/4;
    double phi = M_PI/2;
    ask_for_pimult("Enter beta as fraction of pi",beta,0.5);
    ask_for_pimult("Enter gamma as fraction of pi",gamma,1.0);
    ask_for_pimult("Enter theta as fraction of pi",theta,0.25);
    ask_for_pimult("Enter phi as fraction of pi",phi,0.5);
    qubit_instate = Eigen::Vector2cd(std::cos(theta),std::sin(theta)*std::complex(std::cos(phi),std::sin(phi)));
    ask_for_uint("Enter the interaction pattern (0 = random, 1 = ordered, 2 = 3 = biased)",interaction_pattern,0);
}

void angle_renyi_multicqw(){
    const unsigned int n_qubits = 2;
    const unsigned int qubitstate_size = 1 << n_qubits;
    const unsigned int max_time = 10000;
    unsigned int interaction_pattern = 0;
    std::string saves_path = "angles/";
    std::string seed_path = "batch_instates/";
    double theta = M_PI/4;
    double phi = M_PI/2;
    set_angles_pattern(interaction_pattern);
    std::string Qsums_filename = "("+std::to_string(beta)+","+std::to_string(gamma)+","+std::to_string(theta)+","+std::to_string(phi)+").txt"; // Only really good for manual one by one calculations
    std::string seed_filename;
    std::string suffix;
    name_interaction_seed(interaction_pattern,suffix,seed_filename);
    std::vector<unsigned int> interaction_seed(max_time);
    generate_or_parse_seed(n_qubits,seed_path+seed_filename,max_time,interaction_pattern,interaction_seed);
    std::vector<Eigen::VectorXcd> fin_state(D);
    evolve_seed_to_T(n_qubits,qubitstate_size,max_time,interaction_seed,fin_state);
    std::vector<Eigen::VectorXd> Qsums(D);
    calculate_sumofQ2(n_qubits,qubitstate_size,fin_state,max_time,Qsums);
    average_sumofQ2(Qsums);
    save_sums(Qsums[0],saves_path+Qsums_filename);
}

void batch_coin_operator_multicqw(){
    const unsigned int n_qubits = 2;
    const unsigned int qubitstate_size = 1 << n_qubits;
    const unsigned int max_time = 10000;
    unsigned int betas_res = 100;
    unsigned int gammas_res = 100;
    unsigned int interaction_pattern = 0;
    ask_for_uint("Enter the resolution in beta",betas_res,100);
    ask_for_uint("Enter the resolution in gamma",gammas_res,100);
    ask_for_uint("Enter the interaction pattern (0 = random, 1 = ordered, 2 = 3 = biased)",interaction_pattern,0);
    
    double theta = 3*M_PI/8; // Set initial state
    double phi = M_PI/2;
    qubit_instate = Eigen::Vector2cd(std::cos(theta),std::sin(theta)*std::complex(std::cos(phi),std::sin(phi))); // cos(theta)|0> + sin(theta)e^(iphi)|1>

    Eigen::VectorXd betas;
    Eigen::VectorXd gammas;
    std::string suffix = "";
    if (betas_res > 1) {
        betas = Eigen::VectorXd::LinSpaced(betas_res, 0, M_PI/2);
    } else {
        betas = Eigen::VectorXd::LinSpaced(1, M_PI/2, M_PI/2); // For full resolution in gamma
        suffix = "_gamma";
    }
    if (gammas_res > 1) {
        gammas = Eigen::VectorXd::LinSpaced(gammas_res, 0, 2*M_PI);
        gammas[std::round(gammas_res/4)] = M_PI/2; // Ensure gamma passes through pi/2 for the optimal coin
    } else {
        gammas = Eigen::VectorXd::LinSpaced(1, M_PI/2, M_PI/2); // For full resolution in beta
        suffix = "_beta";
    }
    if (betas_res == 1 && gammas_res == 1) {
        suffix = "optimal_coin";
    }
    std::string seed_filename = "";
    name_interaction_seed(interaction_pattern,suffix,seed_filename);
    std::vector<unsigned int> interaction_seed(max_time);
    std::string concurrences_filename = "concurrences"+suffix+".txt";
    std::string saves_path = "batch_coins/";
    generate_or_parse_seed(n_qubits,saves_path+seed_filename,max_time,interaction_pattern,interaction_seed);

    Eigen::VectorXd steady_concurrences = Eigen::VectorXd::Zero(betas_res*gammas_res);
    for (unsigned int b = 0; b < betas_res; b++) {
        for (unsigned int g = 0; g < gammas_res; g++) {
            const unsigned int n = b * gammas_res + g;
            std::string concurrence_filename = "batch_coins/"+std::to_string(n)+suffix+".txt";
            beta = betas[b];
            gamma = gammas[g];
            std::vector<Eigen::VectorXcd> fin_state(D);
            evolve_seed_to_T(n_qubits,qubitstate_size,max_time,interaction_seed,fin_state);
            steady_concurrences[n] = steady_ensemble_concurrence(n_qubits,fin_state,max_time);
        }
    }
    save_steady_concurrences(steady_concurrences,saves_path+concurrences_filename);
}

void batch_initial_states_coins_multicqw(){
    const unsigned int n_qubits = 2;
    const unsigned int qubitstate_size = 1 << n_qubits;
    const unsigned int max_time = 10000;
    unsigned int betas_res = 100;
    unsigned int thetas_res = 100;
    std::string input = "";
    std::cout << "Enter the resolution in beta" << std::endl;
    std::cin >> input;
    parse_unsignedint(input,betas_res);
    input = "";
    std::cout << "Enter the resolution in theta" << std::endl;
    std::cin >> input;
    parse_unsignedint(input,thetas_res);
    const Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(betas_res, 0, M_PI/2);
    const Eigen::VectorXd thetas = Eigen::VectorXd::LinSpaced(thetas_res, 0, 2*M_PI);
    unsigned int interaction_pattern = 1;
    for (unsigned int b = 0; b < betas_res; b++) {
        for (unsigned int g = 0; g < thetas_res; g++) {
            const unsigned int n = b * thetas_res + g;
            std::string seed_filename = "batch_instates_coin/"+std::to_string(n)+".txt";
            std::string concurrence_filename = "batch_instates_coin/"+std::to_string(n)+".txt";
            beta = betas[b];
            double theta = thetas[g];
            qubit_instate = Eigen::Vector2cd(std::complex(std::cos(theta),0.0),std::complex(0.0,std::sin(theta))); // cos(theta)|0> + isin(theta)|1>
            std::vector<Eigen::VectorXcd> fin_state(D);
            generate_and_evolve_seed_to_T(n_qubits,qubitstate_size,max_time,seed_filename,interaction_pattern,fin_state);
            calculate_save_average_concurrences(n_qubits,fin_state,max_time,concurrence_filename);
        }
    }
}

void batch_initial_states_multicqw(){
    const unsigned int n_qubits = 2;
    const unsigned int qubitstate_size = 1 << n_qubits;
    const unsigned int max_time = 5000;
    unsigned int thetas_res = 100;
    unsigned int phis_res = 100;
    beta = M_PI/2;
    gamma = M_PI/2; // Optimal coin for |0>_y
    unsigned int interaction_pattern = 0;
    ask_for_uint("Enter the resolution in theta",thetas_res,100);
    ask_for_uint("Enter the resolution in phi",phis_res,100);
    ask_for_uint("Enter the interaction pattern (0 = random, 1 = ordered, 2 = 3 = biased)",interaction_pattern,0);
    Eigen::VectorXd thetas;
    Eigen::VectorXd phis;
    std::string suffix = "";
    if (thetas_res > 1) {
        thetas = Eigen::VectorXd::LinSpaced(thetas_res, 0, 2*M_PI);
    } else {
        thetas = Eigen::VectorXd::LinSpaced(1, 3*M_PI/8, 3*M_PI/8); // For full resolution in phi
        suffix = "_phi";
    }
    if (phis_res > 1) {
        phis = Eigen::VectorXd::LinSpaced(phis_res, 0, 2*M_PI);
    }
    else {
        phis = Eigen::VectorXd::LinSpaced(1, M_PI/2, M_PI/2); // For full resolution in theta
        suffix = "_theta";
    }
    if (thetas_res == 1 && phis_res == 1) {
        suffix = "optimal_state";
    }
    // suffix = suffix + "_1001";
    std::string seed_filename = "";
    name_interaction_seed(interaction_pattern,suffix,seed_filename);
    std::vector<unsigned int> interaction_seed(max_time);
    std::string concurrences_filename = "concurrences"+suffix+".txt";
    std::string saves_path = "batch_instates/";
    generate_or_parse_seed(n_qubits,saves_path+seed_filename,max_time,interaction_pattern,interaction_seed);

    Eigen::VectorXd steady_concurrences = Eigen::VectorXd::Zero(thetas_res*phis_res);
    for (unsigned int j = 0; j < thetas_res; j++) {
        for (unsigned int k = 0; k < phis_res; k++) {
            const unsigned int n = j * phis_res + k;
            double theta = thetas[j];
            double phi = phis[k];
            qubit_instate = Eigen::Vector2cd(std::cos(theta),std::sin(theta)*std::complex(std::cos(phi),std::sin(phi))); // cos(theta)|0> + sin(theta)e^(iphi)|1>
            std::vector<Eigen::VectorXcd> fin_state(D);
            evolve_seed_to_T(n_qubits,qubitstate_size,max_time,interaction_seed,fin_state);
            steady_concurrences[n] = steady_ensemble_concurrence(n_qubits,fin_state,max_time);
        }
    }
    save_steady_concurrences(steady_concurrences,saves_path+concurrences_filename);
}

// angles = (beta,gamma,theta,phi)
double angle_concurrence(const std::vector<double> &angles,std::vector<double> &grad, void* f_data) {
    opt_params* params = static_cast<opt_params*>(f_data);
    std::vector<Eigen::VectorXcd> fin_state(D);
    beta = angles[0];
    gamma = angles[1];
    qubit_instate = Eigen::Vector2cd(std::cos(angles[2]),std::sin(angles[2])*std::complex(std::cos(angles[3]),std::sin(angles[3])));
    evolve_seed_to_T(params->n_qubits,params->qubitstate_size,params->max_time,params->interaction_seed,fin_state);
    return steady_ensemble_concurrence(params->n_qubits,fin_state,params->max_time);
}

// angles = (beta,gamma)
double state_concurrence(const std::vector<double> &angles,std::vector<double> &grad, void* f_data) {
    opt_params* params = static_cast<opt_params*>(f_data);
    std::vector<Eigen::VectorXcd> fin_state(D);
    beta = angles[0];
    gamma = angles[1];
    evolve_seed_to_T(params->n_qubits,params->qubitstate_size,params->max_time,params->interaction_seed,fin_state);
    return steady_ensemble_concurrence(params->n_qubits,fin_state,params->max_time);
}

void optimize_state_concurrence() {
    const unsigned int n_qubits = 2;
    const unsigned int qubitstate_size = 1 << n_qubits;
    const unsigned int max_time = 10000;
    double theta;
    double phi;
    ask_for_pimult("Enter theta as fraction of pi",theta,0.25);
    ask_for_pimult("Enter phi as fraction of pi",phi,0.5);
    qubit_instate = Eigen::Vector2cd(std::cos(theta),std::sin(theta)*std::complex(std::cos(phi),std::sin(phi)));

    unsigned int interaction_pattern = 0;
    std::string seed_filename = "";
    std::string seed_path = "batch_instates/";
    std::string suffix = "";
    std::vector<unsigned int> interaction_seed(max_time+1);
    ask_for_uint("Enter the interaction pattern (0 = random, 1 = ordered, 2 = 3 = biased)",interaction_pattern,0);
    name_interaction_seed(interaction_pattern,suffix,seed_filename);
    generate_or_parse_seed(n_qubits,seed_path+seed_filename,max_time,interaction_pattern,interaction_seed);

    opt_params params = opt_params(n_qubits,qubitstate_size,max_time,interaction_seed);
    nlopt::opt opt(nlopt::G_MLSL_LDS,2);
    opt.set_local_optimizer(nlopt::opt(nlopt::LN_COBYLA,2));
    opt.set_max_objective(state_concurrence,&params);
    std::vector<double> upper_bounds(2,M_PI);
    upper_bounds[0] = M_PI/2;
    const std::vector<double> lower_bounds(2,0.0);
    opt.set_upper_bounds(upper_bounds);
    opt.set_lower_bounds(lower_bounds);
    opt.set_maxtime(600);
    opt.set_ftol_abs(0.005);
    std::vector<double> guess(2,M_PI/2);
    double max_conc = 0;
    try {
        nlopt::result result = opt.optimize(guess,max_conc);
        std::cout << max_conc << " at (" << guess[0] << ", " << guess[1] << ")\n";
    }
    catch (std::runtime_error &e) {
        std::cerr << "Optimization failed: " << e.what() << "\n";
        std::cerr << "Best value before failure: " << max_conc << " at (" << guess[0] << ", " << guess[1] << ")\n";
    }
}

void optimize_angle_concurrence() {
    const unsigned int n_qubits = 2;
    const unsigned int qubitstate_size = 1 << n_qubits;
    const unsigned int max_time = 10000;
    unsigned int interaction_pattern = 0;
    std::string seed_filename = "";
    std::string seed_path = "batch_instates/";
    std::string suffix = "";
    std::vector<unsigned int> interaction_seed(max_time+1);
    ask_for_uint("Enter the interaction pattern (0 = random, 1 = ordered, 2 = 3 = biased)",interaction_pattern,0);
    name_interaction_seed(interaction_pattern,suffix,seed_filename);
    generate_or_parse_seed(n_qubits,seed_path+seed_filename,max_time,interaction_pattern,interaction_seed);
    opt_params params = opt_params(n_qubits,qubitstate_size,max_time,interaction_seed);
    nlopt::opt opt(nlopt::G_MLSL_LDS,4); 
    opt.set_local_optimizer(nlopt::opt(nlopt::LN_COBYLA,4));
    opt.set_max_objective(angle_concurrence,&params);
    std::vector<double> upper_bounds(4,M_PI);
    upper_bounds[0] = M_PI/2;
    const std::vector<double> lower_bounds(4,0.0);
    opt.set_upper_bounds(upper_bounds);
    opt.set_lower_bounds(lower_bounds);
    opt.set_maxtime(600);
    opt.set_ftol_abs(0.005);
    std::vector<double> guess(4);
    // guess[0] = M_PI/2;
    // guess[1] = M_PI/2;
    // guess[2] = M_PI/4;
    // guess[3] = M_PI/2;
    guess[0] = 0.542511;
    guess[1] = 0.543133;
    guess[2] = 2.31285;
    guess[3] = 0.682252;
    double max_conc = 0;
    try {
        nlopt::result result = opt.optimize(guess,max_conc);
        std::cout << max_conc << " at (" << guess[0] << ", " << guess[1] << ", " << guess[2] << ", " << guess[3] << ")\n";
    }
    catch (std::runtime_error &e) {
        std::cerr << "Optimization failed: " << e.what() << "\n";
        std::cerr << "Best value before failure: " << max_conc << " at (" << guess[0] << ", " << guess[1] << ", " << guess[2] << ", " << guess[3] << ")\n";
    }
}

void batch_qubits_qsums() {
    const unsigned int max_time = 5000;
    unsigned int interaction_pattern = 0;
    const unsigned int qubits[8] = {2,3,4,5,6,7,8,9};
    set_angles_pattern(interaction_pattern);
    char prefix;
    switch(interaction_pattern){
        case 0:
            prefix = 'r';
            break;
        case 1:
            prefix = 'o';
            break;
        case 2:
            prefix = 'b';
            break;
        case 3:
            prefix = 'c';
            break;
    }
    bool save_state = false;
    bool calculate_qsums = true;
    bool calculate_probabilities = true;
    bool calculate_final_Qfuncs = false;
    for (unsigned int n_qubits : qubits) {
        if (n_qubits == 5 || n_qubits == 10) {
            calculate_final_Qfuncs = true;
        } else {
            calculate_final_Qfuncs = false;
        }
        std::cout << n_qubits << std::endl;
        const unsigned int qubitstate_size = 1 << n_qubits;
        std::string seed_filename = "seed_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
        std::string Qsums_filename = "qsums_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
        std::string probs_filename = "probs_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
        std::string squared_Qfuncs_filename = "sqfuncs_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
        std::string output_filename = "state_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
        seed_filename.insert(seed_filename.begin(),prefix);
        Qsums_filename.insert(Qsums_filename.begin(),prefix);
        probs_filename.insert(probs_filename.begin(),prefix);
        output_filename.insert(output_filename.begin(),prefix);
        general_multicqw(n_qubits,qubitstate_size,max_time,seed_filename,Qsums_filename,probs_filename,squared_Qfuncs_filename,output_filename,save_state,calculate_qsums,calculate_probabilities,calculate_final_Qfuncs,interaction_pattern);
    }
}

int main() {
    // optimize_state_concurrence();
    // angle_renyi_multicqw();
    batch_qubits_qsums();
    // batch_coin_operator_multicqw();
    // batch_initial_states_multicqw();
    // optimize_angle_concurrence();

    return 0;
}

/*
Ordered
0.624426 at (0.659746, 1.81939, 2.30709, 3.02222)
0.602134 at (0.542511, 0.543133, 2.31285, 0.682252)


|0>
0.488775 at (1.5708, 3.00255)
|1>
0.488773 at (1.5708, 1.84983)
|0>_x
0.626278 at (0.586342, 1.6015)
|1>_x
0.626258 at (0.575914, 1.60221)
|0>_y
0.626299 at (0.587033, 3.09554)
|1>_y
0.626299 at (0.587033, 3.09554)


Random
0.526521 at (1.56913, 0.0718192, 2.74202, 3.09294)
|0>
0.475008 at (1.5708, 0.0530242)
|1>
0.475006 at (1.5708, 2.41698)
|0>_x
0.474797 at (1.56875, 0.00690582)
|1>_x
0.474643 at (1.56598, 0.00915533)
|0>_y
0.474812 at (1.56876, 1.54844)
|1>_y
0.474664 at (1.56876, 1.62425)
*/