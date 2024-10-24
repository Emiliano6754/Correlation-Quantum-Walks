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

# define M_PI 3.14159265358979323846  /* pi */

// Consider compiling with  -O3 -ffast-math to optimize powers

// Cap of 28 qubits for sure, unless unsigned int in qubitstate_size and related are changed for unsigned long

unsigned int D = 101;
double w = 2*EIGEN_PI/D;
const double sqrt2 = sqrt(2);
const std::complex<double> im(0.0,1.0);
const Eigen::Vector2cd qubit_instate(std::complex(1.0/sqrt2,0.0),std::complex(0.0,1.0/sqrt2)); // Assumes all qubits are initialized to this state
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
void write_state_reordered(const unsigned int &qubitstate_size, const std::vector<Eigen::VectorXcd> &state, const unsigned int &TimeSize, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/states/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
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
inline void transform_evolved_state(const std::vector<Eigen::VectorXcd> &pevolved_state, std::vector<Eigen::VectorXcd> &finstate) {
    const double sqrtDInv = 1/sqrt(D);
    #pragma omp parallel for
    for (unsigned int m = 0; m < D; m++) {
        finstate[m] = sqrtDInv * pevolved_state[0];
        for (unsigned int p = 1; p < D; p++) {
            finstate[m] += sqrtDInv * omega(-int(m*p)) * pevolved_state[p];
        }
    }
}

// Calculates all sums of Q^2 and returns P(m)*sum Q^2 in Qsums, with shape (D,steps_taken). Assumes Qsums already contains the correct number of vectors, but initializes each vector to a steps_taken number of 0s. Can be parallelized by taking each m independently. Maybe could be optimized changind Qsums to an Eigen matrix/vectors.
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

// For now, it just goes from 0 to T stupidly and it doesn't parallelize anything
inline void generate_and_evolve_seed_toT(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const unsigned int &T, const std::string &seed_filename, const std::string &Qsums_filename, const std::string &probs_filename, const std::string &squared_Qfuncs_filename, const std::string &output_filename, const bool &save_state, const bool &calculate_qsums, const bool &calculate_probabilities, const unsigned int &calculate_final_Qfuncs, const unsigned int &interaction_pattern) {
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
    std::vector<unsigned int> interaction_seed(T);
    std::cout << "Parsing seed" << std::endl;
    parse_interaction_seed(seed_filename,0,T,interaction_seed);
    std::cout << "Initializing qubits" << std::endl;
    std::vector<std::vector<Eigen::Vector2cd>> fullqubitstates_buffers(D);
    initialize_qubitstates_buffer(n_qubits,0,interaction_seed,fullqubitstates_buffers);
    std::cout << "Evolving state" << std::endl;
    std::vector<Eigen::VectorXcd> full_pevolved_state(D);
    evolution_fromt_toT(n_qubits,qubitstate_size,0,T,interaction_seed,fullqubitstates_buffers,full_pevolved_state);
    std::cout << "Transforming state" << std::endl;
    std::vector<Eigen::VectorXcd> finstate(D);
    transform_evolved_state(full_pevolved_state,finstate);
    if (calculate_qsums) {
        std::vector<Eigen::VectorXd> Qsums(D);
        if (calculate_probabilities) {
            std::cout << "Calculating sums of Q^2 and probabilities" << std::endl;
            std::vector<Eigen::VectorXd> probs(D);
            auto start = std::chrono::high_resolution_clock::now();
            calculate_sumofQ2_with_probs(n_qubits,qubitstate_size,finstate,T,Qsums,probs);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> duration = end - start;
            std::cout << "Calculating took " << duration.count() << "s" << std::endl;
            std::cout << "Saving probabilies" << std::endl;
            save_probs(probs,probs_filename);
            if (calculate_final_Qfuncs) {
                std::cout << "Calculating final Q^2" << std::endl;
                std::vector<Eigen::MatrixXd> squared_Qfuncs(D,Eigen::MatrixXd::Zero(qubitstate_size,qubitstate_size));
                calculate_squared_Qfuncs_at_T(n_qubits,qubitstate_size,T-1,finstate,probs,squared_Qfuncs); // Calculated at T-1, since T is not evaluated
                average_squared_Qfuncs(squared_Qfuncs);
                std::cout << "Saving final Q^2" << std::endl;
                save_squared_Qfuncs(squared_Qfuncs,squared_Qfuncs_filename);
            }
        } else{
            std::cout << "Calculating sums of Q^2" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            calculate_sumofQ2(n_qubits,qubitstate_size,finstate,T,Qsums);
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
        calculate_probs(qubitstate_size,T,finstate,probs);
        std::cout << "Saving probabilies" << std::endl;
        save_probs(probs,probs_filename);
        if (calculate_final_Qfuncs) {
            std::cout << "Calculating final Q^2" << std::endl;
            std::vector<Eigen::MatrixXd> squared_Qfuncs(D,Eigen::MatrixXd::Zero(qubitstate_size,qubitstate_size));
            calculate_squared_Qfuncs_at_T(n_qubits,qubitstate_size,T-1,finstate,probs,squared_Qfuncs);
            average_squared_Qfuncs(squared_Qfuncs);
            std::cout << "Saving final Q^2" << std::endl;
            save_squared_Qfuncs(squared_Qfuncs,squared_Qfuncs_filename);
        }
    }
    if (save_state) {
        std::cout << "Writing reordered state" << std::endl;
        write_state_reordered(qubitstate_size,finstate,T,output_filename);
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
    generate_and_evolve_seed_toT(n_qubits,qubitstate_size,max_time,seed_filename,Qsums_filename,probs_filename,squared_Qfuncs_filename,output_filename,save_state,calculate_qsums,calculate_probabilities,calculate_squared_Qfuncs,interaction_pattern);
}

void batch_coin_operator_multicqw(){
    const unsigned int n_qubits = 2;
    const unsigned int qubitstate_size = 1 << n_qubits;
    const unsigned int max_time = 5000;
    unsigned int betas_res = 100;
    unsigned int gammas_res = 100;
    std::string input = "";
    std::cout << "Enter the resolution in beta" << std::endl;
    std::cin >> input;
    parse_unsignedint(input,betas_res);
    input = "";
    std::cout << "Enter the resolution in gamma" << std::endl;
    std::cin >> input;
    parse_unsignedint(input,gammas_res);
    const Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(betas_res, 0, M_PI/2);
    const Eigen::VectorXd gammas = Eigen::VectorXd::LinSpaced(gammas_res, 0, 2*M_PI);
    bool save_state = true;
    bool calculate_qsums = false;
    bool calculate_probabilities = false;
    bool calculate_squared_Qfuncs = false;
    unsigned int interaction_pattern = 1;
    for (unsigned int b = 0; b < betas_res; b++) {
        for (unsigned int g = 0; g < gammas_res; g++) {
            const unsigned int n = b * gammas_res + g;
            std::string seed_filename = "batch_coins/"+std::to_string(n)+".txt";
            std::string Qsums_filename = "batch_coins/"+std::to_string(n)+".txt";
            std::string probs_filename = "batch_coins/"+std::to_string(n)+".txt";
            std::string output_filename = "batch_coins/"+std::to_string(n)+".txt";
            std::string squared_Qfuncs_filename = "batch_coins/"+std::to_string(n)+".txt";
            beta = betas[b];
            gamma = gammas[g];
            generate_and_evolve_seed_toT(n_qubits,qubitstate_size,max_time,seed_filename,Qsums_filename,probs_filename,squared_Qfuncs_filename,output_filename,save_state,calculate_qsums,calculate_probabilities,calculate_squared_Qfuncs,interaction_pattern);
        }
    }
}

int main() {
    batch_coin_operator_multicqw();
    
    // const unsigned int qubits[2] = {9,10};
    // const unsigned int max_time = 5000;
    // const unsigned int interaction_pattern = 0;
    // char prefix = 'r';
    // bool save_state = false;
    // bool calculate_qsums = true;
    // bool calculate_probabilities = true;
    // for (unsigned int n_qubits : qubits) {
    //     std::cout << n_qubits << std::endl;
    //     const unsigned int qubitstate_size = 1 << n_qubits;
    //     std::string seed_filename = "seed_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    //     std::string Qsums_filename = "qsums_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    //     std::string probs_filename = "probs_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    //     std::string output_filename = "state_dim" + std::to_string(D) + "_q" + std::to_string(n_qubits) + ".txt";
    //     seed_filename.insert(seed_filename.begin(),prefix);
    //     Qsums_filename.insert(Qsums_filename.begin(),prefix);
    //     probs_filename.insert(probs_filename.begin(),prefix);
    //     output_filename.insert(output_filename.begin(),prefix);
    //     generate_and_evolve_seed_toT(n_qubits,qubitstate_size,max_time,seed_filename,Qsums_filename,probs_filename,output_filename,save_state,calculate_qsums,calculate_probabilities,interaction_pattern);
    // }

    return 0;
}