#include "seed_generator.h"
#include<iostream>
#include<filesystem>
#include<cstdlib>
#include<ctime>
#include<fstream>

// Generates a seed where the interaction pattern is random
void generate_random_seed(const unsigned int &n_qubits, const unsigned int &max_time, const std::string &output_filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/seeds/"+output_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::srand(std::time(nullptr));
    if (output_file.is_open()) {
        for (unsigned int n = 0; n < max_time; n++) {
            output_file << std::rand() % n_qubits << std::endl;
        }
    } else {
            std::cout << "Failed to write file" << std::endl;
    }
    output_file.close();
}

// Generates a seed where the qubits interact orderly
void generate_ordered_seed(const unsigned int &n_qubits, const unsigned int &max_time, const std::string &output_filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/seeds/"+output_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (output_file.is_open()) {
        for (unsigned int n = 0; n < max_time; n++) {
            output_file << n % n_qubits << std::endl;
        }
    } else {
            std::cout << "Failed to write file" << std::endl;
    }
    output_file.close();
}

// Generates a seed where the last qubit interacts only once at the beginning, and the rest interact orderly
void generate_biased_seed(const unsigned int &n_qubits, const unsigned int &max_time, const std::string &output_filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/seeds/"+output_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (output_file.is_open()) {
        output_file << n_qubits - 1 << std::endl;
        for (unsigned int n = 0; n < max_time - 1; n++) {
            output_file << n % (n_qubits - 1) << std::endl;
        }
    } else {
            std::cout << "Failed to write file" << std::endl;
    }
    output_file.close();
}

// Generates a seed where only the first qubit interacts more than once, the rest interact orderly at the beginning. Takes the number of steps indicated by max_time, which must be at least n_qubits long.
void generate_completely_biased_seed(const unsigned int &n_qubits, const unsigned int &max_time, const std::string &output_filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/seeds/"+output_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (max_time < n_qubits) {
        std::cout << "Not enough steps to generate a biased seed, returning a null walk" << std::endl;
        output_file.close();
        return;
    }
    if (output_file.is_open()) {
        for (unsigned int qubit = 1; qubit < n_qubits; qubit++) {
            output_file << qubit << std::endl;
        }
        for (unsigned int n = 0; n < max_time - n_qubits + 1; n++) {
            output_file << 0 << std::endl;
        }
    } else {
            std::cout << "Failed to write file" << std::endl;
    }
    output_file.close();
}
