#ifndef SEED_GENERATOR_H
#define SEED_GENERATOR_H

#include<string>

void generate_random_seed(const unsigned int &n_qubits, const unsigned int &max_time, const std::string &output_filename);
void generate_ordered_seed(const unsigned int &n_qubits, const unsigned int &max_time, const std::string &output_filename);
void generate_biased_seed(const unsigned int &n_qubits, const unsigned int &max_time, const std::string &output_filename);
void generate_completely_biased_seed(const unsigned int &n_qubits, const unsigned int &max_time, const std::string &output_filename);

#endif