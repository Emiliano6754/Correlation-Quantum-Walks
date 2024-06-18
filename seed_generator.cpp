#include "seed_generator.h"
#include<iostream>
#include<filesystem>
#include<cstdlib>
#include<ctime>
#include<fstream>

void generate_seed(const unsigned int &qubit_number, const unsigned int &max_time, const std::string &output_filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/"+output_filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    std::srand(std::time(nullptr));
    if (output_file.is_open()) {
        for (unsigned int n = 0; n < max_time; n++) {
            output_file << std::rand() % qubit_number << std::endl;
        }
    } else {
            std::cout << "Failed to write file" << std::endl;
    }
    output_file.close();
}