#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>  // for JSON parsing
#include <filesystem>
#include <cuda_runtime.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

class GPUCodeGenerator {
public:
    GPUCodeGenerator(const fs::path& build_dir);
    ~GPUCodeGenerator();

    // Main function to process and generate GPU code
    bool generate_gpu_code(const std::string& json_file);

private:
    // Similar functionality to fcmp2.py
    std::string create_function_name(const json& sub, const json& ja);
    std::vector<std::string> parse_gpu_types(const json& ja);
    std::vector<std::string> rules_to_cf_param(const json& rules);
    std::string get_gpu_name_with_underscores(const std::string& name);
    std::string parse_imports(const json& ja);

    // GPU-specific code generation
    bool generate_cuda_kernel(const json& ja);
    bool compile_cuda_code(const std::string& cuda_file);
    bool generate_ptx_code(const std::string& cuda_file);
    
    // Utility functions
    bool run_nvcc_command(const std::vector<std::string>& args);
    bool verify_gpu_compatibility();

private:
    fs::path build_dir_;
    fs::path cuda_include_path_;
    fs::path nvcc_path_;
    std::vector<std::string> gpu_architectures_;  // e.g., "sm_70", "sm_75"
};