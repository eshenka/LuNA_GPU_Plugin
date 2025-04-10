#include "gpu_code_generator.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <cstdio>

GPUCodeGenerator::GPUCodeGenerator(const fs::path& build_dir)
    : build_dir_(build_dir)
{
    cuda_include_path_ = fs::path(std::getenv("CUDA_PATH")) / "include";
    nvcc_path_ = fs::path(std::getenv("CUDA_PATH")) / "bin" / "nvcc";
    
    verify_gpu_compatibility();
}

bool GPUCodeGenerator::generate_gpu_code(const std::string& json_file) {
    std::ifstream f(json_file);
    json ja = json::parse(f);

    fs::create_directories(build_dir_ / "cuda_kernels");
    fs::create_directories(build_dir_ / "ptx");

    std::string func_name = create_function_name(ja["sub"], ja);
    if (!generate_cuda_kernel(ja)) {
        std::cerr << "Failed to generate CUDA kernel\n";
        return false;
    }

    std::string cuda_file = (build_dir_ / "cuda_kernels" / (func_name + ".cu")).string();
    if (!compile_cuda_code(cuda_file)) {
        std::cerr << "Failed to compile CUDA code\n";
        return false;
    }

    if (!generate_ptx_code(cuda_file)) {
        std::cerr << "Failed to generate PTX code\n";
        return false;
    }

    std::string gpu_imports = parse_imports(ja);
    std::ofstream header_file(build_dir_ / (ja["code"].get<std::string>() + ".cuh"));
    header_file << gpu_imports;

    return true;
}

std::vector<std::string> GPUCodeGenerator::rules_to_cf_param(const json& rules) {
    bool is_nocpu = false;
    for (const auto& rule : rules) {
        if (rule["ruletype]" == "flags" && rule["flags"].contains("NOCPU")) {
            is_nocpu = true;
            break;
        }
    }
    if is_nocpu {
        return std::vector<std::string>{"GPU"};
    } else {
        return std::vector<std::string>{"HYBRID"};
    }
}

std::string GPUCodeGenerator::create_function_name(const json& sub, const json& ja) {
    std::string device_type = rules_to_cf_param(sub["rules"])[0];
    std::string param_types;
    for (const auto& type : parse_gpu_types(ja)) {
        param_types += type;
    }
    return sub["code"].get<std::string>() + "_" + param_types + "_" + device_type;
}

bool GPUCodeGenerator::generate_cuda_kernel(const json& ja) {
    std::string code = ja["code"].get<std::string>();
    std::string kernel_file = (build_dir_ / "cuda_kernels" / (code + ".cu")).string();
    
    std::ofstream f(kernel_file);
    f << "#include <cuda_runtime.h>\n\n";
    
    // Generate kernel template
    f << "template<typename T>\n";
    f << "__global__ void " << code << "_kernel(";
    
    // Add parameters based on JSON definition
    
    f << ") {\n";
    // Generate kernel body
    f << "}\n";
    
    return true;
}

bool GPUCodeGenerator::compile_cuda_code(const std::string& cuda_file) {
    std::vector<std::string> nvcc_args = {
        nvcc_path_.string(),
        "-c",
        "-O3",
        "--compiler-options",
        "'-fPIC'",
        "-I", cuda_include_path_.string(),
        cuda_file
    };
    
    for (const auto& arch : gpu_architectures_) {
        nvcc_args.push_back("-gencode");
        nvcc_args.push_back("arch=compute_" + arch.substr(3) + 
                           ",code=" + arch);
    }
    
    return run_nvcc_command(nvcc_args);
}

bool GPUCodeGenerator::generate_ptx_code(const std::string& cuda_file) {
    std::vector<std::string> nvcc_args = {
        nvcc_path_.string(),
        "-ptx",
        cuda_file,
        "-o", (build_dir_ / "ptx" / fs::path(cuda_file).stem()).string() + ".ptx"
    };
    
    return run_nvcc_command(nvcc_args);
}

bool GPUCodeGenerator::run_nvcc_command(const std::vector<std::string>& args) {
    std::string cmd;
    for (const auto& arg : args) {
        cmd += arg + " ";
    }
    
    return system(cmd.c_str()) == 0;
}

bool GPUCodeGenerator::verify_gpu_compatibility() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "No CUDA-capable devices found\n";
        return false;
    }
    
    // Check each device's compute capability
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::string sm = "sm_" + std::to_string(prop.major * 10 + prop.minor);
        if (std::find(gpu_architectures_.begin(), 
                     gpu_architectures_.end(), 
                     sm) == gpu_architectures_.end()) {
            gpu_architectures_.push_back(sm);
        }
    }
    
    return true;
}
