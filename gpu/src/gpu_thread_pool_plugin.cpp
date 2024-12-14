#include "gpu_threadpool_plugin.h"
#include "gpu_thread_pool.h"
#include "common.h"
#include <cuda_runtime.h>

AbstractThreadPoolPlugin *GPUThreadPoolPlugin::instance = nullptr;

extern "C"
{
    AbstractThreadPoolPlugin *gpu_threadpool_create_plugin()
    {
        if (GPUThreadPoolPlugin::instance != nullptr)
        {
            ABORT("creating GPUThreadPoolPlugin instance again");
        }
        GPUThreadPoolPlugin::instance = new GPUThreadPoolPlugin();
        return GPUThreadPoolPlugin::instance;
    }
}

const std::string &GPUThreadPoolPlugin::get_name()
{
    return name;
}

GPUThreadPoolPlugin::GPUThreadPoolPlugin()
{
    name = "gpu_threadpool";
}

GPUThreadPoolPlugin::~GPUThreadPoolPlugin()
{
    // Cleanup if needed
}

void GPUThreadPoolPlugin::set_default_flags(Config *config)
{
    config->set_bool_config("--use-gpu", false);
    config->set_bool_config("--gpu-use-timer", false);
    config->set_int_config("--worker-threads-count=", 4);
    // config->set_int_config("--gpu-memory-pool-size=", 1024);
}

bool GPUThreadPoolPlugin::accept_flag(std::string flag, Config *config)
{
    if (flag == "--use-gpu")
    {
        LOG("[gpu_threadpool] flag --use-gpu found");
        config->set_bool_config("--use-gpu", true);
    }
    else if (flag == "--gpu-use-timer")
    {
        LOG("[gpu_threadpool] flag --gpu-use-timer found");
        config->set_bool_config("--gpu-use-timer", true);
    }
    // else if (flag.find("--gpu-memory-pool-size=") == 0)
    // {
    //     try {
    //         int value = std::stoi(flag.substr(22));
    //         if (value < 1)
    //         {
    //             throw std::invalid_argument("Invalid --gpu-memory-pool-size value: argument < 1");
    //         }
    //         config->set_int_config("--gpu-memory-pool-size=", value);
    //     }
    //     catch (const std::invalid_argument& invalid_argument_exception) {
    //         ABORT(invalid_argument_exception.what())
    //     }
    //     catch (const std::out_of_range& out_of_range_exception) {
    //         ABORT(out_of_range_exception.what())
    //     }
    //     LOG("[gpu_threadpool] flag --gpu-memory-pool-size found");
    // }
    else if (flag.find("--worker-threads-count=") == 0)
    {
        try {
            int value = std::stoi(flag.substr(23));
            if (value < 1)
            {
                throw std::invalid_argument("Invalid --worker-threads-count value: argument < 1");
            }
            config->set_int_config("--worker-threads-count=", value);
        }
        catch (const std::invalid_argument& invalid_argument_exception) {
            ABORT(invalid_argument_exception.what())
        }
        catch (const std::out_of_range& out_of_range_exception) {
            ABORT(out_of_range_exception.what())
        }
        LOG("[gpu_threadpool] flag --worker-threads-count found");
    }
    else
    {
        return false;
    }
    return true;
}


// The purpose of this method is to initialize the components necessary for the plugin to work 
// (for example, most libraries require calling methods at the beginning of the program to initialize the library).
// The method returns true if all components required for the plugin have been successfully initialized. 
// false otherwise (in this case the running program will end).
bool GPUThreadPoolPlugin::init(const Config *config)
{
    if (!config->get_bool_config("--use-gpu"))
    {
        return true;
    }

    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess)
    {
        LOG("[gpu_threadpool] No CUDA-capable devices found");
        return false;
    }

    // std::string deviceList = config->get_string_config("--gpu-dev-ids=");
    // if (deviceList.empty())
    // {
    //     LOG("[gpu_threadpool] No GPU devices specified, using device 0");
    //     deviceList = "0";
    // }

    std::string deviceList = "0";
    std::istringstream iss(deviceList);
    std::string token;
    while (std::getline(iss, token, ','))
    {
        try {
            int deviceId = std::stoi(token);
            if (deviceId >= deviceCount)
            {
                LOG("[gpu_threadpool] Invalid device ID: " + token);
                return false;
            }

            cudaError_t error = cudaSetDevice(deviceId);
            if (error != cudaSuccess)
            {
                LOG("[gpu_threadpool] Failed to initialize device " + token);
                return false;
            }

            cudaDeviceProp props;
            error = cudaGetDeviceProperties(&props, deviceId);
            if (error != cudaSuccess)
            {
                LOG("[gpu_threadpool] Failed to get device properties for device " + token);
                return false;
            }

            LOG("[gpu_threadpool] Successfully initialized GPU " + token + 
                " (" + props.name + ")");
        }
        catch (const std::exception& e)
        {
            LOG("[gpu_threadpool] Error parsing device ID: " + token);
            return false;
        }
    }

    // size_t poolSize = config->get_int_config("--gpu-memory-pool-size=") * 1024 * 1024; // Convert MB to bytes
    // if (poolSize > 0)
    // {
    //     cudaError_t error = cudaDeviceSetLimit(cudaLimitMallocHeapSize, poolSize);
    //     if (error != cudaSuccess)
    //     {
    //         LOG("[gpu_threadpool] Failed to set GPU memory pool size");
    //         return false;
    //     }
    // }

    return true;
}

bool GPUThreadPoolPlugin::destroy(const Config *config)
{
    if (!config->get_bool_config("--use-gpu"))
    {
        return true;
    }

    // std::string deviceList = config->get_string_config("--gpu-dev-ids=");
    std::string deviceList = "0";
    std::istringstream iss(deviceList);
    std::string token;
    
    while (std::getline(iss, token, ','))
    {
        try {
            int deviceId = std::stoi(token);
            cudaError_t error = cudaSetDevice(deviceId);
            if (error == cudaSuccess)
            {
                error = cudaDeviceReset();
                if (error != cudaSuccess)
                {
                    LOG("[gpu_threadpool] Failed to reset device " + token);
                    return false;
                }
            }
        }
        catch (const std::exception& e)
        {
            LOG("[gpu_threadpool] Error cleaning up device " + token);
            return false;
        }
    }

    return true;
}



AbstractThreadPool *GPUThreadPoolPlugin::create_thread_pool(const Config *conf_)
{
    if (conf_->get_bool_config("--use-gpu"))
    {
        return new GPUThreadPool(conf_);
    }
    else
    {
        return nullptr;
    }
}