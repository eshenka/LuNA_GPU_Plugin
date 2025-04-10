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
    config->set_int_config("--worker-threads-count=", 4);
}

bool GPUThreadPoolPlugin::accept_flag(std::string flag, Config *config)
{
    if (flag == "--use-gpu")
    {
        LOG("[gpu_threadpool] flag --use-gpu found");
        config->set_bool_config("--use-gpu", true);
    }
    // else if (flag == "--gpu-use-timer")
    // {
    //     LOG("[gpu_threadpool] flag --gpu-use-timer found");
    //     config->set_bool_config("--gpu-use-timer", true);
    // }
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

bool GPUThreadPoolPlugin::init(const Config *config)
{
    if (!config->get_bool_config("--use-gpu"))
    {
        return true;
    }

    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0)
    {
        LOG("[gpu_threadpool] No CUDA-capable devices found");
        return false;
    }


    int deviceId = 0;

    cudaError_t error = cudaSetDevice(deviceId);
    if (error != cudaSuccess)
    {
        LOG("[gpu_threadpool] Failed to initialize device " + std::to_string(deviceId));
        return false;
    }

    cudaDeviceProp props;
    error = cudaGetDeviceProperties(&props, deviceId);
    if (error != cudaSuccess)
    {
        LOG("[gpu_threadpool] Failed to get device properties for device " + std::to_string(deviceId));
        return false;
    }

    LOG("[gpu_threadpool] Successfully initialized GPU " + std::to_string(deviceId) + 
        " (" + props.name + ")");
    

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


    cudaError_t error = cudaDeviceReset();
    if (error == cudaSuccess)
    {
        LOG("[gpu_threadpool] Failed to reset device " + std::to_string(deviceId));
        return false;
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