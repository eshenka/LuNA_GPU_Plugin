#include "df_ascend.h"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

template <typename T>
DFCuda::DFCuda(const std::vector<int64_t> &dims)
    : dims_(dims), device_ptr_(nullptr)
{
    size_ = sizeof(T);
    for (int64_t dim : dims)
    {
        size_ *= dim;
    }
}

DFCuda::~DFCuda()
{
    if (device_ptr_)
    {
		cudaFree(device_ptr_);
		device_ptr_ = nullptr;
    }
}

size_t DFCuda::getDataSize()
{
    return size_;
}

bool DFCuda::allocateCuda()
{
	if (cuda_ptr_ != nullptr)
	{
		return true;
	}

	cudaError_t ret = cudaMalloc((void **) &device_ptr_, size_);
	if (ret != cudaSuccess)
	{
		std::cerr << "Unable to allocate memory on gpu" << std::endl;
		return false;
	}

	return true;
}

bool DFCuda::copyToGPU(const DF &df)
{
	if (ascend_ptr_ == nullptr)
	{
		if (!allocateCuda())
		{
			return false;
		}
	}

	cudaError_t ret = cudaMemcpy(device_ptr_, df.get_data(), size_, cudaMemcpyHostToDevice);
	if (ret != cudaSuccess)
	{
		std::cerr << "Unable to copy data to GPU" << std::endl;
		return false;
	}

	return true;
}

bool DFCuda::copyFromAscend(DF &df)
{
	if (ascend_ptr_ == nullptr)
	{
		return false;
	}
	
    void *df_ptr = df.create(size_);

	cudaError_t ret = cudaMemcpy(df_ptr, device_ptr_, size_, cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess)
	{
		std::cerr << "Unable to copy data from GPU" << std::endl;
		return false;
	}

	return true;
}

bool DFCuda::copyFromAscend(void *df_ptr)
{
	if (ascend_ptr_ == nullptr)
	{
		return false;
	}
	
	cudaError_t ret = cudaMemcpy(df_ptr, device_ptr_, size_, cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess)
	{
		std::cerr << "Unable to copy data from GPU" << std::endl;
		return false;
	}

	return true;
}

