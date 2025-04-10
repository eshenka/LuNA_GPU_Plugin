#include <atomic>
#include <cassert>
#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <cuda_runtime.h>
#include "df.h"

class DFCuda
{
public:

    DFCuda(const std::vector<int64_t> &dims);

    ~DFCuda();

    size_t getDataSize();

    bool allocateCuda();

    bool copyToCuda(const DF &df);

    bool copyFromCuda(DF &df);

    bool copyFromCuda(void *ptr);

private:
    size_t size_;
    std::vector<int64_t> dims_;
    void *device_ptr_;
};