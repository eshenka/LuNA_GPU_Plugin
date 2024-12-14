#include <atomic>
#include <cassert>
#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <cuda_runtime.h>
#include "df.h"

class DFAscend
{
public:

    DFAscend(aclDataType dataType, const std::vector<int64_t> &dims);

    ~DFAscend();

    size_t getDataSize();

    bool allocateCuda();

    bool copyToCuda(const DF &df);

    bool copyFromCuda(DF &df);

    bool copyFromCuda(void *ptr);

private:
    size_t size_;
    aclDataType data_type_;
    std::vector<int64_t> dims_;
    void *device_ptr_;
};