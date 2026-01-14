#include "common.h"
#include "nanovdb.cuh"


template void runNanoVDB<slimvdb::CLOSED, NCLASSES>(
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    int, int,
    nanovdb::cuda::DeviceBuffer&,
    int,
    std::vector<double>,
    std::vector<double>,
    const float,
    const float,
    const float, 
    const float*,
    const uint16_t
);

template void runNanoVDB<slimvdb::OPEN, NCLASSES>(
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    int, int,
    nanovdb::cuda::DeviceBuffer&,
    int,
    std::vector<double>,
    std::vector<double>,
    const float,
    const float,
    const float,
    const float*,
    const uint16_t
);