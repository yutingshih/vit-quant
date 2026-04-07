#pragma once

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                               \
    do {                                               \
        cudaError_t err = expr;                        \
        if (err != cudaSuccess) {                      \
            std::string msg = cudaGetErrorString(err); \
            throw std::runtime_error(msg);             \
        }                                              \
    } while (0)

namespace nb = nanobind;

template <typename T>
using Tensor = nb::ndarray<T, nb::device::cuda, nb::c_contig>;
