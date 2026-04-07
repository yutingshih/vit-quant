#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

template <typename T>
using Tensor = nb::ndarray<T, nb::device::cuda, nb::c_contig>;
