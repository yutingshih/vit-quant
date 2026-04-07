#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "gemmsb_coalesced.cuh"
#include "sgemmsb_cublas.cuh"
#include "vadd.cuh"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(kernels, m) {
    m.doc() = "Custom CUDA kernels";
    m.def("vadd", &vadd, "a"_a, "b"_a, "c"_a);
    m.def("sgemmsb_cublas", &sgemmsb_cublas, "a"_a, "b"_a, "c"_a, "alpha"_a = 1.0f,
          "beta"_a = 0.0f, "stra"_a = -1LL, "strb"_a = -1LL, "strc"_a = -1LL);
    m.def("gemmsb_coalesced", &gemmsb_coalesced<float, float>, "a"_a, "b"_a, "c"_a,
          "alpha"_a = 1.0f, "beta"_a = 0.0f, "stra"_a = -1LL, "strb"_a = -1LL,
          "strc"_a = -1LL);
}
