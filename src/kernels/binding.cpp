#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>

#include "tensor.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void vadd(nb::ndarray<float> a, nb::ndarray<float> b, nb::ndarray<float> c);
void sgemmsb_cublas(Tensor<float> a, Tensor<float> b, Tensor<float> c, float alpha,
                    float beta, int64_t stra, int64_t strb, int64_t strc);

NB_MODULE(kernels, m) {
    m.doc() = "Custom CUDA kernels";
    m.def("vadd", &vadd, "a"_a, "b"_a, "c"_a);
    m.def("sgemm_cublas", &sgemmsb_cublas, "a"_a, "b"_a, "c"_a, "alpha"_a = 1.0f,
          "beta"_a = 0.0f, "stra"_a = -1LL, "strb"_a = -1LL, "strc"_a = -1LL);
}
