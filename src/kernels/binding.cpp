#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

void vadd(nb::ndarray<float> a, nb::ndarray<float> b, nb::ndarray<float> c);
void sgemmsb_cublas(const nb::ndarray<float> a, const nb::ndarray<float> b,
                    nb::ndarray<float> c, float alpha, float beta);

NB_MODULE(kernels, m) {
    m.doc() = "Custom CUDA kernels";
    m.def("vadd", &vadd, "a"_a, "b"_a, "c"_a);
    m.def("sgemm_cublas", &sgemmsb_cublas, "a"_a, "b"_a, "c"_a, "alpha"_a, "beta"_a);
}
