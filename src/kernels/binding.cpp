#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

void vadd(nb::ndarray<float> a, nb::ndarray<float> b, nb::ndarray<float> c);

NB_MODULE(kernels, m) {
    m.doc() = "Custom CUDA kernels";
    m.def("vadd", &vadd, "a"_a, "b"_a, "c"_a);
}
