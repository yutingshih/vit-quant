#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

__global__ void _vadd(const float* a, const float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

void vadd(nb::ndarray<float> a, nb::ndarray<float> b, nb::ndarray<float> c) {
    size_t size = a.size();
    dim3 bs(32);
    dim3 gs(1 + (size - 1) / bs.x);

    _vadd<<<gs, bs>>>(a.data(), b.data(), c.data(), size);
}
