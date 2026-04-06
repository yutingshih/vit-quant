#include <cublas_v2.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

void sgemmsb_cublas(const nb::ndarray<float> a, const nb::ndarray<float> b,
                    nb::ndarray<float> c, float alpha, float beta) {
    const int B = c.shape(0);
    const int M = c.shape(1);
    const int N = c.shape(2);
    const int K = a.shape(2);
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    const long long int stra = M * K;
    const long long int strb = K * N;
    const long long int strc = M * N;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              b.data(), ldb, strb, a.data(), lda, stra, &beta, c.data(),
                              ldc, strc, B);
    cublasDestroy(handle);
}
