#pragma once

#include <cublas_v2.h>

#include <cstdint>

#include "common.cuh"

void sgemmsb_cublas(Tensor<float> a, Tensor<float> b, Tensor<float> c, float alpha,
                    float beta, int64_t stra, int64_t strb, int64_t strc) {
    const int B = c.shape(0);
    const int M = c.shape(1);
    const int N = c.shape(2);
    const int K = a.shape(2);
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    stra = stra >= 0 ? stra : a.stride(0);
    strb = strb >= 0 ? strb : b.stride(0);
    strc = strc >= 0 ? strc : c.stride(0);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              b.data(), ldb, strb, a.data(), lda, stra, &beta, c.data(),
                              ldc, strc, B);
    cublasDestroy(handle);
}
