#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

#include "common.cuh"

template <typename TypeData, typename TypeAcc = TypeData>
__global__ void _gemmsb_coalesced(TypeData* a, TypeData* b, TypeData* c, uint32_t B,
                                  uint32_t M, uint32_t N, uint32_t K, TypeAcc alpha,
                                  TypeAcc beta, int64_t stra, int64_t strb,
                                  int64_t strc) {
    int bat = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (bat >= B || row >= M || col >= N) return;

    TypeAcc sum{};
    for (int k = 0; k < K; k++) {
        sum += static_cast<TypeAcc>(a[bat * stra + row * K + k]) *
               static_cast<TypeAcc>(b[bat * strb + k * N + col]);
    }
    c[bat * strc + row * N + col] =
        static_cast<TypeData>(alpha * sum + beta * c[bat * strc + row * N + col]);
}

template <typename TypeData, typename TypeAcc = TypeData>
void gemmsb_coalesced(Tensor<TypeData> a, Tensor<TypeData> b, Tensor<TypeData> c,
                      TypeAcc alpha, TypeAcc beta, int64_t stra, int64_t strb,
                      int64_t strc) {
    const int B = c.shape(0);
    const int M = c.shape(1);
    const int N = c.shape(2);
    const int K = a.shape(2);
    stra = stra >= 0 ? stra : a.stride(0);
    strb = strb >= 0 ? strb : b.stride(0);
    strc = strc >= 0 ? strc : c.stride(0);

    dim3 bs(16, 16, 4);
    dim3 gs(div_ceil(N, bs.x), div_ceil(M, bs.y), div_ceil(B, bs.z));

    _gemmsb_coalesced<TypeData, TypeAcc><<<gs, bs>>>(
        a.data(), b.data(), c.data(), B, M, N, K, alpha, beta, stra, strb, strc);
    CUDA_CHECK(cudaGetLastError());
}
