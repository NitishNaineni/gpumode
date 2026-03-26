#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu A100

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = """#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ __launch_bounds__(256, 8) void vec_add(
    const float4* __restrict__ A,
    const float4* __restrict__ B,
    float4* __restrict__ out,
    int N_vec) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_vec) {
        float4 a_val = A[idx];
        float4 b_val = B[idx];
        float4 out_val;

        const half2* a = reinterpret_cast<const half2*>(&a_val);
        const half2* b = reinterpret_cast<const half2*>(&b_val);
        half2* o = reinterpret_cast<half2*>(&out_val);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            o[i] = __hadd2(a[i], b[i]);
        }

        out[idx] = out_val;
    }
}

__global__ void vec_add_tail(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ out,
    int start, int N) {

    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __hadd(A[idx], B[idx]);
    }
}

torch::Tensor& vecadd(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out) {
    int N = A.numel();
    int N_vec = N / 8;
    int threads = 256;
    int blocks = (N_vec + threads - 1) / threads;

    vec_add<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(out.data_ptr<at::Half>()),
        N_vec
    );

    int tail_start = N_vec * 8;
    int tail_count = N - tail_start;
    if (tail_count > 0) {
        vec_add_tail<<<1, 256>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            tail_start, N
        );
    }

    return out;
}
"""

CPP_SRC = """// Your C++ function declarations go here
torch::Tensor& vecadd(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out);
"""

module = load_inline(
    name='vecadd_module',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['vecadd'],
    verbose=True,
    extra_cuda_cflags=['-arch=sm_80', '--use_fast_math'],
)

def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    return module.vecadd(A, B, output)
