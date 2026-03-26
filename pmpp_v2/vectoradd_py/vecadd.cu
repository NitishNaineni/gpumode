#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ void vec_add(const float4* A, const float4* B, float4* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 8) {
        float4 a_val = A[idx];
        float4 b_val = B[idx];
        float4 out_val;
        const half2* a = reinterpret_cast<const half2*>(&a_val);
        const half2* b = reinterpret_cast<const half2*>(&b_val);
        half2* o = reinterpret_cast<half2*>(&out_val);

        for (int i{}; i < 4 ; i++){
            o[i] = __hadd2(a[i], b[i]);
        }

        out[idx] = out_val;
    } else if (idx == N / 8) {
        const half* a = reinterpret_cast<const half*>(A);
        const half* b = reinterpret_cast<const half*>(B);
        half* o = reinterpret_cast<half*>(out);

        for (int i{(N / 8) * 8}; i < N ; i++){
            o[i] = __hadd(a[i], b[i]);
        }

    }
}

torch::Tensor& vecadd(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out) {
    int N = A.numel();
    int threads = 256;
    int blocks = (N / 8 + threads) / threads;
    vec_add<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(out.data_ptr<at::Half>()),
        N
    );
    return out;
}
