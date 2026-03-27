// Your CUDA kernel and C++ launcher go here
#include <torch/extension.h>


__global__ void vec_sum(const float4* A, float* out, int N) {
    extern __shared__ float sdata[];
    int stride = gridDim.x * blockDim.x;
    float sum{};

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x ; idx < N / 4 ; idx += stride) {
        float4 a = A[idx];
        sum += a.x + a.y + a.z + a.w;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset){
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        atomicAdd(out, sdata[0]);
    }
}

__global__ void vec_sum_tail(const float* A, float* out, int start, int N) {
    int idx = start + threadIdx.x;
    if (idx < N) {
        atomicAdd(out, A[idx]);
    }
}




torch::Tensor& vecsum(const torch::Tensor& in, torch::Tensor& out) {
    out.zero_();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int N = in.numel();
    int threads{256};
    int shared_mem_size = threads * sizeof(float);

    int blocks_per_SM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_SM, vec_sum, threads, shared_mem_size);

    int blocks = prop.multiProcessorCount * blocks_per_SM;

    int tail_start = (N / 4) * 4;
    int tail_count = N - tail_start;
    if (tail_count > 0) {
        vec_sum_tail<<<1, 256, 256 * sizeof(float)>>>(
            in.data_ptr<float>(),
            out.data_ptr<float>(),
            tail_start, N
        );
    }

    vec_sum<<<blocks, threads, threads * sizeof(float)>>>(
        reinterpret_cast<const float4*>(in.data_ptr<float>()),
        out.data_ptr<float>(),
        N
    );
    return out;
}
