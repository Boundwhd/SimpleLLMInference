#include "matmul_kernel.cuh"

namespace kernel {

    template<const int NUM_THREADS=256>
    __global__ void matmul_f32_kernel(float* input, float* weight, float* out, int hidden_dim) {
        const int tid = threadIdx.x;
        const int row = blockIdx.x;

        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        constexpr int NUM_WARPS = NUM_THREADS / 32;

        float sum = 0.0f;
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            sum += input[i] * weight[row * hidden_dim + i];
        }

        for (int mask = 16; mask >= 1; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
        }

        __shared__ float smem[NUM_WARPS];
        if (lane == 0) {
            smem[warp] = sum;
        }
        __syncthreads();
        
        sum = lane < NUM_WARPS ? smem[lane] : 0.0f;

        for (int mask = 16; mask >= 1; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
        }

        if (tid == 0) {
            out[row] = sum;
        }
    }


    void matmul_kernel_cuda(const mem::Tensor& input, const mem::Tensor& weight,
        const mem::Tensor& output, int32_t dim0, int32_t dim1, float scale) {

        if (input.get_dim(0) != dim1) {
            LOG("Tensor with Wrong Dim!");
        }

        float* in1_ptr = const_cast<float*>(input.ptr<float>());
        float* in2_ptr = const_cast<float*>(weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        const int block_size = 256;
        const int grid_size = dim0;

        matmul_f32_kernel<<<grid_size, block_size>>>(in1_ptr, in2_ptr, out_ptr, dim1);
    }
}
