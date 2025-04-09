#include "emb_kernel.cuh"

namespace kernel {
    __global__ void emb_f32_kernel(float* input, float* output, int32_t token, int hidden_dim_size) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int offset = token * hidden_dim_size;
        if (idx < hidden_dim_size) {
            output[idx] = input[idx + offset];
        }
    }
    
    void emb_kernel_cuda(const mem::Tensor& input, const mem::Tensor& weight,
        const mem::Tensor& output, int32_t vocab_size, int32_t hidden_dim_size) {
    
        int32_t token = *input.ptr<int32_t>();
        if (token > vocab_size) {
            LOG("Token index is greater than vocab size.");
        } else {
            float* dst_ptr = const_cast<float*>(output.ptr<float>());
            float* src_ptr = const_cast<float*>(weight.ptr<float>());

            dim3 block(512);
            dim3 grid((hidden_dim_size + 512 - 1) / 512);
            emb_f32_kernel<<<grid, block>>>(src_ptr, dst_ptr, token, hidden_dim_size);
        }
    }
}
