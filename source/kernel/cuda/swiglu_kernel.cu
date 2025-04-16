#include "swiglu_kernel.cuh"

namespace kernel {
    
    __global__ void swiglu_kernel_f32(float* up, float* gate, float* out, int32_t intermediate_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < intermediate_size) {
            float sig_gate = (1.0f / (1.0f + expf(-gate[idx])));
            out[idx] = sig_gate * up[idx];
        }
        
    }

    void swiglu_kernel_cuda(const mem::Tensor& up, const mem::Tensor& gate, const mem::Tensor& output, int32_t intermediate_size) {

        float* up_ptr = const_cast<float*>(up.ptr<float>());
        float* gate_ptr = const_cast<float*>(gate.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        const int block_size = 512;
        const int grid_size = (intermediate_size + block_size - 1) / block_size;

        swiglu_kernel_f32<<<grid_size, block_size>>>(up_ptr, gate_ptr, out_ptr, intermediate_size);
    }
}
