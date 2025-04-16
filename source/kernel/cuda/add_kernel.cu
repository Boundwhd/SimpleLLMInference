#include "add_kernel.cuh"

namespace kernel {

    __global__ void add_kernel_f32(float* in1, float* in2, float* out, int dim_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < dim_size) {
            out[idx] = in1[idx] + in2[idx];
        }
    }

    void add_kernel_cuda(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& output, int32_t dim_size) {

        float* input_vec1 = const_cast<float*>(input1.ptr<float>());
        float* input_vec2 = const_cast<float*>(input2.ptr<float>());
        float* output_vec = const_cast<float*>(output.ptr<float>());


        const int block_size = 512;
        const int grid_size = (dim_size + block_size - 1) / block_size;

        add_kernel_f32<<<grid_size, block_size>>>(input_vec1, input_vec2, output_vec, dim_size);
    }
};