#include "add_kernel.h"
#include <cblas.h>

namespace kernel {
    void add_kernel_cpu(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& output, int32_t dim_size) {
        float* input_vec1 = const_cast<float*>(input1.ptr<float>());
        float* input_vec2 = const_cast<float*>(input2.ptr<float>());
        float* output_vec = const_cast<float*>(output.ptr<float>());

        auto alloc = input1.get_buffer()->allocator();
        alloc->memcpy(input_vec1, output_vec, dim_size * DataTypeSize, base::MemcpyKind::kMemcpyCPU2CPU);
        
        cblas_saxpy(dim_size, 1.0, input_vec2, 1, output_vec, 1);
    }
}