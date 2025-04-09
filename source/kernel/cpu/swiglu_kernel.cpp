#include "swiglu_kernel.h"
#include <math.h>
namespace kernel {
    
    void swiglu_kernel_cpu(const mem::Tensor& up, const mem::Tensor& gate, const mem::Tensor& output, int32_t intermediate_size) {
        const float* up_ptr = up.ptr<float>();
        const float* gate_ptr = gate.ptr<float>();
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        for (int i = 0; i < intermediate_size; i++) {
            float tmp = 0;
            tmp = (1.0f / (1.0f + std::exp(-gate_ptr[i])));
            out_ptr[i] = tmp * up_ptr[i];
        }
    }
}
