#include "rms_kernel.h"
#include <cmath>

namespace kernel {
    void rmsnorm_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor&output,
        int32_t hidden_dim_size, float eps) {

        const float* in_ptr = (input.ptr<float>());
        const float* wei_ptr = (weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim_size; ++i) {
            sum_sq += in_ptr[i] * in_ptr[i];
        }
        
        float tep = sum_sq / static_cast<float>(hidden_dim_size);
        float rms = std::sqrt(tep + eps);
        float inv_rms = 1.0f / rms;
        for (int i = 0; i < hidden_dim_size; ++i) {
            out_ptr[i] = (in_ptr[i] * inv_rms) * wei_ptr[i];
        }
    }
};