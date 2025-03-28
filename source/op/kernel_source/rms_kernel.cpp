#include "rms_kernel.h"
#include "cblas.h"
#include <cmath>
namespace kernel {
    void rmsnorm_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor&output) {
        const float* in_ptr = (input.ptr<float>());
        const float* wei_ptr = (weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        const int32_t dim = static_cast<int32_t>(input.size());
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum_sq += in_ptr[i] * in_ptr[i];
        }
        float tep = sum_sq / static_cast<float>(dim);
        float rms = std::sqrt(tep + 1e-6f);
        float inv_rms = 1.0f / rms;
        for (int i = 0; i < dim; ++i) {
            out_ptr[i] = (in_ptr[i] * inv_rms) * wei_ptr[i];
        }
    }
};