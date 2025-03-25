#include "rms_kernel.h"
#include "cblas.h"
#include <cmath>
namespace kernel {
    void rmsnorm_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor&output) {
        const float* in_ptr = (input.ptr<float>());
        const float* wei_ptr = (weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        const int32_t dim = static_cast<int32_t>(input.size());
        float sum_sq = cblas_sdot(dim, in_ptr, 1, in_ptr, 1);
        float rms = std::sqrt(sum_sq / dim + 1e-6f);
        float inv_rms = 1.0f / rms;
        for (int i = 0; i < dim; ++i) {
            out_ptr[i] = (in_ptr[i] * inv_rms) * wei_ptr[i];
        }
    }
};