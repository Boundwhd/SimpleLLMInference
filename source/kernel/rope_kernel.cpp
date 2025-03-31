#include "rope_kernel.h"
#include <math.h>
namespace kernel {
    void rope_cache_cal(int head_size, int max_seq_len, const mem::Tensor sin_cache, const mem::Tensor cos_cache) {
        float* sin_ptr = const_cast<float*>(sin_cache.ptr<float>());
        float* cos_ptr = const_cast<float*>(cos_cache.ptr<float>());

        for (int i = 0; i < max_seq_len; i++) {
            for (int d = 0; d < head_size / 2; d++) {
                int tmp = 2 * d;
                float freq = 1.0f / (std::pow(10000.0f, static_cast<float>(tmp) / static_cast<float>(head_size)));
                float val = freq * i;
                float fcr = cosf(val);
                float fci = sinf(val);
                *(sin_ptr + i * (head_size / 2) + d) = fci;
                *(cos_ptr + i * (head_size / 2) + d) = fcr;
            }
        }
    }


    void rope_kernel_cpu(const mem::Tensor& input_q, const mem::Tensor& input_k, const mem::Tensor& pos_now, 
        const mem::Tensor& sin_cache, const mem::Tensor& cos_cache, int32_t dim, int32_t head_size){
            
        const int32_t position = *(pos_now.ptr<int32_t>(0));

        for (int i = 0; i < dim; i += head_size) {
            for (int head_dim = 0; head_dim < head_size / 2; head_dim++) {
                float fci = *(sin_cache.ptr<float>() + position * head_size / 2 + head_dim);
                float fcr = *(cos_cache.ptr<float>() + position * head_size / 2 + head_dim);

                for (int v = 0; v < 2; v++) {
                    float* vec = const_cast<float*>(v == 0 ? input_q.ptr<float>() : input_k.ptr<float>());
                    float v0 = vec[i + head_dim];
                    float v1 = vec[i + head_dim + head_size / 2];
                    vec[i + head_dim] = v0 * fcr - v1 * fci;
                    vec[i + head_dim + head_size / 2] = v1 * fcr + v0 * fci;
                }
            }
        }
    }
}