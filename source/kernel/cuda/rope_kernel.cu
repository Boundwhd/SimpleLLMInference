#include "rope_kernel.cuh"

namespace kernel {

    __global__ void rope_cache_f32_kernel(float* out1, float* out2, int max_seq_len, int head_size, float rope_theta) {
        int idx = threadIdx.x;
        int row = blockIdx.x;

        int tmp = 2 * idx;
        float freq = 1.0f / (powf(rope_theta, static_cast<float>(tmp) / static_cast<float>(head_size)));
        float val = freq * row;
        float fcr = cosf(val);
        float fci = sinf(val);
        
        out1[row * head_size / 2 + idx] = fci;
        out2[row * head_size / 2 + idx] = fcr;
    }

    __global__ void rope_f32_kernel(float* q, float* k, const float* sin, const float* cos, 
        int32_t pos, int32_t hidden_dim, int32_t head_dim) {
        
        int tid = threadIdx.x;
        int head = blockIdx.x;

        float fci = *(sin + pos * head_dim / 2 + tid);
        float fcr = *(cos + pos * head_dim / 2 + tid);

        for (int i = 0; i < 2; i++) {
            float* vec = (i == 0) ? q : k;
            float v0 = *(vec + head * head_dim + tid);
            float v1 = *(vec + head * head_dim + tid + head_dim / 2);
            vec[head * head_dim + tid] = v0 * fcr - v1 * fci;
            vec[head * head_dim + tid + head_dim / 2] = v1 * fcr + v0 * fci;
        }
    }

    void rope_cache_cal_cuda(int head_size, int max_seq_len, const mem::Tensor sin_cache, const mem::Tensor cos_cache, float rope_theta) {
        const int grid_size = max_seq_len;
        const int block_size = head_size / 2;   // 保证 (head_size/2 < 1024)

        float* sin_ptr = const_cast<float*>(sin_cache.ptr<float>());
        float* cos_ptr = const_cast<float*>(cos_cache.ptr<float>());
        rope_cache_f32_kernel<<<grid_size, block_size>>>(sin_ptr, cos_ptr, max_seq_len, head_size, rope_theta);
    }

    void rope_kernel_cuda(const mem::Tensor& input_q, const mem::Tensor& input_k, const mem::Tensor& pos_now, 
        const mem::Tensor& sin_cache, const mem::Tensor& cos_cache, int32_t hidden_dim_size, int32_t head_dim) {

        const int32_t position = *(pos_now.ptr<int32_t>(0)); 
        const int block_size = head_dim / 2;    // assume that head_dim <= 2048;
        const int grid_size = hidden_dim_size / head_dim;
        
        float* q = const_cast<float*>(input_q.ptr<float>());
        float* k = const_cast<float*>(input_k.ptr<float>());
        const float* sin = sin_cache.ptr<float>();
        const float* cos = cos_cache.ptr<float>();
        rope_f32_kernel<<<grid_size, block_size>>>(q, k, sin, cos, position, hidden_dim_size, head_dim);
    }
}
