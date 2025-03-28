#include "rope_kernel.cuh"

namespace kernel {

    __global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
        const uint pos = blockIdx.x;
        const uint idx = threadIdx.x;

        for (int i = idx; i < head_size / 2; i += blockDim.x) {
            float freq = 1.0f / pow(10000.0f, static_cast<float>(i*2) / static_cast<float>(head_size));
            float val = static_cast<float>(pos) * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            *(sin_cache + pos * (head_size / 2) + i) = fci;
            *(cos_cache + pos * (head_size / 2) + i) = fcr;
        }
    }


    __global__ void rope_kernel_cu(float* query, float* key, const float* sin_cache, const float* cos_cache, 
        int32_t head_size, int32_t pos) {
        int tid = threadIdx.x;
        int head = blockIdx.x;

        for (int i = tid; i < head_size / 2; i += blockDim.x) {
            float fci = *(sin_cache + (pos * head_size / 2) + i);
            float fcr = *(cos_cache + (pos * head_size / 2) + i);
            float* vec = query + (head * head_size);
            float v0 = vec[i];
            float v1 = vec[i + head_size / 2];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + head_size / 2] = v1 * fcr + v0 * fci;

            vec = key + (head * head_size);
            v0 = vec[i];
            v1 = vec[i + head_size / 2];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + head_size / 2] = v1 * fcr + v0 * fci;
        }
    } 

    void rope_cache_cu_cal(int head_size, int max_seq_len, const mem::Tensor sin_cache, const mem::Tensor cos_cache) {
        const uint block_size = 256;
        const uint grid_size = max_seq_len;

        float* sin_ptr = const_cast<float*>(sin_cache.ptr<float>());
        float* cos_ptr = const_cast<float*>(cos_cache.ptr<float>());

        sin_cos_calc<<<grid_size, block_size>>>(head_size, max_seq_len, sin_ptr, cos_ptr);
    }
    
    void rope_kernel_cuda(const mem::Tensor& input_q, const mem::Tensor& input_k, const mem::Tensor& pos_now, 
        const mem::Tensor& sin_cache, const mem::Tensor& cos_cache, int32_t dim, int32_t head_size) {
            
        const int32_t pos = pos_now.index<int32_t>(0);
        const uint block_size = 256;
        const uint grid_size = dim / head_size;

        rope_kernel_cu<<<grid_size, block_size>>>(
            const_cast<float*>(input_q.ptr<float>()),
            const_cast<float*>(input_k.ptr<float>()),
            sin_cache.ptr<float>(),
            cos_cache.ptr<float>(),
            head_size,
            pos
        );
    }
};