#include "mha_kernel.cuh"
#include <cfloat>
#define WARP_SIZE   32
#define THREADS     128
namespace kernel {
    // ----- warp_reduce_sum_f32 -----
    __device__ __forceinline__ float warp_reduce_sum_f32(float val) {
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, mask);
        }
        return val;
    }

    // ----- warp_reduce_max_f32 -----
    __device__ __forceinline__ float warp_reduce_max_f32(float val) {
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
        return val;
    }

    // ----- block_reduce_sum_f32 -----
    template<const int NUM_THREADS>
    __device__ float block_reduce_sum_f32(float sum) {
        int warp = threadIdx.x / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;

        sum = warp_reduce_sum_f32(sum);

        __shared__ float smem[NUM_THREADS/WARP_SIZE];
        if (lane == 0) {
            smem[warp] = sum;
        }
        __syncthreads();

        sum = (lane < NUM_THREADS/WARP_SIZE) ? smem[lane] : 0.0f;
        sum = warp_reduce_sum_f32(sum);

        return sum;
    }

    template<const int NUM_THREADS>
    __device__ float block_reduce_max_f32(float max) {
        int warp = threadIdx.x / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;

        max = warp_reduce_max_f32(max);
        __shared__ float smem[NUM_THREADS/WARP_SIZE];
        if (lane == 0) {
            smem[warp] = max;
        }
        __syncthreads();

        max = (lane < NUM_THREADS/WARP_SIZE) ? smem[lane] : 0.0f;
        max = warp_reduce_max_f32(max);

        return max;
    }

    // step1: score calculation
    __global__ void score_cal_kernel_f32(float* query_ptr, float* key_ptr, float* score_ptr, 
        int32_t pos, int32_t max_seq_len, int32_t att_kv_head_group, int32_t head_dim, int32_t kv_hidden_dim) {
        
        int idx = threadIdx.x;
        int head = blockIdx.x;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        query_ptr += head * head_dim;
        score_ptr += head * max_seq_len;
        key_ptr   += (head / att_kv_head_group) * head_dim;

        float sum = 0.0f;

        for (int i = 0; i <= pos; i++) {
            float* cur_key_ptr = key_ptr + i * kv_hidden_dim;
            float* cur_score_ptr = score_ptr + i;
            
            sum = query_ptr[idx] * cur_key_ptr[idx];
            sum = block_reduce_sum_f32<THREADS>(sum);
            
            if (idx == 0) {
                cur_score_ptr[0] = sum * scale;
            }
        }
    }

    __global__ void safe_softmax_kernel_f32(float* score_ptr, int32_t pos, int32_t max_seq_len) {
        int tid = threadIdx.x;
        int row = blockIdx.x;

        // step 1: get the max of one row
        float max = -FLT_MAX;
        for (int i = tid; i <= pos; i += blockDim.x) {
            max = fmax(max, score_ptr[row * max_seq_len + i]);
        }
        max = block_reduce_max_f32<THREADS>(max);

        // step 2: get the sum of one row
        float exp_sum = 0.0f;
        for (int i = tid; i <= pos; i += blockDim.x) {
            exp_sum += expf(score_ptr[row * max_seq_len + i] - max);
        }
        exp_sum = block_reduce_sum_f32<THREADS>(exp_sum);

        // step 3: get the result
        for (int i = tid; i <= pos; i += blockDim.x) {
            score_ptr[row * max_seq_len + i] = expf(score_ptr[row * max_seq_len + i] - max) / exp_sum;
        }
    }

    __global__ void score_value_mul_kernel_f32(float* score_ptr, float* value_cache, float* mha_out_ptr, 
        int pos, int max_seq_len, int head_dim, int att_kv_head_group, int kv_hidden_dim) {
        int idx = threadIdx.x;
        int head = blockIdx.x;

        score_ptr += head * max_seq_len;
        value_cache += (head / att_kv_head_group) * head_dim;
        mha_out_ptr += head * head_dim;

        float val = 0.0f;   
        for (int i = 0; i <= pos; i++) {
            float* cur_score_ptr = score_ptr + i;
            float* cur_value_ptr = value_cache + i * kv_hidden_dim;

            val += cur_score_ptr[0] * cur_value_ptr[idx];
        }
        mha_out_ptr[idx] = val;
    }


    void mha_kernel_cuda(
        const mem::Tensor& query,
        const mem::Tensor& score, 
        const mem::Tensor& key_cache, 
        const mem::Tensor& value_cache, 
        const mem::Tensor& mha_out, 
        int32_t layer_index, 
        int32_t pos, 
        int32_t max_seq_len, 
        int32_t head_dim,
        int32_t hidden_dim, 
        int32_t kv_hidden_dim, 
        int32_t att_kv_head_group,
        int32_t num_attention_heads,
        base::DeviceType device_type
    ) {
        int layer_offset = layer_index * max_seq_len * kv_hidden_dim;
        
        // kernel1: score && softmax
        float* query_ptr = const_cast<float*>(query.ptr<float>());
        float* key_ptr = const_cast<float*>(key_cache.ptr<float>() + layer_offset);
        float* value_ptr = const_cast<float*>(value_cache.ptr<float>() + layer_offset);
        float* score_ptr = const_cast<float*>(score.ptr<float>());
        float* mha_out_ptr = const_cast<float*>(mha_out.ptr<float>());

        int grid_size = num_attention_heads;
        int block_size = head_dim;  // 确保head_dim <= 1024

        score_cal_kernel_f32<<<grid_size, block_size>>>(
            query_ptr, key_ptr, score_ptr, pos, max_seq_len, att_kv_head_group, head_dim, kv_hidden_dim);
        
        block_size = pos <= 1023 ? pos + 1 : 1024;
        safe_softmax_kernel_f32<<<grid_size, block_size>>>(score_ptr, pos, max_seq_len);

        block_size = head_dim;  // 确保head_dim <= 1024
        score_value_mul_kernel_f32<<<grid_size, block_size>>>(score_ptr, value_ptr, mha_out_ptr, pos, max_seq_len, head_dim, att_kv_head_group, kv_hidden_dim);
    }
}