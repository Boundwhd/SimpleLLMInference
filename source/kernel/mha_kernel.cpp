#include "mha_kernel.h"
#include "math.h"
#include "matmul_kernel.h"
#include <algorithm>

namespace kernel {
    void softmax_kernel_cpu(const mem::Tensor& in, int32_t size) {
        float* in_ptr = const_cast<float*>(in.ptr<float>());
        float max_value = *std::max_element(in_ptr, in_ptr + size);

        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            in_ptr[i] = exp(in_ptr[i] - max_value);
            sum += in_ptr[i];
        }

        for (int i = 0; i < size; i++) {
            in_ptr[i] /= sum;
        }
    }   

    void attention_output_kernel(const mem::Tensor& score, const mem::Tensor& value, const mem::Tensor& output, 
        int32_t pos, int32_t head_dim, int32_t kv_hidden_dim) {

        const float* score_ptr = score.ptr<float>();
        float* output_ptr = const_cast<float*>(output.ptr<float>());

        for (int p = 0; p <= pos; p++) {
            const float* value_ptr = value.ptr<float>() + p * kv_hidden_dim;
            for (int j = 0; j < head_dim; j++) {
                output_ptr[j] += score_ptr[p] * value_ptr[j];
            }
        }
    }

    void mha_kernel_cpu(const mem::Tensor& query, const mem::Tensor& score, const mem::Tensor& key_cache,  const mem::Tensor& value_cache, 
        const mem::Tensor& mha_out, int32_t layer_index, int32_t pos, int32_t max_seq_len, int32_t head_dim, int32_t hidden_dim, 
        int32_t kv_hidden_dim, int32_t att_kv_head_group, int32_t num_attention_heads, base::DeviceType device_type) {
            
        int layer_offset = layer_index * max_seq_len * kv_hidden_dim;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        auto alloc = mem::CPUDeviceAllocatorFactory::get_instance();
        for (int h = 0; h < num_attention_heads; h++) {
            float* score_head_addr = const_cast<float*>(score.ptr<float>() + h * max_seq_len);
            float* query_head_addr = const_cast<float*>(query.ptr<float>() + h * head_dim);

            mem::Tensor query_mat({head_dim}, false, nullptr, query_head_addr);
            query_mat.set_device_type(device_type);

            for (int t = 0; t <= pos; t++) {
                int32_t cache_offset = t * kv_hidden_dim + (h / att_kv_head_group) * head_dim;
                const float* key_head_addr = key_cache.ptr<float>() + layer_index + cache_offset;

                mem::Tensor key_mat({1, head_dim}, false, nullptr, const_cast<float*>(key_head_addr));
                mem::Tensor score_mat({1}, false, nullptr, score_head_addr + t);
                key_mat.set_device_type(device_type);
                score_mat.set_device_type(device_type);
                kernel::matmul_kernel_cpu(query_mat, key_mat, score_mat, 1, head_dim, scale);
            }
            mem::Tensor score_mat({pos + 1}, false, nullptr, const_cast<float*>(score_head_addr));
            score_mat.set_device_type(device_type);
            softmax_kernel_cpu(score_mat, pos + 1);

            float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_dim;
            alloc->memset_zero(output_head_ptr, sizeof(float) * head_dim);
            
            mem::Tensor output_mat({head_dim}, false, nullptr, output_head_ptr);
            output_mat.set_device_type(device_type);

            int32_t cache_offset = (h / att_kv_head_group) * head_dim;
            float* value_head_addr = const_cast<float*>(value_cache.ptr<float>()) + layer_index + cache_offset;
            mem::Tensor value_mat({head_dim}, false, nullptr, value_head_addr);

            attention_output_kernel(score_mat, value_mat, output_mat, pos, head_dim, kv_hidden_dim);
        }
    } 
    
}