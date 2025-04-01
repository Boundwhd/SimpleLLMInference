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

    void matmul_mha_out_cpu(const mem::Tensor& score_tensor, const mem::Tensor& value_tensor, 
        const mem::Tensor& output_tensor, int32_t pos, int32_t head_size, int32_t dim) {

        const float* score = score_tensor.ptr<float>();
        float* output = const_cast<float*>(output_tensor.ptr<float>());

        for (int i = 0; i <= pos; i++) {
            const float* value = value_tensor.ptr<float>() + dim * i;
            for (int j = 0; j < head_size; j++) {
                output[j] += score[i] * value[j];
            }
        }
    }

    void mha_kernel_cpu(const mem::Tensor& query,
        const mem::Tensor& score, const mem::Tensor& key_cache, const mem::Tensor& value_cache, 
        const mem::Tensor& mha_out, int32_t layer_index, int32_t pos, int32_t seq_len, int32_t dim,
        int32_t head_num,  int32_t head_size, base::DeviceType device_type) {
    
        int32_t layer_offset = layer_index * seq_len * dim;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

        std::shared_ptr<mem::DeviceAllocator> allocator;
        if (device_type == base::DeviceType::kDeviceCPU) {
            allocator = mem::CPUDeviceAllocatorFactory::get_instance();
        } else {
            allocator = mem::CUDADeviceAllocatorFactory::get_instance();
        }

        for (int h = 0; h < head_num; h++) {
            float* score_addr = const_cast<float*>(score.ptr<float>() + h * seq_len);
            float* query_addr = const_cast<float*>(query.ptr<float>() + h * head_size);

            mem::Tensor query_mat({head_size}, false, nullptr, query_addr);
            query_mat.set_device_type(device_type);

            for (int p = 0; p <= pos; p++) {
                int32_t cache_offset = p * dim + h * head_size;
                const float* key_head_addr = key_cache.ptr<float>() + layer_offset + cache_offset;
                mem::Tensor key_mat({1, head_size}, false, nullptr, const_cast<float*>(key_head_addr));
                mem::Tensor score_mat({1}, false, nullptr, score_addr + p);
                matmul_kernel_cpu(query_mat, key_mat, score_mat, scale);
            }

            mem::Tensor score_head_tensor({pos + 1}, false, nullptr, score_addr);
            softmax_kernel_cpu(score_head_tensor, pos + 1);

            float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
            allocator->memset_zero(output_head_ptr, sizeof(float) * head_size);

            mem::Tensor output_tensor({head_size}, false, nullptr, output_head_ptr);
            int32_t cache_offset = h * head_size;
            float* value_head_addr = const_cast<float*>(value_cache.ptr<float>()) + layer_offset + cache_offset;
            mem::Tensor value_tensor({head_size}, false, nullptr, value_head_addr);
            matmul_mha_out_cpu(score_head_tensor, value_tensor, output_tensor, pos, head_size, dim);
        }   
    }
}