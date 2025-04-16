#ifndef _MHA_CUDA_H_
#define _MHA_CUDA_H_
#include "tensor.h"

namespace kernel {
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
    );
}

#endif